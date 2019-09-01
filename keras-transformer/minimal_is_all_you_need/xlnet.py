
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import codecs
import regex as re
import tensorflow as tf
import keras.backend as K

from backend import keras
from backend import keras, initializers, regularizers, constraints

from backend import keras
from backend import backend as K
from tensorflow.python.ops.random_ops import random_shuffle
from backend import keras, initializers, regularizers, constraints
from backend import keras, activations, initializers, regularizers, constraints, TF_KERAS


class RelativePartialMultiHeadSelfAttention(keras.layers.Layer):

    def __init__(self,
                 units,
                 num_head,
                 activation=None,
                 use_bias=False,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativePartialMultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.num_head = num_head
        self.units_head = units // num_head
        self.activation = activation
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel, self.bias = None, None
        self.att_drop_layer = None

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 5,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias',
            )

        if 0.0 < self.attention_dropout < 1.0:
            self.att_drop_layer = keras.layers.Dropout(self.attention_dropout)
        super(RelativePartialMultiHeadSelfAttention, self).build(input_shape)

    def _reshape_to_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size, seq_len, self.num_head, self.units_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * self.num_head, seq_len, self.units_head))

    def _reshape_from_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // self.num_head, self.num_head, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // self.num_head, seq_len, feature_dim * self.num_head))

    def _reshape_mask(self, mask):
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, self.num_head, 1])
        return K.reshape(mask, (-1, seq_len))

    @staticmethod
    def _relative_shift(x, key_len_expected=-1):
        batch_size, q_len, k_len = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2]
        x = K.reshape(x, (batch_size, k_len, q_len))            # (batch * n_head, prev_len + seq_len + 1, seq_len)
        x = x[:, 1:, :]                                         # (batch * n_head, prev_len + seq_len, seq_len)
        x = K.reshape(x, (batch_size, q_len, k_len - 1))        # (batch * n_head, seq_len, prev_len + seq_len)
        x = tf.slice(x, (0, 0, 0), (-1, -1, key_len_expected))  # (batch * n_head, seq_len, key_len_expected)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, mask=None, training=None):
        (inputs, content, memories,
         segment_mat, segment_embed, relatives,
         bias_context, bias_relative, bias_segment,
         permutation) = inputs
        full = K.concatenate([memories, content], axis=1)     # (batch, prev_len + seq_len, units)

        kernel_q = self.kernel[:, :self.units]
        kernel_kv = self.kernel[:, self.units:self.units * 3]
        kernel_r = self.kernel[:, self.units * 3:self.units * 4]
        kernel_o = self.kernel[:, self.units * 4:self.units * 5]

        bias_q, bias_kv, bias_r, bias_o = (None,) * 4
        if self.use_bias:
            bias_q = self.bias[:self.units]
            bias_kv = self.bias[self.units:self.units * 3]
            bias_r = self.bias[self.units * 3:self.units * 4]
            bias_o = self.bias[self.units * 4:self.units * 5]

        w_q = K.dot(inputs, kernel_q)                    # (batch, seq_len, units)
        w_kv = K.dot(full, kernel_kv)                    # (batch, prev_len + seq_len, units * 2)
        w_r = K.dot(relatives, kernel_r)                 # (batch, prev_len + seq_len, units)
        if self.use_bias:
            w_q = K.bias_add(w_q, bias_q)
            w_kv = K.bias_add(w_kv, bias_kv)
            w_r = K.bias_add(w_r, bias_r)
        if self.activation is not None:
            w_q = self.activation(w_q)
            w_kv = self.activation(w_kv)
            w_r = self.activation(w_r)

        w_k = w_kv[:, :, :self.units]                    # (batch, prev_len + seq_len, units)
        w_v = w_kv[:, :, self.units:]                    # (batch, prev_len + seq_len, units)
        batch_size, q_len, k_len = K.shape(inputs)[0], K.shape(w_q)[1], K.shape(w_k)[1]

        w_qc = K.bias_add(w_q, bias_context)
        w_qc = self._reshape_to_batches(w_qc)            # (batch * n_head, seq_len, units_head)
        w_k = self._reshape_to_batches(w_k)              # (batch * n_head, prev_len + seq_len, units_head)
        a_context = K.batch_dot(w_qc, w_k, axes=2)       # (batch * n_head, seq_len, prev_len + seq_len)

        w_qr = K.bias_add(w_q, bias_relative)
        w_qr = self._reshape_to_batches(w_qr)            # (batch * n_head, seq_len, units_head)
        w_r = self._reshape_to_batches(w_r)              # (batch * n_head, prev_len + seq_len, units_head)
        a_relative = K.batch_dot(w_qr, w_r, axes=2)      # (batch * n_head, seq_len, prev_len + seq_len)
        a_relative = self._relative_shift(               # (batch * n_head, seq_len, prev_len + seq_len)
            a_relative,
            key_len_expected=K.shape(a_context)[-1],
        )

        w_qs = K.bias_add(w_q, bias_segment)
        w_qs = K.reshape(w_qs, (-1, q_len, self.num_head, self.units_head))
        w_qs = K.permute_dimensions(w_qs, (2, 0, 1, 3))               # (n_head, batch, seq_len, units_head)
        segment_embed = K.reshape(K.transpose(segment_embed), (self.num_head, 1, self.units_head, 2))
        segment_embed = K.tile(segment_embed, (1, batch_size, 1, 1))
        w_qs = K.reshape(w_qs, (-1, q_len, self.units_head))
        segment_embed = K.reshape(segment_embed, (-1, self.units_head, 2))
        a_segment = K.batch_dot(w_qs, segment_embed, axes=(2, 1))     # (n_head * batch, seq_len, 2)
        a_segment = K.reshape(a_segment, (self.num_head, batch_size, q_len, 2))
        a_segment = K.permute_dimensions(a_segment, (1, 2, 3, 0))     # (batch, seq_len, 2, n_head)
        segment_mat = K.reshape(segment_mat, (-1, k_len, 2))          # (batch * seq_len, prev_len + seq_len, 2)
        a_segment = K.reshape(a_segment, (-1, 2, self.num_head))      # (batch * seq_len, 2, n_head)
        a_segment = K.batch_dot(segment_mat, a_segment, axes=(2, 1))  # (batch * seq_len, prev_len + seq_len, n_head)
        a_segment = K.reshape(a_segment, (-1, q_len, k_len, self.num_head))
        a_segment = K.reshape(K.permute_dimensions(a_segment, (0, 3, 1, 2)), (-1, q_len, k_len))

        att = (a_context + a_relative + a_segment) / K.sqrt(K.constant(self.units_head, dtype=K.floatx()))
        exp = K.exp(att - K.max(att, axis=-1, keepdims=True))

        permutation = K.tile(K.expand_dims(permutation, axis=1), [1, self.num_head, 1, 1])
        permutation = K.reshape(permutation, (-1, q_len, k_len))
        exp *= permutation
        if mask is not None and mask[0] is not None:
            mask = K.cast(mask[0], K.floatx())
            mask = K.concatenate([K.ones_like(memories[:, :, 0]), mask], axis=1)
            exp *= K.expand_dims(self._reshape_mask(mask), axis=1)

        att = exp / (K.sum(exp, axis=-1, keepdims=True) + K.epsilon())
        if self.att_drop_layer is not None:
            att = self.att_drop_layer(att, training=training)
        w_v = self._reshape_to_batches(w_v)                   # (batch * n_head, prev_len + seq_len, units_head)
        w_o = K.batch_dot(att, w_v)                           # (batch * n_head, seq_len, units_head)

        w_o = self._reshape_from_batches(w_o)                 # (batch, seq_len, units)
        w_o = K.dot(w_o, kernel_o)                            # (batch, seq_len, units)
        if self.use_bias:
            w_o = K.bias_add(w_o, bias_o)
        if self.activation is not None:
            w_o = self.activation(w_o)

        if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
            input_shape = K.int_shape(inputs)
            if input_shape[1] is not None:
                w_o = K.reshape(w_o, (-1,) + input_shape[1:])
        return w_o

    def get_config(self):
        config = {
            'units': self.units,
            'num_head': self.num_head,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(RelativePartialMultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


Attention = RelativePartialMultiHeadSelfAttention


class MaskEmbedding(keras.layers.Layer):

    def __init__(self,
                 units,
                 initializer='uniform',
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super(MaskEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(1, 1, self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='embeddings',
        )
        super(MaskEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        output_mask = None
        if mask is not None:
            output_mask = mask[0]
        return output_mask

    def call(self, inputs, **kwargs):
        token_embed, query = inputs
        query = K.expand_dims(K.cast(query, dtype=K.floatx()), axis=-1)
        return query * self.embeddings + (1.0 - query) * token_embed

    def get_config(self):
        config = {
            'units': self.units,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
            'constraint': constraints.serialize(self.constraint),
        }
        base_config = super(MaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PermutationMask(keras.layers.Layer):

    def __init__(self, enabled=True, directional=True, **kwargs):
        super(PermutationMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.enabled = enabled
        self.directional = directional

    def compute_output_shape(self, input_shape):
        input_shape, memory_shape = input_shape
        seq_len = input_shape[1]
        mem_len = memory_shape[1]
        key_len = None
        if mem_len is not None and seq_len is not None:
            key_len = mem_len + seq_len
        return [(input_shape[0], seq_len, key_len)] * 2

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def call(self, inputs, training=None, **kwargs):
        inputs, memory = inputs
        batch_size = K.shape(inputs)[0]
        seq_len = K.shape(inputs)[1]
        mem_mask = K.tile(K.ones_like(memory[:, :, :1], dtype=K.floatx()), [1, 1, seq_len])
        ranges = K.tile(K.expand_dims(K.arange(0, seq_len), axis=-1), [1, batch_size])
        if self.enabled:
            shuffle = random_shuffle(ranges)
        else:
            shuffle = ranges
        if self.directional:
            shuffled = K.in_train_phase(shuffle, ranges, training)
        else:
            if self.enabled:
                shuffled = K.in_train_phase(shuffle, ranges + seq_len, training)
            else:
                shuffled = ranges + seq_len
        ranges = K.expand_dims(K.permute_dimensions(ranges, [1, 0]), axis=-1)
        shuffled = K.expand_dims(K.permute_dimensions(shuffled, [1, 0]), axis=1)
        content_mask = K.cast(ranges <= shuffled, dtype=K.floatx())
        ranges = K.arange(0, seq_len)
        eye = K.equal(K.expand_dims(ranges, axis=0), K.expand_dims(ranges, axis=-1))
        eye = K.expand_dims(K.cast(eye, dtype=K.floatx()), axis=0)
        query_mask = content_mask * (1.0 - eye)
        content_mask = K.concatenate([mem_mask, content_mask], axis=1)
        query_mask = K.concatenate([mem_mask, query_mask], axis=1)
        return [
            K.permute_dimensions(content_mask, [0, 2, 1]),
            K.permute_dimensions(query_mask, [0, 2, 1]),
        ]

    def get_config(self):
        config = {
            'enabled': self.enabled,
            'directional': self.directional,
        }
        base_config = super(PermutationMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SegmentBias(keras.layers.Layer):

    def __init__(self,
                 units,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(SegmentBias, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.bias_context, self.bias_relative = None, None

    def compute_output_shape(self, input_shape):
        return self.units,

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        self.bias_context = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_segment',
        )
        super(SegmentBias, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.identity(self.bias_context)

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(SegmentBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, output_dim, clamp_len=None, directional=True, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.output_dim = output_dim
        self.clamp_len = clamp_len
        self.directional = directional

    def compute_output_shape(self, input_shape):
        input_shape, memory_shape = input_shape
        mem_len = None
        if input_shape[1] is not None and memory_shape[1] is not None:
            mem_len = input_shape[1] + memory_shape[1]
        return input_shape[0],  mem_len, memory_shape[2]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, **kwargs):
        q_len, m_len = K.shape(inputs[0])[1], K.shape(inputs[1])[1]
        k_len = q_len + m_len
        start, stop = k_len, -1
        if not self.directional:
            stop = -q_len
        inputs = K.tile(
            K.expand_dims(K.arange(start, stop, -1, dtype=K.floatx()), axis=0),
            [K.shape(inputs[0])[0], 1],
        )
        if self.clamp_len is not None:
            inputs = K.clip(inputs, min_value=0, max_value=self.clamp_len)
        inputs = K.expand_dims(inputs, axis=-1)
        output_dim = K.cast(self.output_dim, K.floatx())
        ranges = K.expand_dims(K.arange(0.0, self.output_dim, 2.0), axis=0) / output_dim
        inverse = 1.0 / K.pow(10000.0, ranges)
        positions = inputs * inverse
        return K.concatenate([K.sin(positions), K.cos(positions)], axis=-1)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'clamp_len': self.clamp_len,
            'directional': self.directional,
        }
        base_config = super(PositionalEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RelativeSegmentEmbedding(keras.layers.Embedding):

    def __init__(self, units, **kwargs):
        kwargs['input_dim'] = 2
        kwargs['output_dim'] = units
        super(RelativeSegmentEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units

    def compute_output_shape(self, input_shape):
        segment_shape, memory_shape = input_shape
        mem_len = None
        if segment_shape[1] is not None and memory_shape[1] is not None:
            mem_len = segment_shape[1] + memory_shape[1]
        return [(segment_shape[0], segment_shape[1], mem_len, 2), (2, memory_shape[2])]

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def call(self, inputs):
        segment, memory = inputs
        full = K.concatenate([K.zeros_like(memory[:, :, 0]), segment], axis=1)
        relative = K.not_equal(K.expand_dims(segment, axis=-1), K.expand_dims(full, axis=1))
        relative = K.one_hot(K.cast(relative, 'uint8'), 2)
        return [relative, K.identity(self.embeddings)]

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(RelativeSegmentEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CreateMask(keras.layers.Layer):

    def __init__(self, mask_value=0., **kwargs):
        super(CreateMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, self.mask_value)

    def call(self, inputs, **kwargs):
        return K.zeros_like(inputs)

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(CreateMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RemoveMask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        return K.identity(inputs)


class RestoreMask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(RestoreMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def call(self, inputs, **kwargs):
        return K.identity(inputs[0])


def gelu(x): return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class FeedForward(keras.layers.Layer):

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y

class LayerNormalization(keras.layers.Layer):

    def __init__(self, center=True, scale=True, epsilon=None, gamma_initializer='ones', beta_initializer='zeros', gamma_regularizer=None, beta_regularizer=None, gamma_constraint=None, beta_constraint=None, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

class Memory(keras.layers.Layer):

    def __init__(self, batch_size, memory_len, target_len, output_dim, **kwargs):
        super(Memory, self).__init__(**kwargs)
        self.supports_masking = True
        self.stateful = True
        self.batch_size = batch_size
        self.memory_len = memory_len
        self.target_len = target_len
        self.output_dim = output_dim
        self.memory = None

    def build(self, input_shape):
        self.memory = self.add_weight( shape=(self.batch_size, self.memory_len + self.target_len, self.output_dim), initializer='zeros', trainable=False, name='memory', )
        super(Memory, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], None, self.output_dim

    def compute_mask(self, inputs, mask=None):
        if mask is None: return None
        return mask[0]

    def call(self, inputs, **kwargs):
        inputs, memory_length = inputs
        memory_length = K.cast(memory_length[0][0], 'int32')
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')

        # Build new memory
        pad = K.tile(inputs[0:1, ...], (self.batch_size - batch_size, 1, 1))
        padded = K.concatenate([inputs, pad], axis=0)
        new_memory = K.concatenate([self.memory, padded], axis=1)
        new_memory = tf.slice( new_memory, (0, seq_len, 0), (self.batch_size, self.memory_len + self.target_len, self.output_dim), )
        self.add_update(K.update(self.memory, new_memory), inputs)
        old_memory = tf.slice( new_memory, (0, K.maximum(0, self.memory_len + self.target_len - seq_len - memory_length), 0), (batch_size, K.minimum(self.memory_len, memory_length), self.output_dim), )
        return old_memory

    def get_config(self):
        config = { 'batch_size': self.batch_size, 'memory_len': self.memory_len, 'target_len': self.target_len, 'output_dim': self.output_dim, }
        base_config = super(Memory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RelativeBias(keras.layers.Layer):
    def __init__(self, units, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        super(RelativeBias, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_context, self.bias_relative = None, None

    def compute_output_shape(self, input_shape):
        return [(self.units,)] * 2

    def compute_mask(self, inputs, mask=None):
        return [None, None]

    def build(self, input_shape):
        self.bias_context = self.add_weight( shape=(self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=K.floatx(), name='bias_context', )
        self.bias_relative = self.add_weight( shape=(self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, dtype=K.floatx(), name='bias_relative', )
        super(RelativeBias, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return [
            K.identity(self.bias_context),
            K.identity(self.bias_relative),
        ]

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(RelativeBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingRet(keras.layers.Embedding):

    def compute_output_shape(self, input_shape):
        return [
            super(EmbeddingRet, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim),
        ]

    def compute_mask(self, inputs, mask=None):
        return [
            super(EmbeddingRet, self).compute_mask(inputs, mask),
            None,
        ]

    def call(self, inputs):
        return [
            super(EmbeddingRet, self).call(inputs),
            K.identity(self.embeddings),
        ]


class EmbeddingSim(keras.layers.Layer):

    def __init__(self,
                 use_bias=True,
                 initializer='zeros',
                 regularizer=None,
                 constraint=None,
                 stop_gradient=False,
                 **kwargs):

        super(EmbeddingSim, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.stop_gradient = stop_gradient
        self.bias = None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
            'stop_gradient': self.stop_gradient,
        }
        base_config = super(EmbeddingSim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.use_bias:
            embed_shape = input_shape[1]
            token_num = int(embed_shape[0])
            self.bias = self.add_weight(
                shape=(token_num,),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
                name='bias',
            )
        super(EmbeddingSim, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_shape, embed_shape = input_shape
        token_num = embed_shape[0]
        return feature_shape[:-1] + (token_num,)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, mask=None, **kwargs):
        inputs, embeddings = inputs
        if self.stop_gradient:
            embeddings = K.stop_gradient(embeddings)
        outputs = K.dot(inputs, K.transpose(embeddings))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        return keras.activations.softmax(outputs)


def get_custom_objects():
    return {
        'gelu': gelu,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
        'CreateMask': CreateMask,
        'RestoreMask': RestoreMask,
        'PositionalEmbedding': PositionalEmbedding,
        'PermutationMask': PermutationMask,
        'MaskEmbedding': MaskEmbedding,
        'RelativeBias': RelativeBias,
        'SegmentBias': SegmentBias,
        'RelativeSegmentEmbedding': RelativeSegmentEmbedding,
        'Memory': Memory,
        'LayerNormalization': LayerNormalization,
        'RelativePartialMultiHeadSelfAttention': Attention,
        'FeedForward': FeedForward,
    }


def set_custom_objects():
    for key, val in get_custom_objects().items():
        keras.utils.get_custom_objects()[key] = val


def XLNet(units=6, training=True, num_token=31, num_block=2, num_head=2, hidden_dim=12, batch_size=2, memory_len=5, target_len=5, dropout=0.1, attention_dropout=0.1, permute=None, mask_index=5, attention_type='uni', clamp_len=None, shared_biases=True):

    if permute is None:
        permute = training

    token_input = keras.layers.Input(
        shape=(target_len,),
        name='Input-Token',
    )
    seg_input = keras.layers.Input(
        shape=(target_len,),
        name='Input-Segment',
    )
    memory_length_input = keras.layers.Input(
        shape=(1,),
        name='Input-Memory-Length',
    )
    inputs = [token_input, seg_input, memory_length_input]
    if training:
        query_input = keras.layers.Input(
            shape=(target_len,),
            name='Input-Mask',
        )
        inputs.append(query_input)
    else:
        query_input = None
    token_embed, embed_weights = EmbeddingRet(
        input_dim=num_token,
        output_dim=units,
        mask_zero=mask_index == 0,
        name='Embed-Token',
    )(token_input)
    if mask_index is not None and mask_index != 0:
        masking = CreateMask(
            mask_value=mask_index,
            name='Masking',
        )(token_input)
        token_embed = RestoreMask(name='Embed-Token-Masked')([token_embed, masking])
    if training:
        mask_embed = MaskEmbedding(
            units=units,
            name='Embed-Mask'
        )([token_embed, query_input])
    else:
        mask_embed = None
    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(
            rate=dropout,
            name='Embed-Token-Dropout'
        )(token_embed)
        if training:
            mask_embed = keras.layers.Dropout(
                rate=dropout,
                name='Embed-Mask-Dropout'
            )(mask_embed)

    memories = [Memory(
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        output_dim=units,
        name='Memory-0',
    )([token_embed, memory_length_input])]

    pos_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        directional=attention_type == 'uni',
        name='Embed-Pos',
    )([token_embed, memories[0]])

    content_mask, query_mask = PermutationMask(
        enabled=permute,
        directional=attention_type == 'uni',
        name='Permutation',
    )([token_embed, memories[0]])

    context_bias, relative_bias, segment_bias = None, None, None
    if shared_biases:
        context_bias, relative_bias = RelativeBias(
            units,
            name='Relative-Bias',
        )(memories[0])
        segment_bias = SegmentBias(
            units,
            name='Segment-Bias',
        )(memories[0])

    content_output, query_output = token_embed, None
    if training:
        query_output = mask_embed

    for i in range(num_block):
        if not shared_biases:
            context_bias, relative_bias = RelativeBias(
                units,
                name='Relative-Bias-{}'.format(i + 1),
            )(memories[i])
            segment_bias = SegmentBias(
                units,
                name='Segment-Bias-{}'.format(i + 1),
            )(memories[i])

        segment_mat, segment_embed = RelativeSegmentEmbedding(
            units=units,
            name='Embed-Segment-{}'.format(i + 1),
        )([seg_input, memories[i]])

        attention = Attention(
            units=units,
            num_head=num_head,
            use_bias=False,
            attention_dropout=attention_dropout,
            name='Attention-{}'.format(i + 1),
        )
        if 0.0 < dropout < 1.0:
            attention_dropout_layer = keras.layers.Dropout(
                rate=dropout,
                name='Attention-Dropout-{}'.format(i + 1),
            )
        else:
            attention_dropout_layer = None
        attention_add = keras.layers.Add(name='Attention-Residual-{}'.format(i + 1))
        attention_layer_norm = LayerNormalization(name='Attention-Normal-{}'.format(i + 1))

        feed_forward = FeedForward(
            units=hidden_dim,
            dropout_rate=dropout,
            activation=gelu,
            name='FeedForward-{}'.format(i + 1),
        )
        if 0.0 < dropout < 1.0:
            feed_forward_dropout = keras.layers.Dropout(
                rate=dropout,
                name='FeedForward-Dropout-{}'.format(i + 1),
            )
        else:
            feed_forward_dropout = None
        feed_forward_add = keras.layers.Add(name='FeedForward-Residual-{}'.format(i + 1))
        feed_forward_layer_norm = LayerNormalization(name='FeedForward-Normal-{}'.format(i + 1))

        content = content_output

        def _build_output(query, mask):
            attention_input = query
            _output = attention([
                query, content, memories[i],
                segment_mat, segment_embed, pos_embed,
                context_bias, relative_bias, segment_bias,
                mask,
            ])
            if attention_dropout_layer is not None:
                _output = attention_dropout_layer(_output)
            _output = attention_add([attention_input, _output])
            _output = attention_layer_norm(_output)

            feed_forward_input = _output
            _output = feed_forward(_output)
            if feed_forward_dropout is not None:
                _output = feed_forward_dropout(_output)
            _output = feed_forward_add([feed_forward_input, _output])
            _output = feed_forward_layer_norm(_output)
            return _output

        content_output = _build_output(content_output, content_mask)
        if training:
            query_output = _build_output(query_output, query_mask)

        if i < num_block - 1:
            memories.append(Memory(
                batch_size=batch_size,
                memory_len=memory_len,
                target_len=target_len,
                output_dim=units,
                name='Memory-{}'.format(i + 1),
            )([content_output, memory_length_input]))

    if training:
        output = EmbeddingSim(name='Softmax')([query_output, embed_weights])
    else:
        output = content_output
    model = keras.models.Model(
        inputs=inputs,
        outputs=output
    )
    return model





try:
    chr = unichr
except Exception as e:
    '''No need to use `unichr` in Python 3'''


class BytePairEncoding(object):

    def __init__(self,
                 token_dict,
                 bpe_rank):
        """Encode and decode of BPE.

        :param token_dict: Maps from encoded token to indices.
        :param bpe_rank: Maps from byte pair to an integer rank.
        """
        self.token_dict = token_dict
        self.token_dict_inv = {v: k for k, v in self.token_dict.items()}
        self.bpe_rank = bpe_rank
        self.byte_encoder = self.init_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.cache = {}

    @staticmethod
    def init_byte_encoder():
        codes = []
        byte_encoder = {code: chr(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr(2 ** 8 + shift)
                shift += 1
        return byte_encoder

    def get_bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self.bpe_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self.bpe_rank:
                break
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self.cache[token] = chars
        return chars

    def encode(self, text):
        indices = []
        for token in re.findall(self.token_pattern, text):
            token = bytearray(token.encode('utf-8'))
            chars = ''.join(self.byte_encoder[code] for code in token)
            indices += [self.token_dict[token] for token in self.get_bpe(chars)]
        return indices

    def decode(self, tokens):
        text = ''.join([self.token_dict_inv[token] for token in tokens])
        return bytearray([self.byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')


def get_bpe_from_files(encoder_path, vocab_path):
    """Get initialized BPE.

    :param encoder_path: Path to 'encoder.json'.
    :param vocab_path: Path to 'vocab.bpe'
    :return: The object from encode and decode strings.
    """
    with codecs.open(encoder_path, 'r', 'utf8') as reader:
        token_dict = json.load(reader)
    bpe_rank = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        reader.readline()
        for rank, line in enumerate(reader):
            line = line.strip()
            if line:
                bpe_rank[tuple(line.split())] = rank
    return BytePairEncoding(token_dict, bpe_rank)

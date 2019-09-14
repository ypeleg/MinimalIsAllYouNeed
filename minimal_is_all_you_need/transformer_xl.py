import numpy as np
from .backend import keras


from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

import tensorflow as tf
from .backend import keras, activations, initializers, regularizers, constraints, TF_KERAS
from .backend import backend as K


from transformer import LayerNormalization
from transformer import PositionEmbedding
from xlnet import PositionalEmbedding

class RelativePartialMultiHeadSelfAttention(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        units: int >= 0. Dimensions of all tensors.
        num_head: int >= 0. Number of heads. Should divide units.
        use_bias: Boolean. Whether to use bias term.
        attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.

    # Input shape
        First 3D tensor with shape: `(batch_size, sequence_length, units)`.
        Second 3D tensor with shape: `(batch_size, previous_sequence_length + sequence_length, units)`.
        Third 3D tensor with shape: `(batch_size, previous_sequence_length, units)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

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

        self.kernel_q, self.kernel_kv, self.kernel_o, self.kernel_r = (None,) * 4
        self.bias_q, self.bias_kv, self.bias_o, self.bias_r = (None,) * 4
        self.att_drop_layer = None

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def build(self, input_shape):
        self.kernel_q = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_q',
        )
        if self.use_bias:
            self.bias_q = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_q',
            )

        self.kernel_kv = self.add_weight(
            shape=(self.units, self.units * 2),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_kv',
        )
        if self.use_bias:
            self.bias_kv = self.add_weight(
                shape=(self.units * 2,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_kv',
            )

        self.kernel_o = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_o',
        )
        if self.use_bias:
            self.bias_o = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_o',
            )

        self.kernel_r = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_r',
        )
        if self.use_bias:
            self.bias_r = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_r',
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
    def _relative_shift(x):
        batch_size, q_len, k_len = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2]
        x = tf.pad(x, [[0, 0], [0, 0], [1, 0]])               # (batch * n_head, seq_len, prev_len + seq_len + 1)
        x = K.reshape(x, (batch_size, k_len + 1, q_len))      # (batch * n_head, prev_len + seq_len + 1, seq_len)
        x = x[:, 1:, :]                                       # (batch * n_head, prev_len + seq_len, seq_len)
        return K.reshape(x, (batch_size, q_len, k_len))       # (batch * n_head, seq_len, prev_len + seq_len)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, mask=None, training=None):
        inputs, relatives, memories, bias_context, bias_relative = inputs
        full = K.concatenate([memories, inputs], axis=1)      # (batch, prev_len + seq_len, units)
        w_q = K.dot(inputs, self.kernel_q)                    # (batch, seq_len, units)
        w_kv = K.dot(full, self.kernel_kv)                    # (batch, prev_len + seq_len, units * 2)
        w_r = K.dot(relatives, self.kernel_r)                 # (batch, prev_len + seq_len, units)
        if self.use_bias:
            w_q = K.bias_add(w_q, self.bias_q)
            w_kv = K.bias_add(w_kv, self.bias_kv)
            w_r = K.bias_add(w_r, self.bias_r)
        if self.activation is not None:
            w_q = self.activation(w_q)
            w_kv = self.activation(w_kv)
            w_r = self.activation(w_r)

        w_k = w_kv[:, :, :self.units]                         # (batch, prev_len + seq_len, units)
        w_v = w_kv[:, :, self.units:]                         # (batch, prev_len + seq_len, units)

        w_qc = K.bias_add(w_q, bias_context)
        w_qc = self._reshape_to_batches(w_qc)                 # (batch * n_head, seq_len, units_head)
        w_k = self._reshape_to_batches(w_k)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_context = K.batch_dot(w_qc, w_k, axes=2)            # (batch * n_head, seq_len, prev_len + seq_len)

        w_qr = K.bias_add(w_q, bias_relative)
        w_qr = self._reshape_to_batches(w_qr)                 # (batch * n_head, seq_len, units_head)
        w_r = self._reshape_to_batches(w_r)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_relative = K.batch_dot(w_qr, w_r, axes=2)           # (batch * n_head, seq_len, prev_len + seq_len)
        a_relative = self._relative_shift(a_relative)         # (batch * n_head, seq_len, prev_len + seq_len)

        att = (a_context + a_relative) / K.sqrt(K.constant(self.units_head, dtype=K.floatx()))
        exp = K.exp(att - K.max(att, axis=-1, keepdims=True))

        q_len, k_len = K.shape(w_q)[1], K.shape(w_k)[1]
        indices = K.expand_dims(K.arange(0, k_len), axis=0)
        upper = K.expand_dims(K.arange(k_len - q_len, k_len), axis=-1)
        exp *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None and mask[0] is not None:
            mask = K.cast(mask[0], K.floatx())
            mask = K.concatenate([K.ones_like(memories[:, :, 0]), mask], axis=1)
            exp *= K.expand_dims(self._reshape_mask(mask), axis=1)

        att = exp / K.sum(exp, axis=-1, keepdims=True)
        if self.att_drop_layer is not None:
            att = self.att_drop_layer(att, training=training)
        w_v = self._reshape_to_batches(w_v)                   # (batch * n_head, prev_len + seq_len, units_head)
        w_o = K.batch_dot(att, w_v)                           # (batch * n_head, seq_len, units_head)

        w_o = self._reshape_from_batches(w_o)                 # (batch, seq_len, units)
        w_o = K.dot(w_o, self.kernel_o)                       # (batch, seq_len, units)
        if self.use_bias:
            w_o = K.bias_add(w_o, self.bias_o)
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


class RelativeBias(keras.layers.Layer):
    """Relative bias weights.

    # Arguments
        units: int >= 0. Number of hidden units.

    # Input shape
        Any tensor.

    # Output shape
        Two 1D tensors with shape: `(units,)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self,
                 units,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
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
        self.bias_context = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_context',
        )
        self.bias_relative = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_relative',
        )
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


from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

from .backend import keras
from .backend import backend as K

from .backend import keras


import tensorflow as tf
from .backend import keras
from .backend import backend as K



class Memory(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        batch_size: int > 0. Maximum batch size.
        memory_len: int > 0. Maximum memory length.
        target_len: int > 0. Maximum length of targets.
        output_dim: int > 0. Dimension of outputs.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        1D tensor with shape: `(batch_size,)` represents length of memory.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length + memory_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

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
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_len + self.target_len, self.output_dim),
            initializer='zeros',
            trainable=False,
            name='memory',
        )
        super(Memory, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], None, self.output_dim

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, **kwargs):
        inputs, memory_length = inputs
        memory_length = K.cast(memory_length[0][0], 'int32')
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')

        # Build new memory
        pad = K.tile(inputs[0:1, ...], (self.batch_size - batch_size, 1, 1))
        padded = K.concatenate([inputs, pad], axis=0)              # (self.batch_size, seq_len, output_dim)
        new_memory = K.concatenate([self.memory, padded], axis=1)  # (self.batch_size, self.memory_len + seq_len, ...)
        new_memory = tf.slice(                                     # (self.batch_size, self.memory_len, output_dim)
            new_memory,
            (0, seq_len, 0),
            (self.batch_size, self.memory_len + self.target_len, self.output_dim),
        )
        self.add_update(K.update(self.memory, new_memory), inputs)

        # Build output
        old_memory = tf.slice(                                     # (batch_size, memory_length, output_dim)
            new_memory,
            (0, K.maximum(0, self.memory_len + self.target_len - seq_len - memory_length), 0),
            (batch_size, K.minimum(self.memory_len, memory_length), self.output_dim),
        )

        return old_memory

    def get_config(self):
        config = {
            'batch_size': self.batch_size,
            'memory_len': self.memory_len,
            'target_len': self.target_len,
            'output_dim': self.output_dim,
        }
        base_config = super(Memory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Scale(keras.layers.Layer):
    """Scale all weights.

    # Arguments
        scale: float.
    """

    def __init__(self, scale, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.supports_masking = True
        self.scale = scale

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def get_config(self):
        config = {
            'scale': self.scale,
        }
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer.
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

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


class AdaptiveSoftmax(keras.layers.Layer):
    """Turns dense vectors into probabilities.
    # Arguments
        input_dim: int > 0. Dimension of input vectors.
        output_dim: int > 0. Number of output classes.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        use_bias: Boolean. Whether to bias terms.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        bind_embeddings: list of boolean. Whether to use the existed embeddings as mapping.
        bind_projections: list of boolean. Whether to use the existed projections as mapping.
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1, use_bias=True,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 bind_embeddings=False,
                 bind_projections=False,
                 **kwargs):
        super(AdaptiveSoftmax, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = input_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != output_dim:
                self.cutoffs.append(output_dim)
        self.div_val = div_val
        self.use_bias = use_bias
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True
        self.cluster_num = 0
        if self.cutoffs is not None:
            self.cluster_num = len(self.cutoffs) - 2

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.bind_embeddings = bind_embeddings
        if not isinstance(bind_embeddings, list):
            self.bind_embeddings = [bind_embeddings] * (self.cluster_num + 1)
        self.bind_projections = bind_projections
        if not isinstance(bind_projections, list):
            self.bind_projections = [bind_projections] * (self.cluster_num + 1)

        self.embeddings, self.projections, self.biases = (None,) * 3
        self.kernel_cluster, self.bias_cluster = None, None

    def build(self, input_shape):
        if self.div_val == 1:
            if not self.bind_embeddings[0]:
                self.embeddings = self.add_weight(
                    shape=(self.output_dim, self.embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings',
                )
            if self.embed_dim != self.input_dim or self.force_projection:
                if not self.bind_projections[0]:
                    self.projections = self.add_weight(
                        shape=(self.embed_dim, self.input_dim),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        name='kernel',
                    )
            if self.use_bias:
                self.biases = self.add_weight(
                    shape=(self.output_dim,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name='bias',
                )
        else:
            self.kernel_cluster = self.add_weight(
                shape=(self.embed_dim, self.cluster_num),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel-cluster',
            )
            if self.use_bias:
                self.bias_cluster = self.add_weight(
                    shape=(self.cluster_num,),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='bias-cluster',
                )
            self.embeddings, self.projections = [], []
            if self.use_bias:
                self.biases = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if self.bind_embeddings[i]:
                    self.embeddings.append(None)
                else:
                    self.embeddings.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                        initializer=self.embeddings_initializer,
                        regularizer=self.embeddings_regularizer,
                        constraint=self.embeddings_constraint,
                        name='embeddings-{}'.format(i),
                    ))
                if self.bind_projections[i]:
                    self.projections.append(None)
                else:
                    if embed_dim != self.input_dim or self.force_projection:
                        self.projections.append(self.add_weight(
                            shape=(embed_dim, self.input_dim),
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            name='kernel-{}'.format(i),
                        ))
                    else:
                        self.projections.append(None)
                if self.use_bias:
                    self.biases.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i],),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        name='bias-{}'.format(i),
                    ))
        super(AdaptiveSoftmax, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.output_dim,)

    def call(self, inputs, **kwargs):
        embeddings = inputs[1:1 + (self.cluster_num + 1)]
        projections = inputs[1 + (self.cluster_num + 1):]
        inputs = inputs[0]
        if self.div_val == 1:
            if self.embed_dim != self.input_dim or self.force_projection:
                projection = self.projections
                if projection is None:
                    projection = projections[0]
                inputs = K.dot(inputs, K.transpose(projection))
            embedding = self.embeddings
            if embedding is None:
                embedding = embeddings[0]
            out = K.dot(inputs, K.transpose(embedding))
            if self.use_bias:
                out = K.bias_add(out, self.biases)
            out = keras.activations.softmax(out, axis=-1)
        else:
            cluster_probs = None
            outputs = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if embed_dim != self.input_dim or self.force_projection:
                    projection = self.projections[i]
                    if projection is None:
                        projection = projections[i]
                    cluster_input = K.dot(inputs, K.transpose(projection))
                else:
                    cluster_input = inputs
                embedding = self.embeddings[i]
                if embedding is None:
                    embedding = embeddings[i]
                cluster_output = K.dot(cluster_input, K.transpose(embedding))
                if self.use_bias:
                    cluster_output = K.bias_add(cluster_output, self.biases[i])
                if cluster_probs is None:
                    cluster_probs = K.dot(cluster_input, self.kernel_cluster)
                    if self.use_bias:
                        cluster_probs = K.bias_add(cluster_probs, self.bias_cluster)
                    cluster_output = K.concatenate([cluster_output, cluster_probs], axis=-1)
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_probs = cluster_output[..., -self.cluster_num:]
                    cluster_output = cluster_output[..., :-self.cluster_num]
                else:
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_output = cluster_output * K.expand_dims(cluster_probs[..., i - 1])
                outputs.append(cluster_output)
            out = K.concatenate(outputs, axis=-1)

        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'use_bias': self.use_bias,
            'force_projection': self.force_projection,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bind_embeddings': self.bind_embeddings,
            'bind_projections': self.bind_projections,
         }
        base_config = super(AdaptiveSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaptiveEmbedding(keras.layers.Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.
    # Arguments
        input_dim: int > 0. Size of the vocabulary.
        output_dim: int > 0. Dimension of the dense embedding after projection if it is not equal to embed_dim.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to the `embeddings` matrix.
        mask_zero: Whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using [recurrent layers](recurrent.md)
            which may take variable length input.
            If this is `True` then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).
    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 mask_zero=False,
                 return_embeddings=False,
                 return_projections=False,
                 **kwargs):
        super(AdaptiveEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = output_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != input_dim:
                self.cutoffs.append(input_dim)
        self.div_val = div_val
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.return_embeddings = return_embeddings
        self.return_projections = return_projections

        self.embeddings = None
        self.projections = None

    def build(self, input_shape):
        if self.div_val == 1:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.embed_dim),
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                name='embeddings',
            )
            if self.embed_dim != self.output_dim or self.force_projection:
                self.projections = self.add_weight(
                    shape=(self.embed_dim, self.output_dim),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel',
                )
        else:
            self.embeddings, self.projections = [], []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                self.embeddings.append(self.add_weight(
                    shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings-{}'.format(i),
                ))
                projection_shape = (embed_dim, self.output_dim)
                if embed_dim == self.output_dim and not self.force_projection:
                    projection_shape = ()
                self.projections.append(self.add_weight(
                    shape=projection_shape,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel-{}'.format(i),
                ))
        super(AdaptiveEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            output_mask = None
        else:
            output_mask = K.not_equal(inputs, 0)
        if self.return_embeddings or self.return_projections:
            output_mask = [output_mask]
        if self.return_embeddings:
            if self.div_val == 1:
                output_mask += [None]
            else:
                output_mask += [None] * len(self.embeddings)
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_mask += [None]
            else:
                output_mask += [None] * len(self.projections)
        return output_mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape + (self.output_dim,)
        if self.return_embeddings or self.return_projections:
            output_shape = [output_shape]
        if self.return_embeddings:
            if self.div_val == 1:
                output_shape += [K.int_shape(self.embeddings)]
            else:
                output_shape += [K.int_shape(embed) for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_shape += [K.int_shape(self.projections)]
            else:
                output_shape += [K.int_shape(proj) for proj in self.projections]
        return output_shape

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        if self.div_val == 1:
            out = K.gather(self.embeddings, inputs)
            if self.embed_dim != self.output_dim or self.force_projection:
                out = K.dot(out, self.projections)
        else:
            out = K.tile(
                K.expand_dims(K.zeros_like(inputs, dtype=K.floatx()), axis=-1),
                (1,) * K.ndim(inputs) + (self.output_dim,),
            )
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                low, high = self.cutoffs[i], self.cutoffs[i + 1]
                mask = K.cast(low <= inputs, K.floatx()) * K.cast(inputs < high, K.floatx())
                selected = K.gather(self.embeddings[i], (inputs - low) * K.cast(mask, 'int32'))
                if embed_dim != self.output_dim or self.force_projection:
                    projected = K.dot(selected, self.projections[i])
                else:
                    projected = selected
                out += projected * K.expand_dims(mask, axis=-1)
        if self.return_embeddings or self.return_projections:
            out = [out]
        if self.return_embeddings:
            if self.div_val == 1:
                out += [self.embeddings]
            else:
                out += [K.identity(embed) for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    out += [self.projections]
            else:
                out += [K.identity(proj) for proj in self.projections]
        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'force_projection': self.force_projection,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'mask_zero': self.mask_zero,
            'return_embeddings': self.return_embeddings,
            'return_projections': self.return_projections,
         }
        base_config = super(AdaptiveEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def TransformerXL(      units=6,
                        embed_dim=16,
                        hidden_dim=12,
                        num_token=13,
                        num_block=1,
                        num_head=2,
                        batch_size=2,
                        memory_len=0,
                        target_len=3,
                        dropout=0.0,
                        attention_dropout=0.0,
                        cutoffs=[3],
                        div_val=2,
                        force_projection=None,
                        bind_embeddings=True,
                        bind_projections=True,
                        clamp_len=None,
                        share_biases=True):
    """Build transformer-XL model.

    :param units: Units inside the transformer.
    :param embed_dim: Dimension of embeddings.
    :param hidden_dim: Dimension inside position-wise feed-forward layer.
    :param num_token: Number of distinct input tokens.
    :param num_block: Number of basic encoder blocks.
    :param num_head: Number of heads for attention.
    :param batch_size: Maximum batch size.
    :param memory_len: The maximum length of memories.
    :param target_len: The length of prediction block.
    :param dropout: General dropout rate.
    :param attention_dropout: Dropout rate inside attention layer.
    :param cutoffs: Cutoffs of adaptive embedding.
    :param div_val: Scale factor of adaptive embedding.
    :param force_projection: Add projection when the dimensions are equal.
    :param bind_embeddings: Whether to bind embeddings to adaptive softmax.
    :param bind_projections: Whether to bind projections to adaptive softmax.
    :param clamp_len: The maximum value of relative position.
    :param share_biases: Whether to use the same biases for all layers.
    :return: The built model.
    """
    token_input = keras.layers.Input(shape=(target_len,), name='Input-Token')
    memory_length_input = keras.layers.Input(shape=(1,), name='Input-Memory-Length')
    inputs = [token_input, memory_length_input]

    results = AdaptiveEmbedding(
        input_dim=num_token,
        output_dim=units,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        mask_zero=True,
        force_projection=force_projection,
        return_embeddings=True,
        return_projections=True,
        name='Embed-Token',
    )(token_input)
    token_embed, embedding_weights = results[0], results[1:]
    token_embed = Scale(scale=np.sqrt(units), name='Embed-Token-Scaled')(token_embed)
    last_memory = Memory(
        batch_size=batch_size,
        memory_len=memory_len,
        target_len=target_len,
        output_dim=units,
        name='Memory-0',
    )([token_embed, memory_length_input])

    position_embed = PositionalEmbedding(
        output_dim=units,
        clamp_len=clamp_len,
        name='Embed-Position',
    )([token_input, last_memory])

    if 0.0 < dropout < 1.0:
        token_embed = keras.layers.Dropout(rate=dropout, name='Embed-Token-Dropped')(token_embed)
        position_embed = keras.layers.Dropout(rate=dropout, name='Embed-Position-Dropped')(position_embed)

    context_bias, relative_bias = None, None
    if share_biases:
        context_bias, relative_bias = RelativeBias(units=units, name='Biases')(last_memory)

    outputs = [token_embed]
    for i in range(num_block):
        block_input, block_output = outputs[-1], outputs[-1]
        if not share_biases:
            context_bias, relative_bias = RelativeBias(units=units, name='Biases-{}'.format(i + 1))(last_memory)
        """
        block_output = RelativePartialMultiHeadSelfAttention(
            units=units,
            num_head=num_head,
            use_bias=False,
            attention_dropout=attention_dropout,
            name='Attention-{}'.format(i + 1),
        )([block_output, position_embed, last_memory, context_bias, relative_bias])
        """
        block_output = block_input # keras.layers.Add(name='Attention-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='Attention-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='Attention-Norm-{}'.format(i + 1))(block_output)

        block_input = block_output
        block_output = FeedForward(
            units=hidden_dim,
            dropout_rate=dropout,
            name='FeedForward-{}'.format(i + 1),
        )(block_output)
        block_output = keras.layers.Add(name='FeedForward-Res-{}'.format(i + 1))([block_input, block_output])
        if 0.0 < dropout < 1.0:
            block_output = keras.layers.Dropout(rate=dropout, name='FeedForward-Dropped-{}'.format(i + 1))(block_output)
        block_output = LayerNormalization(name='FeedForward-Norm-{}'.format(i + 1))(block_output)

        if i < num_block - 1:
            last_memory = Memory(
                batch_size=batch_size,
                memory_len=memory_len,
                target_len=target_len,
                output_dim=units,
                name='Memory-{}'.format(i + 1),
            )([block_output, memory_length_input])

        outputs.append(block_output)

    softmax = AdaptiveSoftmax(
        input_dim=units,
        output_dim=num_token,
        embed_dim=embed_dim,
        cutoffs=cutoffs,
        div_val=div_val,
        force_projection=force_projection,
        bind_embeddings=bind_embeddings,
        bind_projections=bind_projections,
        name='Softmax',
    )(outputs[-1:] + embedding_weights)

    model = keras.models.Model(inputs=inputs, outputs=softmax)
    return model

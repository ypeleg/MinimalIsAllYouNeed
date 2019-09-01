
import numpy as np

from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


class _BaseMultiHeadAttention(Layer):

    def __init__(self, num_heads, use_masking, dropout = 0.0, compression_window_size = None, **kwargs):

        self.num_heads = num_heads
        self.use_masking = use_masking
        self.dropout = dropout
        if (compression_window_size is not None and compression_window_size <= 0): pass
        self.compression_window_size = compression_window_size
        super(_BaseMultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['use_masking'] = self.use_masking
        config['dropout'] = self.dropout
        config['compression_window_size'] = self.compression_window_size
        return config

    def build_output_params(self, d_model):
        self.output_weights = self.add_weight( name='output_weights', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        if self.compression_window_size is not None:
            self.k_conv_kernel = self.add_weight( name='k_conv_kernel', shape=(self.compression_window_size, d_model // self.num_heads, d_model // self.num_heads), initializer='glorot_uniform', trainable=True)
            self.k_conv_bias = self.add_weight( name='k_conv_bias', shape=(d_model // self.num_heads,), initializer='zeros', trainable=True)
            self.v_conv_kernel = self.add_weight( name='v_conv_kernel', shape=(self.compression_window_size, d_model // self.num_heads, d_model // self.num_heads), initializer='glorot_uniform', trainable=True)
            self.v_conv_bias = self.add_weight( name='v_conv_bias', shape=(d_model // self.num_heads,), initializer='zeros', trainable=True)

    def validate_model_dimensionality(self, d_model):
        if d_model % self.num_heads != 0: raise  Exception

    def attention(self, pre_q, pre_v, pre_k, out_seq_len, d_model, training=None):

        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        if self.compression_window_size is None: k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        else:
            if self.use_masking:
                raise NotImplementedError( "Masked memory-compressed attention has not been implemented yet")
            k = K.permute_dimensions(pre_k, [0, 2, 1, 3])
            k, v = [ K.reshape(
                    K.bias_add(
                        K.conv1d(
                            K.reshape(
                                item,
                                (-1,
                                 K.int_shape(item)[-2],
                                 d_model // self.num_heads)),
                            kernel,
                            strides=self.compression_window_size,
                            padding='valid', data_format='channels_last'),
                        bias,
                        data_format='channels_last'),
                    K.concatenate([
                        K.shape(item)[:2],
                        [-1, d_model // self.num_heads]]))
                for item, kernel, bias in (
                    (k, self.k_conv_kernel, self.k_conv_bias),
                    (v, self.v_conv_kernel, self.v_conv_bias))]
            k_transposed = K.permute_dimensions(k, [0, 1, 3, 2])
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads), dtype=K.floatx())
        q_shape = K.int_shape(q)
        k_t_shape = K.int_shape(k_transposed)
        v_shape = K.int_shape(v)
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    K.softmax(
                        self.mask_attention_if_needed(
                            K.batch_dot(
                                K.reshape(q, (-1,) + q_shape[-2:]),
                                K.reshape(k_transposed,
                                          (-1,) + k_t_shape[-2:]))
                            / sqrt_d)),
                    training=training),
                K.reshape(v, (-1,) + v_shape[-2:])),
            (-1, self.num_heads, q_shape[-2], v_shape[-1]))
        attention_heads_merged = K.reshape(
            K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
            (-1, d_model))
        attention_out = K.reshape(
            K.dot(attention_heads_merged, self.output_weights),
            (-1, out_seq_len, d_model))
        return attention_out

    def apply_dropout_if_needed(self, attention_softmax, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(attention_softmax, self.dropout)
            return K.in_train_phase(dropped_softmax, attention_softmax, training=training)
        return attention_softmax

    def mask_attention_if_needed(self, dot_product):
        if not self.use_masking: return dot_product
        last_dims = K.int_shape(dot_product)[-2:]
        low_triangle_ones = ( np.tril(np.ones(last_dims)).reshape((1,) + last_dims))
        inverse_low_triangle = 1 - low_triangle_ones
        close_to_negative_inf = -1e9
        result = ( K.constant(low_triangle_ones, dtype=K.floatx()) * dot_product + K.constant(close_to_negative_inf * inverse_low_triangle))
        return result


class MultiHeadAttention(_BaseMultiHeadAttention):
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise ValueError( 'You must call this layer passing a list of two tensors (for keys/values and queries)')
        values_dim, query_dim = input_shape[0][-1], input_shape[1][-1]
        if query_dim != values_dim: pass
        d_model = query_dim
        self.validate_model_dimensionality(d_model)
        self.kv_weights = self.add_weight( name='kv_weights', shape=(d_model, d_model * 2), initializer='glorot_uniform', trainable=True)
        self.q_weights = self.add_weight( name='q_weights', shape=(d_model, d_model), initializer='glorot_uniform', trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not (isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError( 'You can call this layer only with a list of two tensors (for keys/values and queries)')
        key_values_input, query_input = inputs
        _, value_seq_len, d_model = K.int_shape(key_values_input)
        query_seq_len = K.int_shape(inputs[1])[-2]
        kv = K.dot(K.reshape(key_values_input, [-1, d_model]), self.kv_weights)
        pre_k, pre_v = [
            K.reshape(
                kv[:, i * d_model: (i + 1) * d_model],
                (-1, value_seq_len,
                 self.num_heads, d_model // self.num_heads))
            for i in range(2)]
        pre_q = K.reshape(
            K.dot(K.reshape(query_input, [-1, d_model]), self.q_weights),
            (-1, query_seq_len, self.num_heads, d_model // self.num_heads))
        return self.attention(pre_q, pre_v, pre_k, query_seq_len, d_model, training=kwargs.get('training'))


class MultiHeadSelfAttention(_BaseMultiHeadAttention):
    def build(self, input_shape):
        if not isinstance(input_shape, tuple): raise ValueError('Invalid input')
        d_model = input_shape[-1]
        self.validate_model_dimensionality(d_model)
        self.qkv_weights = self.add_weight( name='qkv_weights', shape=(d_model, d_model * 3), initializer='glorot_uniform', trainable=True)
        self.build_output_params(d_model)
        return super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError('The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        qkv = K.dot(K.reshape(inputs, [-1, d_model]), self.qkv_weights)
        pre_q, pre_k, pre_v = [
            K.reshape(
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model, training=kwargs.get('training'))
        return attention_out

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'MultiHeadAttention': MultiHeadAttention,
})

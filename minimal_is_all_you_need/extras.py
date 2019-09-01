
import math

from keras.engine import Layer
from keras import backend as K
from keras.layers import Embedding
from keras.utils import get_custom_objects
from keras import activations, regularizers


class ReusableEmbedding(Embedding):

    def call(self, inputs, **kwargs):
        result = super(ReusableEmbedding, self).call(inputs, **kwargs)
        return [result, self.embeddings]

    def compute_output_shape(self, input_shape):
        return [super(ReusableEmbedding, self).compute_output_shape(input_shape),
                K.int_shape(self.embeddings)]

    def compute_mask(self, inputs, mask=None):
        return [super(ReusableEmbedding, self).compute_mask(inputs, mask), None]


class TiedOutputEmbedding(Layer):

    def __init__(self, activation=None, add_biases=False, projection_regularizer=None, projection_dropout = 0.0, scaled_attention=False, **kwargs):
        self.activation = activations.get(activation)
        self.add_biases = add_biases
        self.projection_regularizer = regularizers.get(projection_regularizer)
        self.projection_dropout = projection_dropout
        self.scaled_attention = scaled_attention
        super(TiedOutputEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return dict( config, activation=activations.serialize(self.activation), add_biases=self.add_biases, projection_regularizer=regularizers.serialize( self.projection_regularizer), projection_dropout=self.projection_dropout, scaled_attention=self.scaled_attention)

    def build(self, input_shape):
        main_input_shape, embedding_matrix_shape = input_shape
        emb_input_dim, emb_output_dim = embedding_matrix_shape
        assert len(main_input_shape) == 3
        self.projection = self.add_weight( name='kernel', shape=(main_input_shape[-1], emb_output_dim), initializer='glorot_uniform', regularizer=self.projection_regularizer, trainable=True)
        if self.add_biases: self.biases = self.add_weight( name='biases', shape=(emb_output_dim,), initializer='zeros', trainable=True)
        return super(TiedOutputEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        main_input, embedding_matrix = inputs
        input_shape_tensor = K.shape(main_input)
        last_input_dim = K.int_shape(main_input)[-1]
        emb_input_dim, emb_output_dim = K.int_shape(embedding_matrix)
        projected = K.dot(K.reshape(main_input, (-1, last_input_dim)), self.projection)
        if self.add_biases:
            projected = K.bias_add(projected, self.biases, data_format='channels_last')
        if 0 < self.projection_dropout < 1:
            projected = K.in_train_phase( lambda: K.dropout(projected, self.projection_dropout), projected, training=kwargs.get('training'))
        attention = K.dot(projected, K.transpose(embedding_matrix))
        if self.scaled_attention:
            sqrt_d = K.constant(math.sqrt(emb_output_dim), dtype=K.floatx())
            attention = attention / sqrt_d
        result = K.reshape( self.activation(attention), (input_shape_tensor[0], input_shape_tensor[1], emb_input_dim))
        return result

    def compute_output_shape(self, input_shape):
        main_input_shape, embedding_matrix_shape = input_shape
        emb_input_dim, emb_output_dim = embedding_matrix_shape
        return main_input_shape[0], main_input_shape[1], emb_input_dim


get_custom_objects().update({
    'ReusableEmbedding': ReusableEmbedding,
    'TiedOutputEmbedding': TiedOutputEmbedding,
})

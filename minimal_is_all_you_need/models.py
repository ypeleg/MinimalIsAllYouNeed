

from backend import keras
from keras import regularizers
from keras.models import Model
from keras import backend as K

from xlnet import LayerNormalization
from xlnet import PositionalEmbedding
from xlnet import EmbeddingRet, EmbeddingSim
from xlnet import get_custom_objects as get_transformer_custom_objects
from transformer import gelu, attention_builder, feed_forward_builder, PositionEmbedding
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense

from minimal_is_all_you_need.extras import ReusableEmbedding, TiedOutputEmbedding
from minimal_is_all_you_need.position import TransformerCoordinateEmbedding
from minimal_is_all_you_need.transformer import TransformerACT, TransformerBlock


def GPT( max_seq_length=100, vocabulary_size=100, word_embedding_size=100, transformer_depth=5, num_heads=10, transformer_dropout = 0.1, embedding_dropout = 0.6, l2_reg_penalty = 1e-6, confidence_penalty_weight = 0.1):
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    embedding_layer = ReusableEmbedding( vocabulary_size, word_embedding_size, input_length=max_seq_length, name='bpe_embeddings', embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding( projection_regularizer=l2_regularizer, projection_dropout=embedding_dropout, name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding( transformer_depth, name='coordinate_embedding')
    transformer_act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = TransformerBlock( name='transformer', num_heads=num_heads, residual_dropout=transformer_dropout, attention_dropout=transformer_dropout, use_masking=True, vanilla_wiring=False)
    output_softmax_layer = Softmax(name='word_predictions')
    next_step_input, embedding_matrix = embedding_layer(word_ids)
    act_output = next_step_input
    for i in range(transformer_depth):
        next_step_input = coordinate_embedding_layer(next_step_input, step=i)
        next_step_input = transformer_block(next_step_input)
        next_step_input, act_output = transformer_act_layer(next_step_input)
    transformer_act_layer.finalize()
    next_step_input = act_output
    word_predictions = output_softmax_layer( output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    confidence_penalty = K.mean( confidence_penalty_weight * K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def Transformer( max_seq_length=100, vocabulary_size=100, word_embedding_size=100, transformer_depth=5, num_heads=10, transformer_dropout = 0.1, embedding_dropout = 0.6, l2_reg_penalty = 1e-6, confidence_penalty_weight = 0.1):
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    embedding_layer = ReusableEmbedding( vocabulary_size, word_embedding_size, input_length=max_seq_length, name='bpe_embeddings', embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding( projection_regularizer=l2_regularizer, projection_dropout=embedding_dropout, name='word_prediction_logits')
    coordinate_embedding_layer = TransformerCoordinateEmbedding( 1, name='coordinate_embedding')
    output_softmax_layer = Softmax(name='word_predictions')
    next_step_input, embedding_matrix = embedding_layer(word_ids)
    next_step_input = coordinate_embedding_layer(next_step_input, step=0)
    for i in range(transformer_depth):
        next_step_input = ( TransformerBlock( name='transformer' + str(i), num_heads=num_heads, residual_dropout=transformer_dropout, attention_dropout=transformer_dropout, use_masking=True, vanilla_wiring=True)(next_step_input))
    word_predictions = output_softmax_layer( output_layer([next_step_input, embedding_matrix]))
    model = Model(inputs=[word_ids], outputs=[word_predictions])
    confidence_penalty = K.mean( confidence_penalty_weight * K.sum(word_predictions * K.log(word_predictions), axis=-1))
    model.add_loss(confidence_penalty)
    return model


def Bert(max_seq_length=100, vocabulary_size=100, word_embedding_size=100, use_universal_transformer = 0, transformer_depth=5, num_heads=10, transformer_dropout = 0.1, embedding_dropout = 0.6, l2_reg_penalty = 1e-4):
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    segment_ids = Input( shape=(max_seq_length,), dtype='int32', name='segment_ids')
    l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    embedding_layer = ReusableEmbedding(vocabulary_size, word_embedding_size, input_length=max_seq_length, name='bpe_embeddings', embeddings_regularizer=l2_regularizer)
    segment_embedding_layer = Embedding(word_embedding_size, max_seq_length, name='segment_embeddings')
    add_segment_layer = Add(name='add_segment')
    output_layer = TiedOutputEmbedding(projection_regularizer=l2_regularizer, projection_dropout=embedding_dropout, name='word_prediction_logits')
    output_softmax_layer = Softmax(name='word_predictions')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(transformer_depth if use_universal_transformer else 1, name='coordinate_embedding')
    next_step_input, embedding_matrix = embedding_layer(word_ids)
    segment_embeddings = segment_embedding_layer(segment_ids)
    if use_universal_transformer:
        act_layer = TransformerACT(name='adaptive_computation_time')
        transformer_block = TransformerBlock( name='transformer', num_heads=num_heads, residual_dropout=transformer_dropout, attention_dropout=transformer_dropout, use_masking=False)
        act_output = next_step_input
        for i in range(transformer_depth):
            next_step_input = coordinate_embedding_layer(next_step_input, step=i)
            next_step_input = add_segment_layer([next_step_input, segment_embeddings])
            next_step_input = transformer_block(next_step_input)
            next_step_input, act_output = act_layer(next_step_input)
        act_layer.finalize()
        next_step_input = act_output
    else:
        next_step_input = coordinate_embedding_layer(next_step_input, step=0)
        next_step_input = add_segment_layer([next_step_input, segment_embeddings])
        for i in range(transformer_depth):
            next_step_input = (TransformerBlock( name='transformer' + str(i), num_heads=num_heads, residual_dropout=transformer_dropout, attention_dropout=transformer_dropout, use_masking=False, vanilla_wiring=True)(next_step_input))
    word_predictions = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
    cls_node_slice = (Lambda(lambda x: x[:, 0], name='cls_node_slicer')(next_step_input))
    class_prediction = (Dense(1, name='class_prediction', activation='sigmoid')(cls_node_slice))
    model = Model(inputs=[word_ids, segment_ids], outputs=[word_predictions, class_prediction])
    return model


def GPT_2(n_vocab=100, n_ctx=1024, n_embd=768, n_head=12, n_layer=12, batch_size=None, fixed_input_shape=False):
    if fixed_input_shape: input_layer_shape = (batch_size, n_ctx)
    else: input_layer_shape = (batch_size, None)
    input_layer = keras.layers.Input( batch_shape=input_layer_shape, name='Input', )
    embed_token, embeddings = EmbeddingRet( input_dim=n_vocab, output_dim=n_embd, mask_zero=False, name='Embed-Token', )(input_layer)
    embed_token_pos = PositionEmbedding( input_dim=n_ctx, output_dim=n_embd, mode=PositionEmbedding.MODE_ADD, name='Embed-Token-Pos', )(embed_token)
    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = _get_encoder_component( name='Encode-%d' % i, input_layer=last_layer, head_num=n_head, hidden_dim=n_embd * 4, attention_activation=None, feed_forward_activation=gelu, )
    norm_layer = LayerNormalization( name='Norm', )(last_layer)
    output_layer = EmbeddingSim( use_bias=False, name='Output', )([norm_layer, embeddings])
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile( optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, )
    return model

def GPT_3(n_vocab=100, n_ctx=1024, n_embd=128, n_head=96, n_layer=96, batch_size=None, fixed_input_shape=False):
    if fixed_input_shape: input_layer_shape = (batch_size, n_ctx)
    else: input_layer_shape = (batch_size, None)
    input_layer = keras.layers.Input( batch_shape=input_layer_shape, name='Input', )
    embed_token, embeddings = EmbeddingRet( input_dim=n_vocab, output_dim=n_embd, mask_zero=False, name='Embed-Token', )(input_layer)
    embed_token_pos = PositionEmbedding( input_dim=n_ctx, output_dim=n_embd, mode=PositionEmbedding.MODE_ADD, name='Embed-Token-Pos', )(embed_token)
    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = _get_encoder_component( name='Encode-%d' % i, input_layer=last_layer, head_num=n_head, hidden_dim=n_embd * 4, attention_activation=None, feed_forward_activation=gelu, )
    norm_layer = LayerNormalization( name='Norm', )(last_layer)
    output_layer = EmbeddingSim( use_bias=False, name='Output', )([norm_layer, embeddings])
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile( optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, )
    return model

def _wrap_layer(name, input_layer, build_func, trainable=True):
    normal_layer = LayerNormalization( trainable=trainable, name='%s-Norm' % name, )(input_layer)
    build_output = build_func(normal_layer)
    return keras.layers.Add(name='%s-Add' % name)([input_layer, build_output])

def _get_encoder_component(name, input_layer, head_num, hidden_dim, attention_activation=None, feed_forward_activation='relu', trainable=True):
    attention_name = '%s-MultiHeadAtt' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer( name=attention_name, input_layer=input_layer, build_func=attention_builder( name=attention_name, head_num=head_num, activation=attention_activation, history_only=True, trainable=trainable, ), trainable=trainable, )
    feed_forward_layer = _wrap_layer( name=feed_forward_name, input_layer=attention_layer, build_func=feed_forward_builder( name=feed_forward_name, hidden_dim=hidden_dim, activation=feed_forward_activation, trainable=trainable, ), trainable=trainable, )
    return feed_forward_layer

def get_custom_objects():
    custom_objects = get_transformer_custom_objects()
    custom_objects['gelu'] = gelu
    custom_objects['PositionEmbedding'] = PositionEmbedding
    return custom_objects



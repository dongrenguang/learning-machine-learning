# p70
from tensorflow.keras.layers import Layer, Dense, Embedding, Concatenate, Reshape, Add, Subtract, Lambda

def build_fm_first_order(inputs):
    embeddings = []
    for x in inputs:
        embedding = Embedding(input_dim=5000, output_dim=1)(x)
        embedding = Reshape(target_shape=(1,))(embedding)
        embeddings.append(embedding)
    return Add()(embeddings)

class SumLayer(Layer):
    def __init__(slef, **kwargs):
        super(SumLayer, self).__ini__(**kwargs)
    def call(self, inputs):
        inputs = K.expand_dims(inputs)
        return K.sum(inputs, axis=1)
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], 1])

def build_fm_second_order(embeddings):
    # 先相加后平方
    sum_square_result = Lambda(lambda x: x ** 2)(Add()(embeddings))
    # 先平方后相加
    square_sum_result = Add()([Lambda(lambda x: x ** 2)(emb) for emb in embeddings])
    # 两部分相减
    substract_result = Lambda(lambda x: x * 0.5)(Subtract()([sum_square_result, square_sum_result]))
    return SumLayer()(substract_result)

def build_dnn(embeddings):
    x_deep = Dense(units=1024, activation="relu")(embeddings)
    x_deep = Dense(units=512, activation="relu")(x_deep)
    deep_out = Dense(units=256, activation="relu")(x_deep)
    return deep_out

def build_deep_fm(inputs):
    fm_first_order = build_fm_first_order(inputs)

    emb_size = 32
    embeddings = []
    for x in inputs:
        embedding = Embedding(input_dim=5000, output_dim=emb_size)(x)
        embedding = Reshape(target_shape=(emb_size,))(embedding)
        embeddings.append(embedding)

    fm_second_order = build_fm_second_order(embeddings)

    dnn_output = build_dnn(embeddings)

    fm_output = Add()([fm_first_order, fm_second_order])

    output = Dense(units=1, activation="sigmoid")([fm_output, dnn_output])
    return output

# p66
from tensorflow.keras.layers import Dense, Concatenate, Reshape, Embedding

def get_wdl_output(inputs):
    wide_inputs, deep_category_inputs, deep_continue_inputs = inputs

    wide_out = Dense(units=1, activation=None, use_bias=True)(wide_inputs)

    emb_size = 32
    category_embeddings = []
    for x in deep_category_inputs:
        embedding = Embedding(input_dim=5000, output_dim=emb_size)(x)
        embedding = Reshape(target_shape=(emb_size,))(embedding)
        category_embeddings.append(embedding)

    continue_features = []
    for x in deep_continue_inputs:
        feature = Reshape(target_shape=(1,))(x)
        continue_features.append(feature)

    x_deep = Concatenate()(category_embeddings + continue_features)

    x_deep = Dense(units=1024, activation="relu")(x_deep)
    x_deep = Dense(units=512, activation="relu")(x_deep)
    deep_out = Dense(units=256, activation="relu")(x_deep)

    output = Dense(units=1, activation="sigmoid")([wide_out, deep_out])
    return output

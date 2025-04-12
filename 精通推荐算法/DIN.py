# p98
def build_din_attention_output(queries, keys, keys_length):
    """
    @param queries: 广告特征，广告商品各特征的Embedding拼接后的向量，形状[B, H]，B表示一个batch的大小，H表示向量长度
    @param keys: 用户行为序列特征，用户点击过的商品的Embedding拼接后的向量，形状[B, T, H]。T表示包括padding的序列长度
    @param keys_length: 用户行为序列实际的长度，不包括padding
    """
    # 1. 扩充广告特征queries维度，使其与keys相同，方便后续计算
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units]) # [B, T, H]

    # 2. 构建激活函数，由广告特征、用户行为序列特征、二者差和二者元素积组成
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1) # [B, T, H*4]

    # 3. 通过三个全连接层得到注意力权重向量，向量中的每个值代表历史点击过的每个商品的权重。神经元个数分别为[80, 40, 1]
    d_layer_1_all = layers.dense(80, activation=tf.nn.sigmoid, name='f1_att')(din_all)
    d_layer_2_all = layers.dense(40, activation=tf.nn.sigmoid, name='f2_att')(d_layer_1_all)
    d_layer_3_all = layers.dense(1, activation=None, name='f3_att')(d_layer_2_all) # [B, T, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]]) # [B, 1, T]
    outputs = d_layer_3_all

    # 4. Mask处理。为了方便计算，之前将用户行为序列设定为固定长度，例如50。如果只有10个行为，则会对后40个补充padding，故此处需要对padding做mask。
    # 非padding处输出上一步得到的outputs权重，padding处则输出一个很大的负数，经过后续的softmax归一化后几乎为0
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1]) # [B, T]，有实际行为的置True，否则置为False
    key_masks = tf.expand_dim(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings) # [B, 1, T]

    # 5. 缩放处理，假设广告的特征向量长度为d，则此处除以d的开方
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # 6. 归一化，原版的DIN没有做，可以视情况去掉，得到最终的激活权重
    outputs = tf.nn.softmax(outputs) # [B, 1, T]

    # 7. 注意力池化本质上是加权求和池化，将用户行为序列中的各商品向量加权求和，得到输出向量
    outputs = tf.matmul(outputs, keys) # [B, 1, H]
    return outputs

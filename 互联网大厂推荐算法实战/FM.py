# p75
def output_logits_from_bi_interaction(features, embedding_table, params):
    # 见《Neural Factorization Machines for Sparse Predictive Analytics》论文的公式(4)
    fields_embeddings = [] # 每个field的Embedding，是每个Field所包含的Feature Embedding的和
    fields_squared_embeddings = [] # 每个元素，是当前Field所有Feature Embedding的平方的和

    for fieldname, vocabname in field2vocab_mapping.items():
        sp_ids = features[fieldname + "_ids"] # 当前Field下所有稀疏特征的Feature id
        sp_values = features[fieldname + "_values"] # 当前Field下所有稀疏特征对应的值

        # ——————————Embedding
        embed_weights = embedding_table.get_embed_weights(vocabname) # 得到Embedding矩阵
        # 当前Field下所有Feature Embedding求和
        # Embedding: [batch_size, embed_dim]
        embedding = embedding_ops.safe_embedding_lookup_sparse(
            embed_weights, sp_ids, sp_values,
            combiner='sum',
            name='{}_embedding'.format(fieldname))
        fields_embeddings.append(embedding)

        # ______ square of Embedding
        squared_emb_weights = tf.square(embed_weights) # Embedding矩阵求平方
        # 稀疏特征的值求平方
        squared_sp_values = tf.SparseTensor(indices=sp_values.indices,
                                            values=tf.square(sp_values.values),
                                            dense_shape=sp_values.dense_shape)

        # 当前Field下所有Feature Embedding的平方的和
        # squared_embedding: [batch_size, embed_dim]
        squared_embedding = embedding_ops.safe_embedding_lookup_sparse(
            squared_emb_weights, sp_ids, squared_sp_values,
            combiner='sum',
            name='{}_squared_embedding'.format(fieldname))
        fields_squared_embeddings.append(squared_embedding)

    # 所有Feature Embedding先求和，再平方，形状是[batch_size, embed_dim]
    sum_embedding_then_square = tf.square(tf.add_n(fields_embeddings))
    # 所有Feature Embedding先平方，再求和，形状是[batch_size, embed_dim]
    square_embedding_then_sum = tf.add_n(fields_squared_embeddings)
    # 所有特征两两交叉的结果，形状是[batch_size, embed_dim]
    bi_interaction = 0.5 * (sum_embedding_then_square - square_embedding_then_sum)

    # 由FM部分贡献的logits
    logits = tf.layers.dense(bi_interaction, units=1, use_bias=True, activation=None)
    # 因为FM与DNN共享Embedding，所以除了logit，还返回各个Field的Embedding，以便搭建DNN
    return logits, fields_embeddings

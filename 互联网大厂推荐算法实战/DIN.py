# p88
def target_attention():
    target_item_embedding = ... # 候选物料的Embedding，[batch_size, dim_target]
    user_behavior_seq = ... # 某个用户行为序列，[batch_size, seq_len, dim_seq]
    padding_mask = ... # user_behavior_seq中哪些位置是填充的，不需要Attention

    # 把候选物料变形成一个长度为1的序列
    query = tf.reshape(target_item_embedding, [-1, 1, dim_target])


    # atten_result: (batch_size, 1, dim_out)
    attention_layer = MultiHeadAttention(num_heads, dim_key, dim_val, dim_out)
    atten_result, _ = attention_layer(
        q=query, # query就是候选物料
        k=user_behavior_seq,
        v=user_behavior_seq,
        mask=padding_mask
    )

    # reshape去除中间不必要的一维
    # user_interest_emb是提取出来的用户兴趣向量，喂给上层模型，参与CTR建模
    user_interest_emb = tf.reshape(atten_result, [-1, dim_out])
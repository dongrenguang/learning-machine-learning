# p90
def double_attention():
    target_item_embedding = ... # 候选物料的Embedding，[batch_size, dim_target]
    user_behavior_seq = ... # 某个用户行为序列，[batch_size, seq_len, dim_in_seq]
    padding_mask = ... # user_behavior_seq中哪些位置是填充的，不需要Attention
    dim_in_seq = tf.shape(user_behavior_seq)[-1] # 序列中每个元素的长度

    # ********** 第一层做Self-Attention，建模序列内部的依赖性
    self_atten_layer = MultiHeadAttention(num_heads=n_heads1,
                                          dim_key=dim_in_seq,
                                          dim_val=dim_in_seq,
                                          dim_out=dim_in_seq)
    # 做Self-Attention，q=k=v=user_behavior_seq，他们的形状都是[batch_size, len(user_behavior_seq), dim_in_seq]
    self_atten_seq, _ = self_atten_layer(q=user_behavior_seq,
                                         k=user_behavior_seq,
                                         v=user_behavior_seq,
                                         mask=padding_mask)

    # ********** 第二层做Target-Attention，建模序候选物料与行为序列的相关性
    target_atten_layer = MultiHeadAttention(num_heads=n_heads2,
                                            dim_key=dim_key,
                                            dim_val=dim_val,
                                            dim_out=dim_out)
    # 把候选物料变形成一个长度为1的序列
    target_query = tf.reshape(target_item_embedding,[-1, 1, dim_target])
    # atten_result: [batch_size, 1, dim_out]
    atten_result, _ = target_atten_layer(
            q=target_query, # 代表候选物料
            k=self_atten_seq, # 以Self-Attention结果作为Target-Attention的对象
            v=self.atten_seq,
            mask=padding_mask)
    
    # reshape去除中间不必要的一维
    # user_interest_emb是提取出来的用户兴趣向量，喂给上层模型，参与CTR建模
    user_interest_emb = tf.reshape(atten_result, [-1, dim_out])

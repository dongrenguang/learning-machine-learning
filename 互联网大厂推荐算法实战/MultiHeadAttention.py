# p81
def scaled_dot_product_attention(q, k, v, mask):
    """
    输入：
        q: (batch_size, num_heads, seq_len_q, dim_key)
        k: (batch_size, num_heads, seq_len_k, dim_key)
        v: (batch_size, num_heads, seq_len_k, dim_val)
        mask: 必须能够broadcastable to (..., seq_len_q, seq_len_k)的形状
    输出：
        output: q对k/v做Attention的结果，(batch_size, num_heads, seq_len_q, dim_val)
        attention_weights: q对k的注意力权重，(batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # 每个head下，每个q对每个k的注意力权重（尚未归一化）
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 为了使训练更稳定，除以sqrt(dim_key)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 在mask的地方上加上一个极小的负数-1e9，保证在softmax后，mask位置上的权重都是0
    if mask is not None:
        # mask的形状一般是(batch_size, 1, 1, seq_len_k)
        # 但是能够broadcast成与scaled_attention_logits相同的形状
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_logits += (mask * -1e9)

    # 沿着最后一维（也就是seq_len_k）用softmax归一化
    # 保证一个query对所有key的注意力权重之和==1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (batch_size, num_heads, seq_len_q, seq_len_k)

    # v: (bathc_size, num_heads, seq_len_k, dim_val)
    output = tf.matmul(attention_weights, v) # (bathc_size, num_heads, seq_len_q, dim_val)

    return output, attention_weights
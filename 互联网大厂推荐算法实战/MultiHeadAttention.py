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


# p82
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_key, dim_val, dim_out):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_key = dim_key # 每个query和key都要映射成相同的长度
        # 每个value要映射成的长度
        self.dim_val = dim_val if dim_val is not None else dim_key
        # 定义映射矩阵
        self.wq = tf.keras.layers.Dense(num_heads * dim_key)
        self.wk = tf.keras.layers.Dense(num_heads * dim_key)
        self.wv = tf.keras.layers.Dense(num_heads * dim_val)
        self.wo = tf.keras.layers.Dense(dim_out) # dim_out: 希望输出的维度

    def split_heads(self, x, batch_size, dim):
        # 输入x: (batch_size, seq_len, num_heads * dim)
        # 输出x: (batch_size, seq_len, num_heads, dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, dim))
        # 最终输出: (batch_size, num_heads, seq_len, dim)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """
        输入：
            q: (batch_size, seq_len_q, old_dq)
            k: (batch_size, seq_len_k, old_dk)
            v: (batch_size, seq_len_k, old_dv),与k序列相同长度
            mask: 可以为空，否则形状为(batch_size, 1, 1, seq_len_k)，表示哪个key不需要做attention
        输出：
            output: Attention结果，(batch_size, seq_len_q, dim_out)
            attention_weights: Attention权重，(batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # ********** 将输入映射成希望的形状
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size, seq_len_q, num_heads * dim_key)
        k = self.wk(k) # (batch_size, seq_len_k, num_heads * dim_key)
        v = self.wv(v) # (batch_size, seq_len_k, num_heads * dim_val)

        q = self.split_heads(q, batch_size, self.dim_key) # (bs, nh, seq_len_q, dim_key)
        k = self.split_heads(k, batch_size, self.dim_key) # (bs, nh, seq_len_k, dim_key)
        v = self.split_heads(v, batch_size, self.dim_val) # (bs, nh, seq_len_k, dim_val)

        # ********** Multi-Head Attention
        # scaled_attention: (batch_size, num_heads, seq_len_q, dim_val)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # ********** 将Attention结果映射成希望的形状
        # (batch_size, seq_len_q, num_heads, dim_val)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, num_heads * dim_val)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.num_heads * self.dim_val))

        output = self.wo(concat_attention) # (batch_size, seq_len_q, dim_out)
        return output, attention_weights


# p84
def create_padding_mask(seq):
    """
    seq: [batch_size, seq_len]的整数矩阵。如果某个元素==0，代表那个位置是padding
    """
    # (batch_size, seq_len)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 返回结果：(batch_size, 1, 1, seq_len)
    # 加入中间两个长度=1的维度，是为了能够broadcast成希望的形状
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

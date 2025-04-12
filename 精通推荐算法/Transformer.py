# p109
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, emb_dim, num_heads=4, **kwargs):
        """
        @param emb_dim: 输入Embedding维度，必须为多头数目的整数倍
        @param num_heads: 多头的数目，一般取8、4、2等
        """
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        # 定义q、k、v三个权重矩阵
        self.w_query = keras.layers.Dense(256)
        self.w_key = keras.layers.Dense(256)
        self.w_value = keras.layers.Dense(256)

        # 每个头内的向量维度，等于总维度除以多头数
        self.dim_one_head = 256 // num_heads

        # 全连接单元，将多个头连接起来。输出向量与原始输入向量维度相同，从而可以进行残差连接
        self.w_combine_heads = keras.layers.Dense(emb_dim)
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def attention(self, query, key, value):
        """attention计算，softmax((q * k) / sqrt(dk, 0.5)) * v
        @param query: q向量，[batch_size, num_heads, seq_len, dim_one_head]
        @param key: k向量，shape同上
        @param value: v向量，shape同上
        @return: 返回attention计算后的向量，以及attention得分
        """
        # 1. 内积，计算q和k的相关性权重，得到一个标量
        score = tf.matmul(query, key, transpose_b=True) # [batch_size, num_heads, seq_len, seq_len]
        # 2. 计算向量长度，dk
        dim_key = tf.cast(tf.shape(key).shape[-1], tf.float32) # dim_one_head
        # 3. 缩放，向量长度上做归一化
        score = score / tf.math.sqrt(dim_key)
        # 4. softmax归一化，得到权重处于0到1之间
        att_scores = tf.nn.softmax(score, axis=-1) # [batch_size, num_heads, seq_len, seq_len]
        # 权重乘以每个v向量，再累加起来，最终得到一个向量，维度与q、k、v相同
        att_output = tf.matmul(att_scores, value) # [batch_size, num_heads, seq_len, dim_one_head]，维度与输入维度一致
        return att_output, att_scores

    def build_multi_head(self, x_input, batch_size):
        """分割隐向量到多个head中，所以多头并不会带来维度倍增
        @param x_input: 输入向量，可以为q、k、v等向量
        @param batch_size: 单步训练样本量
        @return: 多头矩阵
        """
        x_input = tf.reshape(x_input, shape=(batch_size, -1, self.num_heads, self.dim_one_head)) # [batch_size, seq_len, num_heads, dim_one_head]
        return tf.transpose(x_input, perm=[0, 2, 1, 3]) # [batch_size, num_heads, seq_len, dim_one_head]

    def call(self, inputs, **kwargs):
        """多头注意力计算部分
        @param inputs: 输入参数，包括query、key等
        @return: 返回多头注意力输出向量
        """
        x_query = inputs[0] # [batch_size, seq_len, emb_dim]
        x_key = inputs[1]
        batch_size = tf.shape(x_query)[0]

        # 得到q向量，原始输入经过线性变换，然后进行多头切割
        query = self.w_query(x_query) # [batch_size, seq_len, 256]
        query = self.build_multi_head(query, batch_size) # [batch_size, num_heads, seq_len, dim_one_head]

        # 得到k向量
        key = self.w_key(x_key)
        key = self.build_multi_head(key, batch_size)

        # 得到v向量，v和k用同一个输入
        value = self.w_value(x_key)
        value = self.build_multi_head(value, batch_size)

        # attention计算
        att_output, att_scores = self.attention(query, key, value)
        att_output = tf.transpose(att_output, perm=[0, 2, 1, 3]) # [batch_size, seq_len, num_heads, dim_one_head]
        att_output = tf.reshape(att_output, shape=(batch_size, -1, self.dim_one_head * self.num_heads)) # [batch_size, seq_len, 256]

        # 多头合并，并进行线性连接输出
        output = self.w_combine_heads(att_output) # [batch_size, seq_len, emb_dim]
        return output


# p112
class Transformer(keras.layers.Layer):
    def __init__(self, seq_len, emb_dim, num_heads=4, ff_dim=128, **kwargs):
        # position emb，位置编码向量
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.positions_embedding = Embedding(seq_len, self.emb_dim, input_length=seq_len)

        # 多头注意力层
        self.att = MultiHeadSelfAttention(self.emb_dim, num_heads)

        # 两层前馈神经网络，本质是全连接
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(self.emb_dim)
        ])

        # 两次层归一化，一次为input和attention输出残差连接，另一次为attention输出和全连接输出残差连接
        self.ln1 = keras.layers.LayerNormalization()
        self.ln2 = keras.layers.LayerNormalization()
        self.dropout1 = keras.layers.Dropout(0.3)
        self.dropout2 = keras.layers.Dropout(0.3)

        super(Transformer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """结构：一层多头注意力，叠加两层前馈神经网络，中间加入层归一化和残差连接
        @param inputs: 原始输入向量，可以包括物品ID、类目ID、品牌ID等特征的Embedding
        @return: 单层Transformer结构的输出
        """
        # 1. 构建位置特征，并进行Embedding（和谷歌原始论文中用sin、cos的position不一样）
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        positions_embedding = self.positions_embedding(positions)

        # 2. 将原始输入和位置向量相加，也可以采用BST原文中的concat方法
        x_key = inputs + positions_embedding

        # 3. 多头注意力，利用层归一化和输入进行残差连接
        att_output = self.att(inputs) # 这里并没有用x_key？
        att_output = self.dropout1(att_output)
        output1 = self.ln1(x_key + att_output)

        # 4. 两个全连接层，利用层归一化和输入进行残差连接
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output)
        output2 = self.ln2(output1 + ffn_output) # [batch_size, seq_len, emb_dim]

        # 5. 平均池化，对序列进行压缩
        result = tf.reduce_mean(output2, axis=1) # [batch_size, emb_dim]
        return result

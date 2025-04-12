# p172
def build_esmm(user_field_size, item_field_size, num_user_features, num_item_features, emb_size):
    """
    @param user_field_size: 用户侧特征维度
    @param item_field_size: 物品侧特征维度
    @param num_user_features: 用户侧特征枚举数
    @param num_item_features: 物品侧特征枚举数
    @param emb_size: Embedding维度
    """
    # 1. 定义输入层，包括用户侧和物品侧特征
    # 这里为了简便，让CTR和CVR共用一套特征体系，实战中可以不同
    user_input = Input(shape=(user_field_size,), name="user_input")
    item_input = Input(shape=(item_field_size,), name="item_input")

    # 2. 定义Embedding层，注意CTR和CVR共享Embedding层，从而解决CVR任务的数据稀疏问题
    # 同样为了简便，让CTR和CVR共用一套Embedding的输出，实战中可以不同
    user_emb = Embedding(num_user_features, emb_size, input_length=1)(user_input)
    item_emb = Embedding(num_item_features, emb_size, input_length=1)(item_input)

    # 3. 合并Embedding
    all_emb = Concatenate()([user_emb, item_emb])

    # 4. 定义CTR模型结构，采用三个全连接层，这里可以采用任意其他单任务模型结构
    ctr_dense1 = Dense(64, activation="relu", name="ctr_dense1")(all_emb)
    ctr_dense2 = Dense(32, activation="relu", name="ctr_dense2")(ctr_dense1)
    ctr_output = Dense(1, activation="sigmoid", name="ctr_output")(ctr_dense2)

    # 5. 定义CVR模型结构，与CTR模型基本相同
    cvr_dense1 = Dense(64, activation="relu", name="cvr_dense1")(all_emb)
    cvr_dense2 = Dense(32, activation="relu", name="cvr_dense2")(cvr_dense1)
    cvr_output = Dense(1, activation="sigmoid", name="cvr_output")(cvr_dense2)

    # 6. 定义CTCVR的输出，与CTR和CVR的输出相乘得到
    ctcvr_output = Multiply()([ctr_output, cvr_output])

    # 7. 定义Keras model，注意输出使用的是CTR和CTCVR，没有使用CVR，从而可以在全体曝光样本上训练模型，解决SSB问题
    model = Model(inputs=[user_input, item_input], outputs=[ctr_output, ctcvr_output])
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"],
        optimizer=Adam(learning_rate=0.0025),
        loss_weight=[1.0, 1.0],
        metrics=[AUC()]
    )
    return model

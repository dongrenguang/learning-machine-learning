p177
from keras.layers import Input, Embedding, Dense, Concatenate
from keras.models import Model
from keras.metrics import AUC
from keras.optimizers import Adam
import tensorflow as tf

def build_mmoe(user_field_size, item_field_size, num_user_features, num_item_features, emb_size, num_experts, num_tasks):
    """
    @param user_field_size: 用户侧特征维度
    @param item_field_size: 物品侧特征维度
    @param num_user_features: 用户侧特征枚举数
    @param num_item_features: 物品侧特征枚举数
    @param emb_size: Embedding维度
    @param num_experts: 专家网络数目
    @param num_tasks: 任务数目
    """
    # 1. 定义输入层，包括用户侧和物品侧特征
    user_input = Input(shape=(user_field_size,), name="user_input")
    item_inupt = Input(shape=(item_field_size,), name="item_input")

    # 2. 定义Embedding层，多个子任务可以共享一个Embedding层
    user_emb = Embedding(num_user_features, emb_size, input_length=1)(user_input)
    item_emb = Embedding(num_item_features, emb_size, input_length=1)(item_emb)

    # 3. 合并Embedding
    all_emb = Concatenate()([user_emb, item_emb]) # [batch_size, all_emb_size]

    # 4. 定义专家网络，这里每个专家网络为一个全连接层网络，专家网络被所有子任务共享
    # 实际场景中可以选用更复杂的网络结构
    expert_outputs = [] # [batch_size, num_experts, 128]
    for i in range(num_experts):
        expert_output = Dense(128, activation="relu", name=f"export_output_{i}")(all_emb) # [batch_size, 128]
        expert_outputs.append(expert_output)
    
    # 5. 定义门控单元，每个子任务都拥有一个独立的门控单元
    gate_outputs = []
    for i in range(num_tasks):
        # 先利用Embedding特征向量计算每个专家网络的权重
        gate_weight = Dense(num_experts, activation="softmax", name=f"gate_output_{i}")(all_emb) # [batch_size, num_experts, 1]
        # 再对所有专家网络加权求和得到融合后的输出
        gate_output = tf.matmul(expert_outputs, gate_weight, transpose_b=True)
        gate_outputs.append(gate_output)

    # 6. 定义塔网络，采用两个全连接层，每个子任务都拥有一个独立的塔网络
    tower_outputs = []
    for i in range(num_tasks):
        tower_layer1 = Dense(16, activation="relu", name=f"task_{i}_tower_output_1")(gate_outputs[i])
        tower_layer2 = Dense(1, activation="sigmoid", name=f"task_{i}_tower_output_2")(tower_layer1)
        tower_outputs.append(tower_layer2)

    # 7. 定义Keras model，这里以CTR和CVR两个任务为例，损失函数均为交叉熵
    model = Model(inputs=[user_input, item_inupt], outputs=tower_outputs)
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"],
        optimizer=Adam(learning_rate=0.0025),
        loss_weights=[1.0, 1.0],
        metrics=[AUC()]
    )
    return model

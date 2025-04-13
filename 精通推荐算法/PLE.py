# p184
from keras.layers import Input, Embedding, Dense, Concatenate
from keras.models import Model
from keras.metrics import AUC
from keras.optimizers import Adam
import tensorflow as tf

def build_cgc(inputs, num_tasks, num_specific_experts, num_shared_experts, is_last_layer=False):
    """
    @param inputs: 输入层，由所以独占的和共享的专家网络组成，共享的放到最后
    @param num_tasks: 任务数目
    @param num_specific_experts: 每个子任务独占的专家网络数目
    @param num_shared_experts: 所有子任务共享的专家网络数目
    @param is_last_layer: 是否是最后一层cgc
    """
    # 1. 定义任务独占的专家网络
    specific_expert_outputs = []
    for i in range(num_tasks):
        for j in range(num_specific_experts):
            expert_output = Dense(128, activation="relu")(inputs[i])
            specific_expert_outputs.append(expert_output)
    
    # 2. 定义任务共享的专家网络
    shared_expert_outputs = []
    for i in range(num_shared_experts):
        expert_output = Dense(128, activation="relu")(inputs[-1])
        shared_expert_outputs.append(expert_output)
    
    # 3. 定义每个任务经过门控单元融合后的输出
    cgc_outputs = []
    for i in range(num_tasks):
        # 一个任务的专家网络层，由独占的专家网络和共享的专家网络组成，将它们拼接在一起
        num_experts = num_specific_experts + num_shared_experts
        specific_experts = specific_expert_outputs[i * num_specific_experts: (i + 1) * num_specific_experts]
        experts = Concatenate()(specific_experts + shared_expert_outputs)
        # 计算门控单元权重矩阵，经过softmax归一化
        gate_weight = Dense(num_experts, activation="softmax")(inputs[i])
        # 所有专家网络加权求和，得到融合后的输出
        gate_output = tf.matmul(experts, gate_weight, transpose_b=True)
        cgc_outputs.append(gate_output)

    # 4. 如果不是最后一层cgc，则还需要定义共享专家网络的输出
    if not is_last_layer:
        # 这里需要特别注意，共享的输出融合了所有独占的和共享的专家网络
        num_experts = num_tasks * num_specific_experts + num_shared_experts
        experts = Concatenate()(specific_expert_outputs + shared_expert_outputs)
        # 计算门控单元权重矩阵
        gate_weight = Dense(num_experts, activation="softmax")(inputs[-1])
        # 所有专家网络加权求和，得到融合后的输出
        gate_output = tf.matmul(experts, gate_weight, transpose_b=True)
        cgc_outputs.append(gate_output)
    
    return cgc_outputs


# p186
def build_ple(user_field_size, item_field_size, num_user_features, num_item_features, emb_size, num_tasks,
        num_specific_experts, num_shared_experts, num_layers):
    """
    @param user_field_size: 用户侧特征维度
    @param item_field_size: 物品侧特征维度
    @param num_user_features: 用户侧特征枚举数
    @param num_item_features: 物品侧特征枚举数
    @param emb_size: Embedding维度
    @param num_tasks: 任务数目
    @param num_specific_experts: 每个子任务独占的专家网络数目
    @param num_shared_experts: 所有子任务共享的专家网络数目
    @param num_layers: CGC堆叠层数
    """
    # 1. 定义输入层，包括用户侧和物品侧特征
    user_input = Input(shape=(user_field_size,), name="user_input")
    item_input = Input(shape=(item_field_size,), name="item_input")

    # 2. 定义Embedding层
    user_emb = Embedding(num_user_features, emb_size, input_length=1)(user_input)
    item_emb = Embedding(num_item_features, emb_size, input_length=1)(item_input)

    # 3. 合并Embedding
    all_emb = Concatenate()([user_emb, item_emb])
    inputs = [all_emb] * (num_tasks + 1)

    # 4. 堆叠多层CGC
    for i in range(num_layers):
        if i != num_layers - 1:
            # 不是最后一层CGC，构建CGC输出
            cgc_outputs = build_cgc(inputs, num_tasks, num_specific_experts, num_shared_experts, is_last_layer=False)
            # 输出作为下一层的输入
            inputs = cgc_outputs
        else:
            # 最后一层CGC，构建CGC输出
            cgc_outputs = build_cgc(inputs, num_tasks, num_specific_experts, num_shared_experts, is_last_layer=True)

    # 5. 定义塔网络结构层，采用两个全连接层，每个子任务都拥有一个独立的塔网络
    tower_outputs = []
    for i in range(num_tasks):
        tower_layer1 = Dense(16, activation="relu", name=f"task_{i}_tower_output_1")(cgc_outputs[i])
        tower_layer2 = Dense(1, activation="sigmoid", name=f"task_{i}_tower_output_2")(tower_layer1)
        tower_outputs.append(tower_layer2)

    # 6. 定义keras model，这里以CTR和CVR两个任务为例，其损失函数均为交叉熵
    model = Model(inputs=[user_input, item_input], outputs=tower_outputs)
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"],
        optimizer=Adam(learning_rate=0.0025),
        loss_weights=[1.0, 1.0],
        metrics=[AUC()],
    )
    return model

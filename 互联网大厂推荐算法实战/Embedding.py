# p38
import tensorflow as tf

unq_categories = ["music", "movie", "finance", "game", "military", "history"]
# 这一层负责将string转化成int型id
id_mapping_layer = tf.keras.layers.StringLookup(vocabulary=unq_categories)

emb_layer = tf.keras.layers.Embedding(
    input_dim=len(unq_categories) + 1, # 多加一维是为了处理输入不包括在字典中的情况
    output_dim=4
)

cate_input = ... # [batch_size, 1]
cate_ids = id_mapping_layer(cate_input)
cate_embedding = emb_layer(cate_ids) # [batch_size, 4]

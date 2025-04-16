# p190
class PLE(MultiTaskModel):
    def gate(self, selector_fea, vec_feas):
        """
        输入：
            vec_feas是一个长度=N的数组，N是专家的个数，数组中的每个元素都是[B, D]，其中B是batch_size，D是每个专家的输出维度
            selector_fea：针对某个task，生成各专家权重的小网络的输入特征
        输出：
            针对某个task，给所有专家加权相加的结果，形状是[B, D]
        """
        # vec: [B, N, D]，将所有专家的输出拼接起来
        vec = tf.stack(vec_feas, axis=1)

        # gate: [B, N]，N是专家的个数
        # gate代表根据selector_fea生成的各专家的权重
        gate = tf.layers.dense(inputs=selector_fea, units=len(vec_feas), activation=None)
        gate = tf.nn.softmax(gate, axis=1)
        gate = tf.expand_dims(gate, -1) # gate: 变形成[B, N, 1]

        # vec: [B, N, D]
        # gate: [B, N, 1]
        # task_input: [B, N, D]
        task_input = tf.multiply(vec, gate)
        # task_input: [B ,D]
        task_input = tf.reduce_sum(task_input, axis=1)
        return task_input

    def experts_layer(self, deep_fea, expert_num, experts_cfg):
        """
        输入：
            deep_fea: 专家的输入
            expert_num: 专家的个数
            experts_cfg: 专家的网络结构配置
        输出：
            一个长度等于专家个数的数组，每个元素是一个专家的输出
        """
        tower_outputs = []
        for expert_id in range(expert_num):
            tower_dnn = DNN(experts_cfg)
            tower_output = tower_dnn(deep_fea)
            tower_outputs.append(tower_output)
        return tower_outputs

    def CGC_layer(self, extraction_networks_cfg, extraction_network_fea, shared_expert_fea, final_flag):
        """
        输入：
            extraction_networks_cfg: 网络结构配置
            extraction_network_fea: 下层每个task的输出
            shared_expert_fea: 下层共享部分的输出
            final_flag: 是否是最后一层，因为最后一层就没必要再建模共享部分了
        """
        layer_name = extraction_networks_cfg.network_name

        # ********** 共享专家
        # 这些expert的输入都是shared_expert_fea，也就是下层共享专家的总输出
        # 一共有extraction_networks_cfg.share_num个共享专家
        # expert_shared_out是一个长度=extraction_networks_cfg.share_num的数组
        # 数组中每个元素都是[B, D]，B=batch_size，D=每个专家的输出长度
        expert_shared_out = self.experts_layer(
            shared_expert_fea, extraction_networks_cfg.share_num,
            extraction_networks_cfg.share_expert_net, layer_name + '_share/dnn')
        
        # ********** 每个task独享部分的建模
        experts_outs = [] # 所有task的所有专家的输出
        cgc_layer_outs = [] # 所有task的独享输出
        for task_idx in range(self._task_nums):
            name = layer_name + '_task_%d' % task_idx

            # 针对当前task（编号task_idx）
            # 其输入是extraction_network_fea[task_idx]，也就是下层第task_idx个任务的输出
            # experts_out是一个长度=extraction_networks_cfg.expert_num_per_task的数组
            # extraction_networks_cfg.expert_num_per_task是当前task配置的专家的个数
            # 数组中的每个元素都是[B, D]，B=batch_size，D=每个专家的输出长度
            experts_out = self.experts_layer(
                extraction_network_fea[task_idx],
                extraction_networks_cfg.expert_num_per_task,
                extraction_networks_cfg.task_expert_net, name)

            # 针对task_idx这个task，融合各相关专家的输出
            # 参与融合的experts是experts_out(当前task的experts)+expert_shared_out(共享expert)
            # 根据extraction_network_fea[task_idx]（即下层第task_idx个任务的输出）生成各expert的权重
            # cgc_layer_out: [B, D]
            cgc_layer_out = self.gate(extraction_network_fea[task_idx],
                                      experts_out + expert_shared_out, name)
            
            experts_outs.extend(experts_out) # 收集当前task中的各专家的输出
            cgc_layer_outs.append(cgc_layer_out)
            
        # ********** 所有task共享部分的建模
        if final_flag:
            shared_layer_out = None # 如果是最后一层，没必要再建模共享的部分
        else:
            # 针对共享部分，融合各相关专家的输出
            # 参与融合的专家是experts_outs(所有task的所有experts) + expert_shared_out(共享专家)
            # 根据shared_expert_fea（即下层共享部分的输出）生成各专家的权重
            # shared_layer_out: [B, D]
            shared_layer_out = self.gate(shared_expert_fea,
                                         experts_outs + expert_shared_out,
                                         layer_name + '_share')
        
        # cgc_layer_outs：一个长度是#tasks的数组，每个元素的形状都是[B, D]，代表本层对某一个task的输出
        # shared_layer_out: 本层共享部分的输出，形状也是[B, D]
        return cgc_layer_outs, shared_layer_out

    def build_predict_graph(self):
        # 最底层，每个task独享的输入特征，和共享的输入特征相同，都是self._features
        extraction_network_fea = [self._features] * self._task_nums
        shared_expert_fea = self._features

        # ********** 提取对各task的输入
        final_flag = False
        # 循环遍历，一共要经历多层Experts的Extraction
        for idx in range(len(self._model_config.extraction_networks)):
            # extraction_network是当前层的网络配置
            extraction_network = self._model_config.extraction_networks[idx]

            if idx == len(self._model_config.extraction_networks) - 1:
                final_flag = True

            # extraction_network_fea: 既是输入也是输出
            # 一个长度是#tasks的数组，每个元素的形状都是[B, D]，代表本层对某一个task的输出
            # shared_expert_fea: 本层共享部分的输出，形状也是[B, D]
            extraction_network_fea, shared_expert_fea = self.CGC_layer(
                extraction_network, # 本层的网络结构配置
                extraction_network_fea, # 上一层各task的输出
                shared_expert_fea, # 上一层共享部分的输出
                final_flag
            )
        
        # ********** 各task的预测
        tower_outputs = {}
        # 遍历每个task
        for i, task_tower_cfg in enumerate(self._model_config.task_towers):
            # task_tower_cfg是当前task tower的配置
            tower_name = task_tower_cfg.tower_name
            tower_dnn = DNN(task_tower_cfg.dnn, name=tower_name)

            # extraction_network_fea[i]是多层专家提取出来的对第i个task的输入
            tower_output = tower_dnn(extraction_network_fea[i])
            tower_outputs[tower_name] = tower_output

        return tower_outputs

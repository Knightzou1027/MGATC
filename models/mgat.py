import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MultiViewGAT(nn.Module):
    """
    多视图图注意力网络
    - 使用PyG的GATConv实现视图内的消息传递
    - 实现节点级别的视图融合
    """

    # 三个视图：时间、空间、属性，单视图里的注意力heads可选
    def __init__(self, in_features, hidden_dim, embedding_dim, dropout=0.6, alpha=0.2, num_views=3, heads=1):
        super(MultiViewGAT, self).__init__()
        self.dropout = dropout
        self.num_views = num_views
        self.embedding_dim = embedding_dim

        # 第一层GAT（每个视图一个）
        self.conv1 = nn.ModuleList([
            GATConv(
                in_features,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                negative_slope=alpha
            ) for _ in range(num_views)
        ])

        # 第二层GAT（每个视图一个）
        self.conv2 = nn.ModuleList([
            GATConv(
                hidden_dim,
                embedding_dim // heads,
                heads=heads,
                dropout=dropout,
                negative_slope=alpha
            ) for _ in range(num_views)
        ])

        # 节点特定的视图融合网络
        # 使用mlp函数为每个节点学习不同视图的权重
        self.view_attention = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, adj_list):
        """
        前向传播

        参数:
            x: [N, in_features] - 节点特征矩阵
            adj_list: 长度为num_views的列表，每个元素是包含边索引和权重的字典

        返回:
            z: [N, embedding_dim] - 节点嵌入
            recon: [N, N] - 重构的邻接矩阵
        """
        batch_size = x.size(0)
        multi_view_output = []

        # 1. 对每个视图单独进行消息传递
        for view_idx in range(self.num_views):
            # 获取当前视图的边索引和权重
            edge_index = adj_list[view_idx]['edge_index']
            edge_weight = adj_list[view_idx].get('edge_weight', None)

            # 第一层GAT，输出的节点表示 [N, hidden_dim]
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = self.conv1[view_idx](h, edge_index, edge_weight)
            h = F.elu(h)

            # 第二层GAT
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2[view_idx](h, edge_index, edge_weight)

            multi_view_output.append(h)
        # 不同视图的节点嵌入添加到列表中

        # 2. 计算节点特定的视图注意力权重
        # 将所有视图输出堆叠 [num_views, N, embedding_dim]
        stacked_outputs = torch.stack(multi_view_output)

        # 为每个节点计算视图权重
        view_weights = torch.zeros(batch_size, self.num_views, device=x.device)

        for i in range(batch_size):
            # 获取节点i在所有视图下的表示 [num_views, embedding_dim]
            node_views = stacked_outputs[:, i, :]

            # 计算节点i对每个视图的注意力分数
            attn_scores = torch.zeros(self.num_views, device=x.device)
            for v in range(self.num_views):
                # 对每个视图表示应用注意力网络得到分数，squeeze将张量抓华为标量值赋值
                attn_scores[v] = self.view_attention(node_views[v]).squeeze()

            # 使用softmax归一化得到权重
            view_weights[i] = F.softmax(attn_scores, dim=0)

        # 3. 加权融合多个视图的节点表示
        # 初始化 z 为零张量，形状与 multi_view_output[0] 相同
        z = torch.zeros_like(multi_view_output[0])
        for v in range(self.num_views):
            # 节点级别的视图融合
            z += multi_view_output[v] * view_weights[:, v].unsqueeze(1)

        # 4. 应用ReLU和L2归一化以满足EGAE要求
        # 应用ReLU确保非负性 (对应论文公式(17)在最后一层应用ReLU)
        z_relu = F.relu(z)

        # 对嵌入进行L2归一化以便生成相似度矩阵
        # 使用epsilon防止除零错误
        epsilon = 1e-7
        z_norm = F.normalize(z_relu, p=2, dim=1, eps=epsilon)

        # 通过内积计算节点相似度，用于重构邻接矩阵
        recon = torch.sigmoid(torch.mm(z_norm, z_norm.t()))

        return z, recon, view_weights  # 额外返回视图权重

    def compute_regularization_loss(self):
        """
        计算参数正则化损失 - 鼓励不同视图的参数相似性
        """
        first_param = next(self.parameters())  # 获取模型参数以确定 device 和 dtype
        reg_loss = torch.tensor(0.0, dtype=torch.float, device=first_param.device)

        # 第一层GAT参数正则化
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                # 线性层权重正则化
                w_i = self.conv1[i].lin_src.weight
                w_j = self.conv1[j].lin_src.weight
                reg_loss += torch.sum((w_i - w_j) ** 2)

                # 注意力权重正则化
                att_src_i = self.conv1[i].att_src
                att_src_j = self.conv1[j].att_src
                reg_loss += torch.sum((att_src_i - att_src_j) ** 2)

                att_dst_i = self.conv1[i].att_dst
                att_dst_j = self.conv1[j].att_dst
                reg_loss += torch.sum((att_dst_i - att_dst_j) ** 2)

                # 偏置项正则化 (可选)
                # 检查偏置是否存在且不为 None
                if hasattr(self.conv1[i], 'bias') and self.conv1[i].bias is not None and \
                        hasattr(self.conv1[j], 'bias') and self.conv1[j].bias is not None:
                    bias_i = self.conv1[i].bias
                    bias_j = self.conv1[j].bias
                    reg_loss += torch.sum((bias_i - bias_j) ** 2)

        # 第二层GAT参数正则化
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                # 线性层权重正则化
                w_i = self.conv2[i].lin_src.weight
                w_j = self.conv2[j].lin_src.weight
                reg_loss += torch.sum((w_i - w_j) ** 2)

                # 注意力权重正则化
                att_src_i = self.conv2[i].att_src
                att_src_j = self.conv2[j].att_src
                reg_loss += torch.sum((att_src_i - att_src_j) ** 2)

                att_dst_i = self.conv2[i].att_dst
                att_dst_j = self.conv2[j].att_dst
                reg_loss += torch.sum((att_dst_i - att_dst_j) ** 2)

                # 偏置项正则化 (可选)
                if hasattr(self.conv2[i], 'bias') and self.conv2[i].bias is not None and \
                        hasattr(self.conv2[j], 'bias') and self.conv2[j].bias is not None:
                    bias_i = self.conv2[i].bias
                    bias_j = self.conv2[j].bias
                    reg_loss += torch.sum((bias_i - bias_j) ** 2)


        return reg_loss

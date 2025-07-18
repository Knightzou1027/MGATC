import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteringModule(nn.Module):
    """
    深度聚类模块 - 选择需要的目标数量
    """

    def __init__(self, embedding_dim, n_clusters=6, alpha=1.0):
        super(ClusteringModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters  # 固定聚类数量
        self.alpha = alpha  # t分布自由度参数

        # 初始化聚类中心，其中聚类中心的信息是可学习的参数，参与梯度计算和优化
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """随机初始化聚类中心"""
        nn.init.xavier_uniform_(self.cluster_centers)

    def init_cluster_centers(self, centers):
        """
        用提供的聚类中心初始化

        参数:
            centers: Tensor [n_clusters, embedding_dim] - 初始聚类中心
        """
        assert centers.size(0) == self.n_clusters, "聚类中心数量不匹配"
        assert centers.size(1) == self.embedding_dim, "嵌入维度不匹配"
        self.cluster_centers.data = centers

    def forward(self, embeddings):
        """
        计算软聚类分配

        参数:
            embeddings: Tensor [N, embedding_dim] - 节点嵌入

        返回:
            q: Tensor [N, n_clusters] - 软聚类分配矩阵
        """
        # 计算每个样本到每个聚类中心的平方距离
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(embeddings.unsqueeze(1) - self.cluster_centers, 2), 2) / self.alpha)

        # 调整分布
        q = q.pow((self.alpha + 1.0) / 2.0)

        # 归一化以确保每行和为1
        q = q / q.sum(1, keepdim=True)

        return q


def target_distribution(q):
    """
    计算目标分布

    参数:
        q: Tensor [N, n_clusters] - 软聚类分配矩阵

    返回:
        p: Tensor [N, n_clusters] - 目标分布
    """
    # 计算每个聚类的"硬度"
    weight = q ** 2 / q.sum(0)

    # 归一化
    p = (weight.t() / weight.sum(1)).t()

    return p

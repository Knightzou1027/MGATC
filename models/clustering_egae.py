import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import logging
from sklearn.utils.extmath import randomized_svd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 辅助函数：安全地执行 SVD
import torch
import numpy as np
import logging
from sklearn.utils.extmath import randomized_svd  # 需要 scikit-learn


def safe_svd(matrix, num_components, use_eigh_for_symmetric=True,  # 新增参数
             use_randomized_svd_on_cpu_fallback=True, n_iter_randomized=7):
    """
    安全地执行 SVD 或对称特征分解 (eigh) 并提取主成分。
    如果 matrix 被认为是是对称的，并且 use_eigh_for_symmetric=True，则优先尝试 eigh。
    否则，尝试 torch.linalg.svd。
    如果GPU失败，回退到CPU。
    """
    device = matrix.device
    logging.info(f"Attempting SVD/EIGH for matrix of shape {matrix.shape} on device: {device}")

    # 检查矩阵是否近似对称 (可选但推荐)
    # is_symmetric = torch.allclose(matrix, matrix.T, atol=1e-5) # 根据需要调整容差
    # if use_eigh_for_symmetric and not is_symmetric:
    #     logging.warning("Matrix is not symmetric, but use_eigh_for_symmetric is True. Falling back to SVD or proceeding with eigh with caution.")
    # 这里的 matrix 是 ZZT，理论上一定是堆成的，所以可以省略这个检查或假设其对称

    if use_eigh_for_symmetric:  # 并且我们知道 ZZT 是对称的
        try:
            logging.info(f"Attempting torch.linalg.eigh on {device} (for symmetric matrix)...")
            # eigh 返回: eigenvalues (L), eigenvectors (E)
            # 特征值默认按升序排列
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

            # 我们需要与最大的 num_components 个特征值对应的特征向量
            # 因此，获取降序排列的特征值的索引
            sorted_indices = torch.argsort(eigenvalues, descending=True)

            # 根据排序后的索引选择特征向量
            top_k_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

            indicator = top_k_eigenvectors
            logging.info(f"torch.linalg.eigh on {device} successful.")
            return indicator.contiguous()
        except Exception as e_eigh:
            logging.warning(f"torch.linalg.eigh on {device} failed: {e_eigh}. Falling back to torch.linalg.svd or CPU.")
            # 如果eigh失败，可以尝试svd或者直接进入CPU回退逻辑

    # 如果不使用eigh或者eigh失败了，尝试torch.linalg.svd
    try:
        logging.info(f"Attempting torch.linalg.svd on {device} with full_matrices=False...")
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if U.shape[1] < num_components:
            logging.warning(f"torch.linalg.svd (full_matrices=False) on {device} returned U with {U.shape[1]} columns, "
                            f"which is less than num_components ({num_components}). This means rank < num_components. "
                            f"Using all available {U.shape[1]} columns for the indicator matrix.")
            indicator = U[:, :U.shape[1]]
        else:
            indicator = U[:, :num_components]

        logging.info(f"torch.linalg.svd on {device} with full_matrices=False successful.")
        return indicator.contiguous()

    except Exception as e_gpu_svd:  # 修改了异常变量名以区分
        logging.warning(f"torch.linalg.svd on {device} also failed: {e_gpu_svd}. Falling back to CPU SVD.")

        matrix_cpu = matrix.detach().cpu().numpy()

        if use_randomized_svd_on_cpu_fallback:
            logging.info(
                f"Attempting randomized SVD on CPU with {num_components} components and n_iter={n_iter_randomized}...")
            try:
                U_np, S_np, Vh_np = randomized_svd(
                    matrix_cpu,
                    n_components=num_components,
                    n_iter=n_iter_randomized,
                    random_state=42
                )
                logging.info("Randomized SVD on CPU successful.")
                return torch.from_numpy(U_np.copy()).to(dtype=matrix.dtype, device=device)
            except Exception as e_random_svd:
                logging.warning(
                    f"Randomized SVD on CPU also failed: {e_random_svd}. Falling back to NumPy's exact SVD on CPU.")
                U_np_exact, _, _ = np.linalg.svd(matrix_cpu, full_matrices=False)
                indicator_np_exact = U_np_exact[:, :num_components]
                logging.info("NumPy's exact SVD on CPU successful as final fallback.")
                return torch.from_numpy(indicator_np_exact.copy()).to(dtype=matrix.dtype, device=device)
        else:
            logging.info("Attempting NumPy's exact SVD on CPU...")
            U_np_exact, _, _ = np.linalg.svd(matrix_cpu, full_matrices=False)
            indicator_np_exact = U_np_exact[:, :num_components]
            logging.info("NumPy's exact SVD on CPU successful.")
            return torch.from_numpy(indicator_np_exact.copy()).to(dtype=matrix.dtype, device=device)


class EGACECLoss(nn.Module):
    """
    基于EGAE论文的聚类损失模块 (Relaxed K-Means Loss)。
    包含计算聚类损失和更新聚类指示矩阵的功能。
    没有可学习的参数。
    """

    def __init__(self, embedding_dim, n_clusters, device):
        super(EGACECLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.device = device
        self.indicator = None # 聚类指示矩阵 P

    def update_indicator(self, embeddings):
        """
        根据当前嵌入，使用SVD更新聚类指示矩阵 P。
        对应EGAE论文中的 Algorithm 1 和公式 (14)。
        """
        if embeddings.requires_grad:
            embeddings = embeddings.detach()

        # 计算 ZZ^T
        ZZT = torch.mm(embeddings, embeddings.t())

        # 对 ZZ^T 进行SVD，提取前 n_clusters 个左奇异向量作为指示矩阵 P
        self.indicator = safe_svd(ZZT, self.n_clusters)
        self.indicator = self.indicator.to(self.device).detach()

    def forward(self, embeddings):
        """
        计算基于放松 K-Means 的聚类损失。
        对应EGAE论文中的公式 (24): Jc = tr(ZZ^T) - tr(P^T ZZ^T P)
        """
        if self.indicator is None:
            # 如果指示矩阵还未初始化，则先初始化一次
            self.update_indicator(embeddings)
            logging.info("EGACECLoss indicator initialized in forward.")

        # 计算 ZZ^T (基于当前的 embeddings)
        ZZT = torch.mm(embeddings, embeddings.t())

        # 计算 tr(ZZ^T)。对于L2归一化的嵌入，tr(ZZ^T) 等于节点数量
        # 使用 sum(embeddings * embeddings) 或 embeddings.norm(p=2)**2 也是等价的
        # 最简单直接就是 sum(diag(ZZT)) 或者对于归一化嵌入直接用 N
        # 这里使用 sum(diag(ZZT)) 更具通用性，即使嵌入未严格归一化
        term1 = torch.sum(torch.diag(ZZT))

        # 计算 tr(P^T ZZ^T P)
        # P = self.indicator
        # tr(P^T ZZT P) = tr((ZZT @ P).T @ P) - 另一种计算迹的方式
        term2 = torch.trace(torch.mm(self.indicator.t(), torch.mm(ZZT, self.indicator)))


        # 聚类损失 Jc = tr(ZZ^T) - tr(P^T ZZ^T P)
        loss_c = term1 - term2

        return loss_c

    def get_cluster_assignments(self, embeddings):
        """
        根据当前的聚类指示矩阵 P，使用 K-Means 获得聚类分配结果。
        对应EGAE论文中的 Algorithm 1。
        """
        if self.indicator is None:
             # 如果指示矩阵还未计算，先计算
             self.update_indicator(embeddings)
             logging.warning("EGACECLoss indicator was not initialized before getting assignments. Initialized now.")


        # 对指示矩阵 P 的行进行 K-Means 聚类
        indicator_np = self.indicator.detach().cpu().numpy()

        # 确保 numpy 数组是 float 类型以便 KMeans
        if indicator_np.dtype != np.float32 and indicator_np.dtype != np.float64:
             indicator_np = indicator_np.astype(np.float32)

        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init='auto', random_state=42)
        cluster_ids = kmeans.fit_predict(indicator_np)

        return cluster_ids, self.indicator
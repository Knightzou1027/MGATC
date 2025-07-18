import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clustering(embeddings, cluster_ids):
    """
    评估无标签数据的聚类结果

    参数:
        embeddings: ndarray, 节点嵌入
        cluster_ids: ndarray, 聚类分配结果

    返回:
        评估指标字典
    """
    results = {}

    # 轮廓系数 (范围: -1 到 1, 越高越好)
    # 衡量样本与自己所在簇的相似度与其他簇的相似度之间的比较
    try:
        results['silhouette'] = silhouette_score(embeddings, cluster_ids)
    except:
        results['silhouette'] = 0

    # 戴维斯-布尔丁指数 (范围: >=0, 越低越好)
    # 衡量簇内距离与簇间距离的比率
    try:
        results['davies_bouldin'] = davies_bouldin_score(embeddings, cluster_ids)
    except:
        results['davies_bouldin'] = 1

    # 卡林斯基-哈拉巴兹指数 (范围: >=0, 越高越好)
    # 簇间离散度与簇内紧密度的比率
    try:
        results['calinski_harabasz'] = calinski_harabasz_score(embeddings, cluster_ids)
    except:
        results['calinski_harabasz'] = 0

    return results


def compute_cluster_statistics(embeddings, cluster_ids):
    """
    计算聚类统计信息

    参数:
        embeddings: ndarray, 节点嵌入
        cluster_ids: ndarray, 聚类分配结果

    返回:
        cluster_stats: dict, 聚类统计信息
    """
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # 聚类大小
    cluster_sizes = np.array([np.sum(cluster_ids == c) for c in unique_clusters])

    # 聚类紧密度 (计算各聚类内部样本到质心的平均距离)
    centroids = np.array([embeddings[cluster_ids == c].mean(axis=0) for c in unique_clusters])
    intra_dists = np.array([
        np.mean(np.linalg.norm(embeddings[cluster_ids == c] - centroids[i], axis=1))
        if np.sum(cluster_ids == c) > 0 else 0
        for i, c in enumerate(unique_clusters)
    ])

    # 聚类间距离 (质心之间的欧氏距离)
    inter_dists = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            inter_dists[i, j] = inter_dists[j, i] = dist

    # 返回统计信息
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'cluster_sizes_mean': np.mean(cluster_sizes),
        'cluster_sizes_std': np.std(cluster_sizes),
        'intra_dists': intra_dists,
        'intra_dists_mean': np.mean(intra_dists),
        'inter_dists_mean': np.mean(inter_dists) if n_clusters > 1 else 0,
        'centroids': centroids
    }


# def compute_clustering_quality(embeddings, cluster_ids):
#     """
#     计算聚类质量综合得分
#
#     参数:
#         embeddings: ndarray, 节点嵌入
#         cluster_ids: ndarray, 聚类分配结果
#
#     返回:
#         quality_score: float, 聚类质量得分 (越高越好)
#     """
#     stats = compute_cluster_statistics(embeddings, cluster_ids)
#
#     # 内聚度 (越小越好，取负值使其方向一致)
#     cohesion = -stats['intra_dists_mean']
#
#     # 分离度 (越大越好)
#     separation = stats['inter_dists_mean']
#
#     # 组合得分 (越大越好)
#     if stats['n_clusters'] > 1:
#         # 计算簇间分离与簇内凝聚的比率
#         separation_to_cohesion = separation / (stats['intra_dists_mean'] + 1e-10)
#
#         # 考虑簇大小的均衡性 (越接近1越均衡)
#         size_balance = 1.0 - (stats['cluster_sizes_std'] / (stats['cluster_sizes_mean'] + 1e-10))
#         size_balance = max(0.0, min(1.0, size_balance))  # 确保在[0,1]范围内
#
#         # 综合评分
#         quality_score = cohesion + separation + separation_to_cohesion + size_balance
#     else:
#         # 只有一个簇时仅考虑内聚度
#         quality_score = cohesion
#
#     return quality_score

def evaluate_clustering_comprehensive(embeddings, cluster_ids):
    """
    全面评估聚类结果，并返回所有度量指标和统计信息

    参数:
        embeddings: ndarray, 节点嵌入
        cluster_ids: ndarray, 聚类分配结果

    返回:
        comprehensive_results: dict, 全面评估结果
    """
    # 获取基本指标
    basic_metrics = evaluate_clustering(embeddings, cluster_ids)

    # 获取聚类统计信息
    stats = compute_cluster_statistics(embeddings, cluster_ids)

    # 这些指标在聚类无效时，其基于统计的计算可能返回0或inf
    cohesion = -stats['intra_dists_mean']  # 内聚度 (负平均簇内距离，越大越好)
    separation = stats['inter_dists_mean']  # 平均簇间距离 (越大越好)

    separation_to_cohesion = 0.0  # 初始化比值为0
    if stats['intra_dists_mean'] > 1e-10:  # 避免除以零
        separation_to_cohesion = separation / stats['intra_dists_mean']  # 分离度与内聚度比值 (越大越好)

    size_balance = 0.0  # 初始化大小均衡性为0
    if stats['cluster_sizes_mean'] > 1e-10:  # 避免除以零
        size_balance = 1.0 - (stats['cluster_sizes_std'] / stats['cluster_sizes_mean'])  # 大小均衡性 (越接近1越均衡)
        size_balance = max(0.0, min(1.0, size_balance))  # 确保在[0,1]范围内

        # 合并所有结果
    comprehensive_results = {
        **basic_metrics,  # 包含 silhouette, davies_bouldin, calinski_harabasz
        'cohesion': cohesion,
        'separation': separation,
        'separation_to_cohesion': separation_to_cohesion,
        'size_balance': size_balance,
        'cluster_count': stats['n_clusters'],  # 实际的簇数量
        'avg_cluster_size': stats['cluster_sizes_mean'],
        'cluster_size_std': stats['cluster_sizes_std'],
        # 'avg_intra_distance': stats['intra_dists_mean'], # 已通过 cohesion 表示
        # 'avg_inter_distance': stats['inter_dists_mean'], # 已通过 separation 表示
        'cluster_sizes': stats['cluster_sizes'].tolist()  # 添加簇大小列表，方便查看具体分布

    }

    return comprehensive_results

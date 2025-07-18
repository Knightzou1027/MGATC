import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

from utils.metrics import evaluate_clustering, compute_cluster_statistics, calculate_purity
from utils.visualizers import plot_embeddings, plot_confusion_matrix
from models.clustering import ClusteringModule


class Evaluator:
    """
    用于模型评估和结果可视化的类。
    """

    def __init__(self, config, model=None, data=None, features=None, adj_list=None, labels=None):
        """
        初始化评估器。

        参数:
            config (Config): 配置对象，包含所有设置。
            model (torch.nn.Module, 可选): 要评估的模型。
            data (DataFrame, 可选): 数据集。
            features (ndarray, 可选): 特征矩阵。
            adj_list (list, 可选): 邻接矩阵列表。
            labels (ndarray, 可选): 真实标签。
        """
        self.config = config
        self.model = model
        self.data = data
        self.features = features
        self.adj_list = adj_list
        self.true_labels = labels

        # 结果存储
        self.embeddings = None
        self.cluster_ids = None
        self.metrics = {}
        self.cluster_stats = {}

    def set_model(self, model):
        """
        设置要评估的模型。

        参数:
            model (torch.nn.Module): 模型。
        """
        self.model = model
        return self

    def set_data(self, data, features=None, adj_list=None, labels=None):
        """
        设置评估数据。

        参数:
            data (DataFrame): 数据集。
            features (ndarray, 可选): 特征矩阵。
            adj_list (list, 可选): 邻接矩阵列表。
            labels (ndarray, 可选): 真实标签。
        """
        self.data = data
        if features is not None:
            self.features = features
        if adj_list is not None:
            self.adj_list = adj_list
        if labels is not None:
            self.true_labels = labels
        return self

    def generate_embeddings(self):
        """
        使用模型生成节点嵌入。
        """
        if self.model is None or self.features is None or self.adj_list is None:
            raise ValueError("模型、特征或邻接矩阵未设置")

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(self.features).to(self.config.device)
            adj_tensors = [torch.FloatTensor(adj).to(self.config.device) for adj in self.adj_list]

            # 获取嵌入
            embeddings, _ = self.model(x, adj_tensors)
            self.embeddings = embeddings.cpu().numpy()

        return self.embeddings

    def perform_clustering(self, n_clusters=None):
        """
        在嵌入上执行聚类。

        参数:
            n_clusters (int, 可选): 聚类数量。如果为 None，则使用配置中的最小聚类数。
        """
        if self.embeddings is None:
            self.generate_embeddings()

        if n_clusters is None:
            n_clusters = self.config.min_clusters

        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.seed, n_init=10)
        self.cluster_ids = kmeans.fit_predict(self.embeddings)

        # 存储聚类中心
        self.cluster_centers = kmeans.cluster_centers_

        return self.cluster_ids

    def evaluate(self, n_clusters=None):
        """
        评估聚类结果。

        参数:
            n_clusters (int, 可选): 聚类数量。如果为 None，则使用配置中的最小聚类数。
        """
        if self.cluster_ids is None:
            self.perform_clustering(n_clusters)

        # 计算评估指标
        self.metrics = evaluate_clustering(self.embeddings, self.cluster_ids, self.true_labels)

        # 计算聚类统计信息
        self.cluster_stats = compute_cluster_statistics(self.embeddings, self.cluster_ids)

        # 如果有真实标签，计算额外的指标
        if self.true_labels is not None:
            self.metrics['purity'] = calculate_purity(self.cluster_ids, self.true_labels)

        return self.metrics

    def find_optimal_clusters(self, min_clusters=None, max_clusters=None):
        """
        寻找最佳聚类数量。

        参数:
            min_clusters (int, 可选): 最小聚类数量。
            max_clusters (int, 可选): 最大聚类数量。
        """
        if min_clusters is None:
            min_clusters = self.config.min_clusters
        if max_clusters is None:
            max_clusters = self.config.max_clusters

        if self.embeddings is None:
            self.generate_embeddings()

        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        for n_clusters in range(min_clusters, max_clusters + 1):
            # 执行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.seed, n_init=10)
            cluster_ids = kmeans.fit_predict(self.embeddings)

            # 计算指标
            try:
                sil = silhouette_score(self.embeddings, cluster_ids)
                db = davies_bouldin_score(self.embeddings, cluster_ids)
                ch = calinski_harabasz_score(self.embeddings, cluster_ids)

                silhouette_scores.append(sil)
                davies_bouldin_scores.append(db)
                calinski_harabasz_scores.append(ch)
            except:
                silhouette_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
                calinski_harabasz_scores.append(0)

        # 找到最佳聚类数量（较高的轮廓系数，较低的 Davies-Bouldin 指标）
        best_sil_idx = np.argmax(silhouette_scores)
        best_db_idx = np.argmin(davies_bouldin_scores)
        best_ch_idx = np.argmax(calinski_harabasz_scores)

        # 使用投票机制结合结果
        cluster_votes = np.zeros(max_clusters - min_clusters + 1)
        cluster_votes[best_sil_idx] += 1
        cluster_votes[best_db_idx] += 1
        cluster_votes[best_ch_idx] += 1

        best_n_clusters = min_clusters + np.argmax(cluster_votes)

        # 如果有平局，优先选择轮廓系数
        if np.sum(cluster_votes == np.max(cluster_votes)) > 1:
            best_n_clusters = min_clusters + best_sil_idx

        # 存储分数以供后续分析
        self.cluster_scores = {
            'n_clusters_range': list(range(min_clusters, max_clusters + 1)),
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores,
            'best_n_clusters': best_n_clusters
        }

        # 使用最佳聚类数量进行最终聚类
        self.perform_clustering(best_n_clusters)
        self.evaluate()

        return best_n_clusters

    def visualize_embeddings(self, show=True, save_path=None):
        """
        使用 t-SNE 可视化节点嵌入。

        参数:
            show (bool): 是否显示图像。
            save_path (str, 可选): 图像保存路径。
        """
        if self.embeddings is None:
            self.generate_embeddings()

        # 绘制带有聚类分配的嵌入
        plt_fig = plot_embeddings(
            self.embeddings,
            self.cluster_ids if hasattr(self, 'cluster_ids') else None,
            title='节点嵌入与聚类分配'
        )

        # 如果有真实标签，也按真实标签可视化
        if self.true_labels is not None:
            plt_fig2 = plot_embeddings(
                self.embeddings,
                self.true_labels,
                title='节点嵌入与真实标签',
                colormap='plasma'
            )

        # 保存图像
        if save_path is not None:
            plt_fig.savefig(os.path.join(save_path, 'embeddings_clusters.png'))
            if self.true_labels is not None:
                plt_fig2.savefig(os.path.join(save_path, 'embeddings_true_labels.png'))

        # 显示图像
        if show:
            plt.show()
        else:
            plt.close('all')

    def plot_clustering_metrics(self, show=True, save_path=None):
        """
        绘制不同聚类数量下的聚类指标。

        参数:
            show (bool): 是否显示图像。
            save_path (str, 可选): 图像保存路径。
        """
        if not hasattr(self, 'cluster_scores'):
            raise ValueError("没有可用的聚类分数。请先运行 find_optimal_clusters 方法。")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # 绘制轮廓系数
        ax1.plot(self.cluster_scores['n_clusters_range'], self.cluster_scores['silhouette_scores'], 'o-')
        ax1.axvline(self.cluster_scores['best_n_clusters'], color='r', linestyle='--')
        ax1.set_title('轮廓系数 vs. 聚类数量')
        ax1.set_xlabel('聚类数量')
        ax1.set_ylabel('轮廓系数')
        ax1.grid(True)

        # 绘制 Davies-Bouldin 指标
        ax2.plot(self.cluster_scores['n_clusters_range'], self.cluster_scores['davies_bouldin_scores'], 'o-')
        ax2.axvline(self.cluster_scores['best_n_clusters'], color='r', linestyle='--')
        ax2.set_title('Davies-Bouldin 指标 vs. 聚类数量')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('Davies-Bouldin 指标')
        ax2.grid(True)

        # 绘制 Calinski-Harabasz 指标
        ax3.plot(self.cluster_scores['n_clusters_range'], self.cluster_scores['calinski_harabasz_scores'], 'o-')
        ax3.axvline(self.cluster_scores['best_n_clusters'], color='r', linestyle='--')
        ax3.set_title('Calinski-Harabasz 指标 vs. 聚类数量')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('Calinski-Harabasz 指标')
        ax3.grid(True)

        plt.tight_layout()

        # 保存图像
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'clustering_metrics.png'))

        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def analyze_clusters(self):
        """
        分析聚类属性和成员特征。
        """
        if self.cluster_ids is None:
            self.perform_clustering()

        # 基本聚类统计
        unique_clusters = np.unique(self.cluster_ids)
        n_clusters = len(unique_clusters)

        print(f"聚类数量: {n_clusters}")
        print(f"聚类大小: {np.bincount(self.cluster_ids)}")

        # 创建报告 DataFrame
        cluster_report = pd.DataFrame({
            'Cluster': unique_clusters,
            'Size': [np.sum(self.cluster_ids == c) for c in unique_clusters],
            'Percentage': [np.mean(self.cluster_ids == c) * 100 for c in unique_clusters]
        })

        # 如果有原始数据，添加更多信息
        if self.data is not None:
            # 对每个聚类计算聚合统计
            for c in unique_clusters:
                cluster_data = self.data.iloc[self.cluster_ids == c]

                # 添加任何相关的统计信息
                for col in self.data.select_dtypes(include=['int64', 'float64']).columns:
                    try:
                        cluster_report.loc[cluster_report['Cluster'] == c, f'Avg_{col}'] = cluster_data[col].mean()
                    except:
                        pass

                # 对于分类列，找到最常见的值
                for col in self.data.select_dtypes(include=['object', 'category']).columns:
                    try:
                        most_common = cluster_data[col].mode()[0]
                        cluster_report.loc[cluster_report['Cluster'] == c, f'Common_{col}'] = most_common
                    except:
                        pass

        # 添加 cluster_stats 中的指标
        if hasattr(self, 'cluster_stats') and self.cluster_stats:
            for i, c in enumerate(unique_clusters):
                if i < len(self.cluster_stats['intra_dists']):
                    cluster_report.loc[cluster_report['Cluster'] == c, 'Intra_Dist'] = \
                        self.cluster_stats['intra_dists'][i]

        self.cluster_report = cluster_report
        return cluster_report

    def save_results(self, save_path):
        """
        保存评估结果。

        参数:
            save_path (str): 结果保存路径。
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存指标
        if self.metrics:
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)

        # 保存聚类分配
        if self.cluster_ids is not None:
            np.save(os.path.join(save_path, 'cluster_ids.npy'), self.cluster_ids)

        # 保存嵌入
        if self.embeddings is not None:
            np.save(os.path.join(save_path, 'embeddings.npy'), self.embeddings)

        # 保存聚类报告
        if hasattr(self, 'cluster_report'):
            self.cluster_report.to_csv(os.path.join(save_path, 'cluster_report.csv'), index=False)

        # 保存聚类分数
        if hasattr(self, 'cluster_scores'):
            scores_df = pd.DataFrame({
                'n_clusters': self.cluster_scores['n_clusters_range'],
                'silhouette': self.cluster_scores['silhouette_scores'],
                'davies_bouldin': self.cluster_scores['davies_bouldin_scores'],
                'calinski_harabasz': self.cluster_scores['calinski_harabasz_scores']
            })
            scores_df.to_csv(os.path.join(save_path, 'cluster_scores.csv'), index=False)

        # 保存可视化结果
        if self.config.plot_results:
            # 嵌入可视化
            self.visualize_embeddings(show=False, save_path=save_path)

            # 如果有真实标签，绘制混淆矩阵
            if self.true_labels is not None and self.cluster_ids is not None:
                cm_fig = plot_confusion_matrix(self.true_labels, self.cluster_ids)
                cm_fig.savefig(os.path.join(save_path, 'confusion_matrix.png'))
                plt.close()

            # 聚类指标图
            if hasattr(self, 'cluster_scores'):
                self.plot_clustering_metrics(show=False, save_path=save_path)

        print(f"结果已保存至 {save_path}")


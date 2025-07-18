import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd
import networkx as nx


def plot_embeddings(embeddings, labels=None, title='Node Embeddings', colormap='viridis', figsize=(10, 8)):
    """
    使用t-SNE可视化嵌入

    参数:
        embeddings: ndarray/tensor, 节点嵌入
        labels: ndarray, 可选的节点标签/聚类结果
        title: str, 图表标题
        colormap: str, 颜色映射
        figsize: tuple, 图像大小
    """
    # 转换为numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # 使用t-SNE降维到2D平面
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘图
    plt.figure(figsize=figsize)

    if labels is not None:
        # 创建散点图，颜色表示类别
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap=colormap,
            alpha=0.7,
            s=50
        )

        # 添加颜色条
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 20:  # 如果类别不太多，添加图例
            legend1 = plt.legend(
                *scatter.legend_elements(),
                title="Clusters",
                loc="upper right"
            )
            plt.gca().add_artist(legend1)
    else:
        # 没有标签时，使用单一颜色
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.7,
            s=50
        )

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    return plt


def plot_adjacency_matrices(adj_list, titles=None, figsize=(15, 5)):
    """
    可视化多个邻接矩阵

    参数:
        adj_list: list, 邻接矩阵列表
        titles: list, 标题列表
        figsize: tuple, 图像大小
    """
    n_matrices = len(adj_list)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)

    # 如果只有一个矩阵，axes不是数组
    if n_matrices == 1:
        axes = [axes]

    # 设置默认标题
    if titles is None:
        titles = [f'View {i + 1}' for i in range(n_matrices)]

    # 绘制每个邻接矩阵
    for i, (adj, title) in enumerate(zip(adj_list, titles)):
        # 转换为numpy数组
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()

        # 使用热图绘制
        sns.heatmap(
            adj,
            ax=axes[i],
            cmap='viridis',
            vmin=0,
            vmax=max(1, adj.max()),
            square=True
        )
        axes[i].set_title(title)

    plt.tight_layout()
    return plt


def plot_graphs(adj_list, node_features=None, node_colors=None, titles=None, figsize=(15, 5),
                node_size=100, edge_threshold=0.1, max_nodes=100):
    """
    可视化多个图

    参数:
        adj_list: list, 邻接矩阵列表
        node_features: ndarray, 节点特征
        node_colors: ndarray, 节点颜色（如聚类结果）
        titles: list, 标题列表
        figsize: tuple, 图像大小
        node_size: int, 节点大小
        edge_threshold: float, 边权重阈值
        max_nodes: int, 最大节点数（防止图太大）
    """
    n_graphs = len(adj_list)
    fig, axes = plt.subplots(1, n_graphs, figsize=figsize)

    # 如果只有一个图，axes不是数组
    if n_graphs == 1:
        axes = [axes]

    # 设置默认标题
    if titles is None:
        titles = [f'Graph {i + 1}' for i in range(n_graphs)]

    # 处理节点数
    n_nodes = adj_list[0].shape[0]
    if n_nodes > max_nodes:
        sample_idx = np.random.choice(n_nodes, max_nodes, replace=False)
    else:
        sample_idx = np.arange(n_nodes)

    # 绘制每个图
    for i, (adj, title) in enumerate(zip(adj_list, titles)):
        # 转换为numpy数组
        if isinstance(adj, torch.Tensor):
            adj = adj.detach().cpu().numpy()

        # 采样
        adj_sampled = adj[sample_idx, :][:, sample_idx]

        # 创建NetworkX图
        G = nx.Graph()

        # 添加节点
        G.add_nodes_from(range(len(sample_idx)))

        # 添加边（仅添加权重大于阈值的边）
        edges = []
        for u in range(len(sample_idx)):
            for v in range(u + 1, len(sample_idx)):
                weight = adj_sampled[u, v]
                if weight > edge_threshold:
                    edges.append((u, v, {'weight': weight}))
        G.add_edges_from(edges)

        # 设置节点颜色
        if node_colors is not None:
            colors = node_colors[sample_idx]
        else:
            colors = 'skyblue'

        # 绘制图
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos=pos,
            ax=axes[i],
            with_labels=False,
            node_color=colors,
            node_size=node_size,
            width=[G[u][v]['weight'] * 2 for u, v in G.edges()],
            alpha=0.8,
            cmap='viridis'
        )

        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    return plt


def plot_training_history(history, figsize=(10, 6)):
    """
    绘制训练历史

    参数:
        history: dict, 训练历史字典
        figsize: tuple, 图像大小
    """
    plt.figure(figsize=figsize)

    # 遍历字典中的所有指标
    for metric, values in history.items():
        plt.plot(values, label=metric)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    return plt


def plot_confusion_matrix(true_labels, predicted_labels, figsize=(8, 6)):
    """
    绘制混淆矩阵

    参数:
        true_labels: ndarray, 真实标签
        predicted_labels: ndarray, 预测标签
        figsize: tuple, 图像大小
    """
    from sklearn.metrics import confusion_matrix

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    return plt

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import os
import random
import numpy as np
from torch_geometric.data import Data
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def visualize_pyg_graph(graph_data, title="Graph Visualization", num_nodes=None, sample_nodes=500, plot_weights=True, show_labels=False, filename=None, output_csv_dir=None):
    """
    可视化一个 PyTorch Geometric 格式的图。
    该函数将 PyG 图转换为 NetworkX 图进行绘制，并可选择显示节点子集。
    同时，它会导出完整的边数据（包含权重）到 CSV 文件，以供其他工具使用。
    """
    if graph_data.get('error'):
        print(f"无法可视化图 '{title}'，因为它在构建时出错: {graph_data['error']}")
        return

    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight'] # 这是一个 tensor

    if edge_index.numel() == 0:
        print(f"图 '{title}' 为空（没有边），跳过可视化和CSV导出。")
        return

    # 尝试推断节点数量
    if num_nodes is None:
        if edge_index.numel() > 0:
            num_nodes = edge_index.max().item() + 1
        else:
            num_nodes = 0
        if num_nodes == 0:
            print(f"图 '{title}' 没有可用的节点数，跳过可视化和CSV导出。")
            return

    # --- 新增：为 CSV 导出直接从原始 PyG 数据构建边列表 ---
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(filename))[0]

        edges_data_full = []
        # 确保 edge_weight 存在且元素数量与 edge_index 的边数一致
        # edge_index 是 [2, num_edges]
        # edge_weight 应该是 [num_edges]
        if edge_weight is not None and edge_weight.numel() == edge_index.shape[1]:
            for i in range(edge_index.shape[1]): # 遍历所有边
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                weight = edge_weight[i].item() # 直接从 edge_weight tensor 中获取权重
                edges_data_full.append({'Source': src, 'Target': dst, 'Weight': weight})
            print(f"检测到 {len(edges_data_full)} 条边及对应权重，将导出完整边数据。")
        else:
            # 如果没有权重或者权重数量不匹配，则只导出源和目标
            print("警告：未检测到有效边权重或权重数量与边数量不匹配，将只导出源和目标节点。")
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                edges_data_full.append({'Source': src, 'Target': dst})

        if edges_data_full:
            edges_df_full = pd.DataFrame(edges_data_full)
            edges_csv_path = os.path.join(output_csv_dir, f"{base_filename}_full_edges.csv")
            edges_df_full.to_csv(edges_csv_path, index=False)
            print(f"完整边数据已保存到: {edges_csv_path} ({len(edges_df_full)} 条边)")
        else:
            print(f"图 '{title}' 没有边，跳过完整边数据CSV导出。")
    # --- CSV 导出结束 ---


    # --- 以下是 matplotlib 可视化部分的逻辑，保持抽样 ---
    # 这个 pyg_data_object_for_viz 用于生成抽样图进行 matplotlib 可视化
    pyg_data_object_for_viz = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
    G_viz = to_networkx(pyg_data_object_for_viz, to_undirected=False) # 先生成完整nx图

    if sample_nodes and num_nodes > sample_nodes:
        print(f"图 '{title}' 节点数过多 ({num_nodes})，正在随机抽样 {sample_nodes} 个节点及其相关边进行可视化...")
        nodes_to_sample = list(G_viz.nodes()) # 从完整图中获取所有节点
        if len(nodes_to_sample) > sample_nodes:
            sampled_nodes = random.sample(nodes_to_sample, sample_nodes)
            G_sampled = G_viz.subgraph(sampled_nodes).copy()
            G_sampled.remove_nodes_from(list(nx.isolates(G_sampled))) # 移除抽样后可能产生的孤立节点
            G_viz = G_sampled # 用抽样后的图进行可视化
        else:
            print("  抽样节点数大于或等于实际节点数，不进行抽样。")
    elif sample_nodes == 0:
        print(f"为图 '{title}' 绘制所有 {num_nodes} 个节点...")

    if not G_viz.nodes():
        print(f"抽样后图 '{title}' 变为空图，跳过可视化。")
        return

    plt.figure(figsize=(10, 8))
    # 对于 Time 图可能需要更大的迭代次数来分散
    if "Time" in title:
        pos = nx.spring_layout(G_viz, seed=42, iterations=200, k=0.1)
    else:
        pos = nx.spring_layout(G_viz, seed=42, iterations=50)

    node_color = 'skyblue'
    edge_color = 'gray'
    alpha = 0.5
    width = 0.5

    # 绘制权重逻辑不变，它只影响 matplotlib 绘图
    if plot_weights and edge_weight is not None and edge_weight.numel() > 0 and G_viz.edges():
        weights = []
        for u, v, data in G_viz.edges(data=True):
            # 这里的逻辑依然需要健壮，因为G_viz是采样后的NetworkX图，to_networkx可能如何映射权重会有差异
            # 最可靠的方式是回到原始的edge_weight来映射，但那样会更复杂
            # 暂时保持现有逻辑，因为matplotlib只是为了预览，重点在CSV导出
            if 'weight' in data:
                weights.append(data['weight'])
            elif 'edge_attr' in data and isinstance(data['edge_attr'], (torch.Tensor, np.ndarray)):
                if data['edge_attr'].ndim == 1 and data['edge_attr'].numel() == 1:
                    weights.append(data['edge_attr'].item())
                else:
                    weights.append(1.0) # 如果 edge_attr 是多维或非单元素，这里给个默认值以便可视化
            else:
                weights.append(1.0) # 如果 plot_weights 为 True 但没有找到权重，仍给个默认值

        if weights:
            min_w, max_w = min(weights), max(weights)
            if max_w > min_w:
                normalized_weights = [(w - min_w) / (max_w - min_w) for w in weights]
                edge_colors = plt.cm.viridis(normalized_weights)
                widths = [w * 3 + 0.1 for w in normalized_weights]
            else:
                edge_colors = ['gray'] * len(weights)
                widths = [0.5] * len(weights)
        else:
            edge_colors = []
            widths = []
    else:
        edge_colors = edge_color
        widths = width

    nx.draw_networkx_nodes(G_viz, pos, node_color=node_color, node_size=50, alpha=0.9)
    nx.draw_networkx_edges(G_viz, pos, edge_color=edge_colors, width=widths, alpha=alpha, arrowsize=8)

    if show_labels:
        nx.draw_networkx_labels(G_viz, pos, font_size=8, font_color='black')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图已保存到: {filename}")
        except Exception as e:
            print(f"保存图时出错: {e}")
    plt.show()


if __name__ == '__main__':
    # 为了避免 OMP 错误，建议在脚本顶部添加此行
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --- 配置区域 (根据你之前保存图的路径和文件名进行修改) ---
    GRAPH_DIR = './data'
    K_NEIGHBORS = 100
    GRAPH_PREFIX = 'adaptive_undir_knn_graph'

    try:
        original_data_path = './data/od_idv_train_sz_1022.xlsx'
        import pandas as pd
        if original_data_path.endswith('.xlsx'):
            original_df = pd.read_excel(original_data_path)
        elif original_data_path.endswith('.csv'):
            original_df = pd.read_csv(original_data_path)
        else:
            raise ValueError(f"不支持的文件格式: {original_data_path}. 请提供 .xlsx 或 .csv 文件。")

        if 'age' in original_df.columns:
            if original_df['age'].dtype == 'object':
                 original_df['age'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
            original_df.dropna(subset=['age'], inplace=True)

        num_original_nodes = len(original_df)
        print(f"从原始数据文件 '{original_data_path}' 中推断出总节点数: {num_original_nodes}")
    except FileNotFoundError:
        print(f"警告：原始数据文件 '{original_data_path}' 未找到。将尝试从图数据中推断节点数，这可能不准确。")
        num_original_nodes = None
    except Exception as e:
        print(f"警告：加载原始数据以获取节点数时出错: {e}。将尝试从图数据中推断节点数。")
        num_original_nodes = None


    output_viz_dir = './graph_visualizations'
    os.makedirs(output_viz_dir, exist_ok=True)

    output_csv_dir = './graph_csv_data'
    os.makedirs(output_csv_dir, exist_ok=True)

    graph_files = {
        'Time': os.path.join(GRAPH_DIR, f"{GRAPH_PREFIX}_time_k{K_NEIGHBORS}.pt"),
        'Space': os.path.join(GRAPH_DIR, f"{GRAPH_PREFIX}_space_k{K_NEIGHBORS}.pt"),
        'Attribute': os.path.join(GRAPH_DIR, f"{GRAPH_PREFIX}_attr_k{K_NEIGHBORS}.pt")
    }

    print("\n--- 正在加载并可视化图 ---")

    for view_name, file_path in graph_files.items():
        print(f"\n正在处理 {view_name} 图...")
        if not os.path.exists(file_path):
            print(f"错误：图文件未找到: {file_path}。跳过此图的可视化。")
            continue

        try:
            graph_data = torch.load(file_path)
            print(f"  成功加载图文件: {file_path}")

            viz_filename = os.path.join(output_viz_dir, f"{GRAPH_PREFIX}_{view_name.lower()}_viz_k{K_NEIGHBORS}.png")

            visualize_pyg_graph(
                graph_data,
                title=f"{view_name} 图 (K={K_NEIGHBORS})",
                num_nodes=num_original_nodes,
                sample_nodes=500, # 抽样500个节点用于 matplotlib 可视化
                plot_weights=True,
                show_labels=False,
                filename=viz_filename,
                output_csv_dir=output_csv_dir # 传入 CSV 输出目录
            )

        except Exception as e:
            print(f"加载或可视化 {view_name} 图时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 图可视化任务完成 ---")
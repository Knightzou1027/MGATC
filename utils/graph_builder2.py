import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, haversine_distances
from sklearn.neighbors import NearestNeighbors # For adaptive sigmas
from scipy.sparse import csr_matrix
import warnings
import os
from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix # Ensure PyG imports are correct


def time_str_to_minutes(time_str):
    """将 'HH:MM' 或 'HH:MM:SS' 格式的时间字符串转换为从午夜开始的分钟数"""
    try:
        t = pd.to_datetime(time_str, format='%H:%M:%S', errors='raise').time()
    except ValueError:
        try:
             t = pd.to_datetime(time_str, format='%H:%M', errors='raise').time()
        except (ValueError, TypeError):
            return np.nan
    return t.hour * 60 + t.minute + t.second / 60.0

def get_adaptive_sigmas(distances, k_prime=7, epsilon=1e-9):
    """
    (Same as previous correct version)
    计算每个节点的自适应带宽 (sigma)，基于其到第 k' 近邻的距离。
    """
    n = distances.shape[0]
    sigmas = np.zeros(n)
    if n <= 1: return np.full(n, epsilon)

    k_to_query = min(k_prime + 1, n -1)
    if k_to_query <= 0: k_to_query = 1

    try:
        valid_distances = np.nan_to_num(distances, nan=np.inf)
        np.fill_diagonal(valid_distances, 0)
        nbrs = NearestNeighbors(n_neighbors=k_to_query, algorithm='brute', metric='precomputed').fit(valid_distances)
        k_prime_distances = nbrs.kneighbors(valid_distances)[0][:, -1]
        k_prime_distances[np.isinf(k_prime_distances)] = np.nan
        sigmas = k_prime_distances + epsilon
        if np.isnan(sigmas).any():
            finite_distances = distances[np.isfinite(distances) & (distances > 0)]
            mean_finite_dist = np.mean(finite_distances) if len(finite_distances) > 0 else 1.0
            sigmas[np.isnan(sigmas)] = mean_finite_dist + epsilon
            # print(f"  [Warning] 部分节点的自适应 Sigma 计算失败，使用平均有限距离 {mean_finite_dist:.4f} 进行填充。") # 静默处理
        elif np.isinf(sigmas).any(): # If sigmas still contain inf after filling NaN
            finite_distances = distances[np.isfinite(distances) & (distances > 0)]
            mean_finite_dist = np.mean(finite_distances) if len(finite_distances) > 0 else 1.0
            sigmas[np.isinf(sigmas)] = mean_finite_dist + epsilon
            print(f"  [Warning] 部分节点的自适应 Sigma 为 inf，使用平均有限距离 {mean_finite_dist:.4f} 进行填充。")

    except Exception as e:
        print(f"  [Error] 计算自适应 Sigmas 时出错: {e}. 返回默认值。")
        finite_distances = distances[np.isfinite(distances) & (distances > 0)]
        mean_finite_dist = np.mean(finite_distances) if len(finite_distances) > 0 else 1.0
        sigmas = np.full(n, mean_finite_dist + epsilon)
    sigmas[sigmas < epsilon] = epsilon
    return sigmas


def adj_to_edge_index(adj):
    """(Same as provided)"""
    row, col = adj.nonzero()
    edge_index = torch.from_numpy(np.vstack((row, col))).long()
    edge_weight = torch.from_numpy(adj.data).float()
    return {'edge_index': edge_index, 'edge_weight': edge_weight}

def build_knn_from_similarity(similarity_matrix, k, self_loops=False):
    """(Same as previous correct version, optimized KNN building)"""
    n = similarity_matrix.shape[0]
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=-np.inf)
    if not self_loops:
        np.fill_diagonal(similarity_matrix, -np.inf)
    max_k = n - (1 if not self_loops else 0)
    k = min(k, max_k)
    if k <= 0:
        warnings.warn("k 必须为正数。将返回空图。")
        return csr_matrix((n, n))

    top_k_indices_unsorted = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
    row_indices_for_gather = np.arange(n)[:, None]
    top_k_similarities = similarity_matrix[row_indices_for_gather, top_k_indices_unsorted]
    indices_within_k = np.argsort(-top_k_similarities, axis=1)
    top_k_indices_sorted = top_k_indices_unsorted[row_indices_for_gather, indices_within_k]
    row_indices_final = np.arange(n).repeat(k)
    col_indices_final = top_k_indices_sorted.flatten()
    weights_final = similarity_matrix[row_indices_final, col_indices_final]
    valid_edges = weights_final > -np.inf
    row_indices_final = row_indices_final[valid_edges]
    col_indices_final = col_indices_final[valid_edges]
    weights_final = weights_final[valid_edges]
    adj = csr_matrix((weights_final, (row_indices_final, col_indices_final)), shape=(n, n))
    return adj

# --- 主要的多视图 KNN 图构建函数 (修改版 V2) ---
def build_multi_view_graphs_knn(data, k=100, k_prime_adaptive=7, # 新增 k_prime
                               s_time_col='s_time', # 只用出发时间
                               lat_col='d_lat', lon_col='d_lon', # 用于计算空间距离
                               vec_cols=None, # POI 嵌入向量列
                               output_format='pyg',
                               make_undirected=True,
                               epsilon=1e-9): # 防止除零
    """
    建多视图稀疏 KNN 图 (时间、空间、属性)。
    - 时间视图: 基于出发时间 (s_time_col)，使用向量化周期计算和自适应带宽高斯相似度。
    - 空间视图: 基于 Haversine 距离 (lat_col, lon_col)，使用自适应带宽高斯相似度。
    - 属性视图: 基于 POI 嵌入向量 (vec_cols) 的余弦相似度。

    参数:
        data (pd.DataFrame): 预处理数据。
        k (int): KNN 的 K 值。
        k_prime_adaptive (int): 计算自适应带宽使用的邻居数量 k'.
        s_time_col (str): 出发时间列名 ('HH:MM' 或 'HH:MM:SS')。
        lat_col, lon_col (str): 目的地纬度、经度列名。
        vec_cols (list): 目的地 POI 嵌入向量的列名列表。
        output_format (str): 'pyg' 或 'adj'。
        make_undirected (bool): 如果为 True 且 output_format='pyg'，则将图转为无向。
        epsilon (float): 防止计算中除零的小值.

    返回:
        list: 包含三个图的列表 ('pyg' 字典或 'adj' CSR 矩阵格式)。
    """
    n_samples = len(data)
    if n_samples == 0: raise ValueError("输入数据为空！")
    if vec_cols is None or not isinstance(vec_cols, list) or len(vec_cols) == 0:
        raise ValueError("必须提供 `vec_cols` 参数，且为非空列名列表。")
    required_cols = [s_time_col, lat_col, lon_col] + vec_cols
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 中缺少必需列: {missing_cols}")
    # 假定数据已清理，但仍做基本检查
    if data[required_cols].isnull().values.any():
         warnings.warn("输入数据中发现 NaN 值，尽管已跳过预处理步骤。这可能导致图构建错误或结果不准确。")


    print(f"为 {n_samples} 个节点构建 KNN 图 (k={k}, k_prime={k_prime_adaptive})...")

    # --- 1. 准备数据 & 计算距离 ---
    print("  计算距离矩阵...")
    # 时间距离 (使用向量化周期计算)
    try:
        s_time_minutes = data[s_time_col].apply(time_str_to_minutes).values
        # Check for NaN *after* conversion, as conversion itself might produce NaNs
        nan_mask_s = np.isnan(s_time_minutes)
        if nan_mask_s.any():
            warnings.warn(f"列 '{s_time_col}' 解析后包含 NaN 值。对应节点在时间图中可能无边或结果不准确。")

        print("    使用向量化计算时间差...")
        diff_s_abs = np.abs(s_time_minutes[:, None] - s_time_minutes).astype(float) # N x N
        time_distances = np.minimum(diff_s_abs, 1440 - diff_s_abs) # N x N
        # Ensure NaNs propagate correctly
        time_distances[nan_mask_s, :] = np.nan
        time_distances[:, nan_mask_s] = np.nan
        print("    时间距离计算完成。")

    except Exception as e:
        raise ValueError(f"处理时间列 '{s_time_col}' 时出错: {e}")

    # 空间距离 (Haversine)
    try:
        coords_deg = data[[lat_col, lon_col]].values
        # Check for NaN *before* conversion to radians
        nan_coords_mask = np.isnan(coords_deg).any(axis=1)
        if nan_coords_mask.any():
            warnings.warn(f"坐标列 '{lat_col}' 或 '{lon_col}' 包含 NaN 值。对应节点在空间图中可能无边或结果不准确。")

        # Convert to radians, NaNs will propagate
        coords_rad = np.radians(coords_deg)
        # Calculate distances, NaNs in input lead to NaNs in output
        space_distances_km = haversine_distances(coords_rad, coords_rad) * 6371
        # No need to mask again, NaNs should already be there if input was NaN
        print("    空间距离计算完成。")
    except Exception as e:
        raise ValueError(f"处理空间列 '{lat_col}', '{lon_col}' 时出错: {e}")

    # 属性数据 (POI 嵌入向量)
    try:
        vectors = data[vec_cols].values
        nan_vector_mask = np.isnan(vectors).any(axis=1)
        if nan_vector_mask.any():
            warnings.warn(f"属性嵌入列 '{vec_cols}' 中包含 NaN 值。对应节点的余弦相似度可能为 NaN。")
        print("    属性向量准备完成。")
    except Exception as e:
        raise ValueError(f"处理属性列 '{vec_cols}' 时出错: {e}")

    # --- 2. 计算自适应带宽和相似度矩阵 ---
    print("  计算自适应带宽和相似度矩阵...")
    # 时间相似度
    print("    计算时间视图相似度...")
    time_sigmas = get_adaptive_sigmas(time_distances, k_prime=k_prime_adaptive, epsilon=epsilon)
    sigma_product_time = time_sigmas[:, None] * time_sigmas[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        W_t = np.exp(- (time_distances ** 2) / (sigma_product_time + epsilon))
    # No need to mask again, NaN distances should produce NaN similarity

    # 空间相似度
    print("    计算空间视图相似度...")
    space_sigmas = get_adaptive_sigmas(space_distances_km, k_prime=k_prime_adaptive, epsilon=epsilon)
    sigma_product_space = space_sigmas[:, None] * space_sigmas[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        W_s = np.exp(- (space_distances_km ** 2) / (sigma_product_space + epsilon))
    # No need to mask again

    # 属性相似度 (余弦相似度)
    print("    计算属性视图相似度...")
    try:
        W_a = cosine_similarity(vectors) # Result might contain NaNs if input vectors have NaNs
        if np.isnan(W_a).any():
             warnings.warn("余弦相似度计算结果包含 NaN，可能是由于输入向量中的 NaN 值。")

    except Exception as e:
        print(f"  [Error] 计算余弦相似度时出错: {e}. 属性图可能不准确。")
        W_a = np.full((n_samples, n_samples), np.nan)

    # --- 3. 构建 KNN 图 (仍然是有向的) ---
    print("  构建有向 KNN 图...")
    time_adj_knn_dir = build_knn_from_similarity(W_t, k)
    print(f"    有向时间图: {time_adj_knn_dir.nnz} 条边")
    space_adj_knn_dir = build_knn_from_similarity(W_s, k)
    print(f"    有向空间图: {space_adj_knn_dir.nnz} 条边")
    attr_adj_knn_dir = build_knn_from_similarity(W_a, k)
    print(f"    有向属性图: {attr_adj_knn_dir.nnz} 条边")

    # --- 4. 格式化输出 ---
    # (Output formatting remains the same as the previous correct version)
    if output_format.lower() == 'pyg':
        print("  转换为 PyG 格式...")
        graphs = []
        adj_list = [time_adj_knn_dir, space_adj_knn_dir, attr_adj_knn_dir]
        view_names = ['Time', 'Space', 'Attribute']

        for i, adj_sparse in enumerate(adj_list):
            try:
                edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)
                graph_dict = {'edge_index': edge_index, 'edge_weight': edge_weight}
                print(f"    视图 {view_names[i]} (有向): {edge_index.shape[1]} 条边")
            except Exception as e:
                print(f"  [Error] 转换视图 {view_names[i]} 到 PyG 格式时出错: {e}")
                graph_dict = {'edge_index': torch.empty((2, 0), dtype=torch.long),
                              'edge_weight': torch.empty((0,), dtype=torch.float),
                              'error': str(e)}
            graphs.append(graph_dict)

        if make_undirected:
            print("  转换为无向图 (使用 PyG to_undirected, reduce='mean')...")
            undirected_graphs = []
            for i, graph_dir in enumerate(graphs):
                 if 'error' in graph_dir:
                     print(f"    跳过视图 {view_names[i]} 的无向化，因为它在转换时出错。")
                     undirected_graphs.append({
                         'edge_index': torch.empty((2, 0), dtype=torch.long),
                         'edge_weight': torch.empty((0,), dtype=torch.float),
                         'error': graph_dir.get('error', 'Skipped due to previous error.')
                     })
                     continue

                 num_nodes = n_samples
                 num_edges_directed = graph_dir['edge_index'].shape[1]
                 try:
                    edge_index_undir, edge_weight_undir = to_undirected(
                        graph_dir['edge_index'], graph_dir['edge_weight'], num_nodes=num_nodes, reduce="mean"
                    )
                    undirected_graphs.append({
                        'edge_index': edge_index_undir,
                        'edge_weight': edge_weight_undir
                    })
                    print(f"    视图 {view_names[i]} (无向): 从 {num_edges_directed} 条边变为 {edge_index_undir.shape[1]} 条边")
                 except Exception as e:
                    print(f"  [Error] 视图 {view_names[i]} 无向化时出错: {e}")
                    undirected_graphs.append({
                         'edge_index': torch.empty((2, 0), dtype=torch.long),
                         'edge_weight': torch.empty((0,), dtype=torch.float),
                         'error': f'Undirected conversion failed: {e}'
                     })
            graphs = undirected_graphs

        print("图构建完成。")
        return graphs

    elif output_format.lower() == 'adj':
        if make_undirected:
             warnings.warn("make_undirected=True 对 output_format='adj' 暂未实现对称化，仍返回有向邻接矩阵。")
        print("图构建完成 (返回 SciPy CSR 邻接矩阵)。")
        return [time_adj_knn_dir, space_adj_knn_dir, attr_adj_knn_dir]
    else:
        raise ValueError(f"未知的 output_format: {output_format}。请选择 'pyg' 或 'adj'。")


# --- 主程序入口：用于直接运行此脚本进行图构建和保存 ---
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning) # 忽略一些pandas/numpy的警告
    warnings.filterwarnings("ignore", category=RuntimeWarning) # 忽略计算中可能出现的 RuntimeWarning

    # --- 配置区域 ---
    PREPROCESSED_DATA_PATH = 'Q:\LEARNNING-POSTGRADUATE\BaiduSyncdisk\LEARNNING-POSTGRADUATE\站点深化\ATP inferred\MGGDC_1018\data\od_idv_train_ptype0_gz_1018.xlsx'
    # !! 定义你的目的地 POI 嵌入向量列名 !!
    VECTOR_COLUMNS = [f'vec_{i+1}' for i in range(100)]
    # !! 定义图数据保存的目录和文件名前缀 !!
    OUTPUT_DIR = './data'
    # 修改文件名前缀以反映自适应带宽和无向图
    GRAPH_PREFIX = 'adaptive_undir_knn_graph_gz_ptype0'
    # !! 设置 KNN 的 K 值 !!
    K_NEIGHBORS = 100
    # !! 设置自适应带宽的 k' 值 !!
    K_PRIME_ADAPTIVE = 7

    # !! 检查并按需修改列名映射 (适配简化后的特征) !!
    COLUMN_MAPPING = {
        's_time_col': 's_time',      # 出发时间列
        'lat_col': 'd_lat',          # 目的地纬度列
        'lon_col': 'd_lon',          # 目的地经度列
        'vec_cols': VECTOR_COLUMNS   # POI 嵌入向量列列表
    }
    # -----------------

    print(f"--- 开始执行图构建任务 (在 graph_builder.py 中) ---")
    print(f"将从以下文件加载数据: {PREPROCESSED_DATA_PATH}")
    print(f"将使用 K = {K_NEIGHBORS}, K' = {K_PRIME_ADAPTIVE} 构建 KNN 图")
    print(f"图数据将保存到目录: {OUTPUT_DIR} (格式: 无向 PyG)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 1. 加载数据
        print("\n步骤 1: 加载预处理数据...")
        if not os.path.exists(PREPROCESSED_DATA_PATH):
             raise FileNotFoundError(f"数据文件未找到: {PREPROCESSED_DATA_PATH}")
        # 根据文件扩展名选择加载方式
        if PREPROCESSED_DATA_PATH.endswith('.xlsx'):
            processed_df = pd.read_excel(PREPROCESSED_DATA_PATH)
        elif PREPROCESSED_DATA_PATH.endswith('.csv'):
            processed_df = pd.read_csv(PREPROCESSED_DATA_PATH) # 可以加 encoding='gbk' 等参数
        else:
            raise ValueError(f"不支持的文件格式: {PREPROCESSED_DATA_PATH}. 请提供 .xlsx 或 .csv 文件。")
        print(f"数据加载成功，形状: {processed_df.shape}")
        if processed_df.empty: raise ValueError("加载的数据为空！")

        # 检查所需列 (基于更新后的 COLUMN_MAPPING)
        required_load_cols = [COLUMN_MAPPING['s_time_col'],
                              COLUMN_MAPPING['lat_col'], COLUMN_MAPPING['lon_col']
                             ] + COLUMN_MAPPING['vec_cols']
        missing_cols = [col for col in required_load_cols if col not in processed_df.columns]
        if missing_cols: raise ValueError(f"数据中缺少以下必需列: {missing_cols}")

        # 步骤 1.5: 检查 NaN (但不填充) - 如果你的数据确实干净，可以注释掉这部分检查
        print("\n步骤 1.5: 检查数据中是否存在 NaN (不进行填充)...")
        if processed_df[required_load_cols].isnull().values.any():
            print("  警告: 在所需列中发现 NaN 值！图构建可能会失败或结果不准确。")
            print(processed_df[required_load_cols].isnull().sum())
            # raise ValueError("输入数据中包含 NaN 值，无法继续图构建。") # 或者直接报错停止
        else:
            print("  未在所需列中检测到 NaN 值。")


        # 2. 构建 无向 图
        print("\n步骤 2: 构建多视图 KNN 无向图...")
        list_of_undirected_graphs_pyg = build_multi_view_graphs_knn(
            processed_df,
            k=K_NEIGHBORS,
            k_prime_adaptive=K_PRIME_ADAPTIVE, # <-- 传递 k'
            output_format='pyg',
            make_undirected=True, # <-- 确保调用转换
            s_time_col=COLUMN_MAPPING['s_time_col'], # 显式传递，避免混淆
            lat_col=COLUMN_MAPPING['lat_col'],
            lon_col=COLUMN_MAPPING['lon_col'],
            vec_cols=COLUMN_MAPPING['vec_cols']
        )

        # 3. 保存 **无向** 图数据
        print("\n步骤 3: 保存无向图数据 (PyG 格式)...")
        # 更新文件名以包含 k'
        time_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_time_k{K_NEIGHBORS}_kprime{K_PRIME_ADAPTIVE}.pt")
        space_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_space_k{K_NEIGHBORS}_kprime{K_PRIME_ADAPTIVE}.pt")
        # 属性图与 k' 无关
        attr_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_attr_k{K_NEIGHBORS}.pt")

        # 检查返回的图列表是否包含错误
        save_success = True
        if isinstance(list_of_undirected_graphs_pyg[0], dict) and 'error' not in list_of_undirected_graphs_pyg[0]:
             # 检查 edge_index 是否为空，避免保存空图 (如果需要)
             if list_of_undirected_graphs_pyg[0]['edge_index'].shape[1] > 0:
                 torch.save(list_of_undirected_graphs_pyg[0], time_graph_path)
                 print(f"  无向时间图已保存到: {time_graph_path}")
             else:
                 print(f"  警告: 无向时间图为空 (0条边)，未保存。")
                 save_success = False # 标记为部分失败
        else:
             print(f"  错误: 未能构建或保存时间图，原因: {list_of_undirected_graphs_pyg[0].get('error', '未知错误')}")
             save_success = False

        if isinstance(list_of_undirected_graphs_pyg[1], dict) and 'error' not in list_of_undirected_graphs_pyg[1]:
            if list_of_undirected_graphs_pyg[1]['edge_index'].shape[1] > 0:
                torch.save(list_of_undirected_graphs_pyg[1], space_graph_path)
                print(f"  无向空间图已保存到: {space_graph_path}")
            else:
                print(f"  警告: 无向空间图为空 (0条边)，未保存。")
                save_success = False
        else:
            print(f"  错误: 未能构建或保存空间图，原因: {list_of_undirected_graphs_pyg[1].get('error', '未知错误')}")
            save_success = False

        if isinstance(list_of_undirected_graphs_pyg[2], dict) and 'error' not in list_of_undirected_graphs_pyg[2]:
            if list_of_undirected_graphs_pyg[2]['edge_index'].shape[1] > 0:
                torch.save(list_of_undirected_graphs_pyg[2], attr_graph_path)
                print(f"  无向属性图已保存到: {attr_graph_path}")
            else:
                print(f"  警告: 无向属性图为空 (0条边)，未保存。")
                save_success = False
        else:
            print(f"  错误: 未能构建或保存属性图，原因: {list_of_undirected_graphs_pyg[2].get('error', '未知错误')}")
            save_success = False


        if save_success:
            print("\n--- 图构建任务成功完成 ---")
        else:
            print("\n--- 图构建任务完成，但部分图未能成功构建或保存 (可能为空或出错) ---")


    except FileNotFoundError as fnf:
        print(f"\n错误: {fnf}")
    except ValueError as ve:
        print(f"\n处理错误: {ve}")
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        import traceback
        traceback.print_exc()
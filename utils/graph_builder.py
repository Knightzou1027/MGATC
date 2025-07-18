import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, haversine_distances
from scipy.sparse import csr_matrix
import datetime
import warnings
import os # 用于保存文件
from torch_geometric.utils import to_undirected # <--- 新增导入

# --- 时间字符串转分钟 ---
def time_str_to_minutes(time_str):
    """将 'HH:MM' 格式的时间字符串转换为从午夜开始的分钟数"""
    try:
        t = pd.to_datetime(time_str, format='%H:%M', errors='raise').time()
        return t.hour * 60 + t.minute
    except (ValueError, TypeError): # 处理无效格式或非字符串输入
        return np.nan

# --- 邻接矩阵转 Edge Index ---
def adj_to_edge_index(adj):
    """将稀疏邻接矩阵 (SciPy CSR) 转为 PyG 的 edge_index 和 edge_weight"""
    row, col = adj.nonzero()
    edge_index = torch.from_numpy(np.vstack((row, col))).long()
    edge_weight = torch.from_numpy(adj.data).float()
    # 返回字典格式，便于后续处理
    return {'edge_index': edge_index, 'edge_weight': edge_weight}

# --- KNN 稀疏化辅助函数 ---
def build_knn_from_similarity(similarity_matrix, k, self_loops=False):
    """从相似度矩阵构建稀疏 **有向** KNN 图""" #<-- 明确这里是有向的
    n = similarity_matrix.shape[0]
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=-np.inf)

    if not self_loops:
        np.fill_diagonal(similarity_matrix, -np.inf)

    k = min(k, n - (1 if not self_loops else 0))
    if k <= 0:
        warnings.warn("k 必须为正数。将返回空图。")
        return csr_matrix((n, n))

    top_k_indices = np.argpartition(similarity_matrix, -k, axis=1)[:, -k:]
    row_indices = np.arange(n).repeat(k)
    col_indices = top_k_indices.flatten()
    weights = similarity_matrix[row_indices, col_indices]

    valid_edges = weights > -np.inf
    row_indices = row_indices[valid_edges]
    col_indices = col_indices[valid_edges]
    weights = weights[valid_edges]

    adj = csr_matrix((weights, (row_indices, col_indices)), shape=(n, n))
    return adj # 返回有向邻接矩阵

# --- 主要的多视图 KNN 图构建函数 ---
def build_multi_view_graphs_knn(data, k=10,
                               s_time_col='s_time', a_time_col='a_time',
                               lat_col='d_lat', lon_col='d_lon',
                               vec_cols=None,
                               age_col='age', trd_col='total_residence_days',
                               output_format='pyg',
                               make_undirected=True): # <--- 新增参数控制是否转为无向
    """
    构建多视图稀疏 KNN 图 (时间、空间、属性)。

    参数:
        data (pd.DataFrame): 预处理数据。
        k (int): KNN 的 K 值。
        s_time_col, a_time_col, lat_col, lon_col, age_col, trd_col (str): 列名。
        vec_cols (list): 目的地嵌入向量的列名列表。
        output_format (str): 'pyg' 或 'adj'。
        make_undirected (bool): 如果为 True 且 output_format='pyg'，则将图转为无向。

    返回:
        list: 包含三个图的列表 ('pyg' 或 'adj' 格式)。
              如果 make_undirected=True 且 output_format='pyg', 返回的是无向图。
    """
    n_samples = len(data)
    if n_samples == 0: raise ValueError("输入数据为空！")
    if vec_cols is None or not isinstance(vec_cols, list) or len(vec_cols) == 0:
        raise ValueError("必须提供 `vec_cols` 参数，且为非空列名列表。")

    print(f"为 {n_samples} 个节点构建 KNN 图 (k={k})...")

    # --- 1. 准备工作和带宽计算 ---
    print("准备数据并计算带宽...")
    # (时间、空间、属性的带宽计算代码同上一个版本，此处省略以保持简洁)
    # ... (省略带宽计算代码) ...
    # 确保以下变量已计算: bw_ts, bw_ta, bw_s, bw_trd
    # 时间数据 ('HH:MM' 转分钟) 和带宽
    try:
        s_time_minutes = data[s_time_col].apply(time_str_to_minutes).values
        a_time_minutes = data[a_time_col].apply(time_str_to_minutes).values

        if np.isnan(s_time_minutes).any() or np.isnan(a_time_minutes).any():
            nan_s = np.isnan(s_time_minutes).sum()
            nan_a = np.isnan(a_time_minutes).sum()
            warnings.warn(f"时间列解析后包含 NaN 值 ({s_time_col}: {nan_s}, {a_time_col}: {nan_a})。这些行在时间图中将没有边。")

        print("  使用向量化计算时间差...")
        diff_s_abs = np.abs(s_time_minutes[:, None] - s_time_minutes).astype(float)
        diff_a_abs = np.abs(a_time_minutes[:, None] - a_time_minutes).astype(float)
        diff_s_circ = np.minimum(diff_s_abs, 1440 - diff_s_abs)
        diff_a_circ = np.minimum(diff_a_abs, 1440 - diff_a_abs)
        nan_mask_s = np.isnan(s_time_minutes)[:, None] | np.isnan(s_time_minutes)[None, :]
        nan_mask_a = np.isnan(a_time_minutes)[:, None] | np.isnan(a_time_minutes)[None, :]
        diff_s_circ[nan_mask_s] = np.nan
        diff_a_circ[nan_mask_a] = np.nan
        print("  时间差计算完成。")

        bw_ts = np.nanmean(diff_s_circ[diff_s_circ > 0]) if np.any(diff_s_circ > 0) else 60.0
        bw_ta = np.nanmean(diff_a_circ[diff_a_circ > 0]) if np.any(diff_a_circ > 0) else 60.0
        print(f"  时间带宽 (循环分钟差): bw_ts={bw_ts:.2f} 分钟, bw_ta={bw_ta:.2f} 分钟")

    except Exception as e:
        raise ValueError(f"处理时间列 ('{s_time_col}', '{a_time_col}') 时出错: {e}")

    # 空间数据 (坐标转弧度) 和带宽
    try:
        coords_rad = np.radians(data[[lat_col, lon_col]].values)
        vectors = data[vec_cols].values
        if np.isnan(coords_rad).any(): raise ValueError("坐标列中发现 NaN")
        if np.isnan(vectors).any(): raise ValueError("向量列中发现 NaN")

        dist_h_rad = haversine_distances(coords_rad)
        dist_h_km = dist_h_rad * 6371
        bw_s = np.mean(dist_h_km[dist_h_km > 0]) if np.any(dist_h_km > 0) else 1.0
        print(f"  空间带宽 (距离): bw_s={bw_s:.2f} 公里")
    except Exception as e:
        raise ValueError(f"处理空间列 ('{lat_col}', '{lon_col}', vec_cols) 时出错: {e}")

    # 属性数据和带宽
    try:
        age_vals = data[age_col].values
        trd_vals = pd.to_numeric(data[trd_col], errors='coerce').values
        if np.isnan(trd_vals).any():
            raise ValueError(f"列 '{trd_col}' 在转为数值型后包含 NaN。")

        print("  使用向量化计算 TRD 差值...")
        diff_trd_all = np.abs(trd_vals[:, None] - trd_vals).astype(float)
        bw_trd = np.nanmean(diff_trd_all[diff_trd_all > 0]) if np.any(diff_trd_all > 0) else 1.0
        print(f"  属性带宽 (TRD): bw_trd={bw_trd:.2f}")
        print("  TRD 差值计算完成。")
    except Exception as e:
        raise ValueError(f"处理属性列 ('{age_col}', '{trd_col}') 时出错: {e}")


    # --- 2. 计算相似度矩阵 ---
    print("计算相似度矩阵...")
    # (时间、空间、属性的相似度计算代码同上一个版本，此处省略以保持简洁)
    # ... (省略相似度计算代码) ...
    # 确保 W_t, W_s, W_a 已计算
    # 时间相似度 (W_t)
    print("  计算时间相似度 (基于循环分钟差)...")
    with np.errstate(divide='ignore', invalid='ignore'):
        w_s = np.exp(-diff_s_circ**2 / (bw_ts**2 + 1e-9))
        w_a = np.exp(-diff_a_circ**2 / (bw_ta**2 + 1e-9))
    W_t = w_s + w_a

    # 空间相似度 (W_s)
    print("  计算空间相似度...")
    with np.errstate(divide='ignore', invalid='ignore'):
        w_h = np.exp(-dist_h_km**2 / (bw_s**2 + 1e-9))
    sim_v = cosine_similarity(vectors)
    sim_v_scaled = (sim_v + 1) / 2
    W_s = w_h + sim_v_scaled

    # 属性相似度 (W_a)
    print("  计算属性相似度...")
    sim_age = (age_vals[:, None] == age_vals).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_trd = np.exp(-diff_trd_all**2 / (bw_trd**2 + 1e-9))
    W_a = sim_age + w_trd


    # --- 3. 构建 KNN 图 (仍然是有向的) ---
    print("构建有向 KNN 图...")
    time_adj_knn_dir = build_knn_from_similarity(W_t, k)
    print(f"  有向时间图: {time_adj_knn_dir.nnz} 条边")
    space_adj_knn_dir = build_knn_from_similarity(W_s, k)
    print(f"  有向空间图: {space_adj_knn_dir.nnz} 条边")
    attr_adj_knn_dir = build_knn_from_similarity(W_a, k)
    print(f"  有向属性图: {attr_adj_knn_dir.nnz} 条边")

    # --- 4. 格式化输出 ---
    if output_format.lower() == 'pyg':
        print("转换为 PyG 格式...")
        graph_t_dir = adj_to_edge_index(time_adj_knn_dir)
        graph_s_dir = adj_to_edge_index(space_adj_knn_dir)
        graph_a_dir = adj_to_edge_index(attr_adj_knn_dir)

        graphs = [graph_t_dir, graph_s_dir, graph_a_dir]

        # --- 4.1 (新增) 如果需要，转换为无向图 ---
        if make_undirected:
            print("转换为无向图...")
            undirected_graphs = []
            for i, graph_dir in enumerate(graphs):
                num_nodes = n_samples # 假设所有节点都包含
                edge_index_undir, edge_weight_undir = to_undirected(
                    graph_dir['edge_index'], graph_dir['edge_weight'], num_nodes=num_nodes
                )
                undirected_graphs.append({
                    'edge_index': edge_index_undir,
                    'edge_weight': edge_weight_undir
                })
                print(f"  视图 {i} (无向): {edge_index_undir.shape[1]} 条边")
            graphs = undirected_graphs # 使用无向图结果

        print("图构建完成。")
        return graphs # 返回包含 (无向) 图字典的列表

    elif output_format.lower() == 'adj':
        # 注意：如果需要无向邻接矩阵，也需要在此处进行对称化处理
        # 例如: time_adj_knn_undir = time_adj_knn_dir + time_adj_knn_dir.T (可能需要处理值)
        # 为保持一致，此处暂时仍返回有向矩阵，如需无向请修改
        if make_undirected:
             warnings.warn("make_undirected=True 对 output_format='adj' 暂未实现对称化，仍返回有向邻接矩阵。")
        print("图构建完成。")
        return [time_adj_knn_dir, space_adj_knn_dir, attr_adj_knn_dir]
    else:
        raise ValueError(f"未知的 output_format: {output_format}。请选择 'pyg' 或 'adj'。")


# --- 主程序入口：用于直接运行此脚本进行图构建和保存 ---
if __name__ == '__main__':

    # --- 配置区域 ---
    # !! 修改为你实际的预处理数据文件路径 !!
    PREPROCESSED_DATA_PATH = 'data/od_idv_train_sz_1022.xlsx' # 示例路径
    # !! 定义你的目的地向量列名 !!
    VECTOR_COLUMNS = [f'vec_{i+1}' for i in range(100)]
    # !! 定义图数据保存的目录和文件名前缀 !!
    OUTPUT_DIR = './data'
    GRAPH_PREFIX = 'undir_knn_graph' # <--- 修改文件名前缀以反映无向
    # !! 设置 KNN 的 K 值 !!
    K_NEIGHBORS = 100

    # !! 检查并按需修改列名映射 !!
    COLUMN_MAPPING = {
        's_time_col': 's_time',
        'a_time_col': 'a_time',
        'lat_col': 'd_lat',
        'lon_col': 'd_lon',
        'age_col': 'age',
        'trd_col': 'total_residence_days',
        'vec_cols': VECTOR_COLUMNS
    }
    # -----------------

    print(f"--- 开始执行图构建任务 (在 graph_builder.py 中) ---")
    print(f"将从以下文件加载数据: {PREPROCESSED_DATA_PATH}")
    print(f"将使用 K = {K_NEIGHBORS} 构建 KNN 图")
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

        # 检查所需列
        required_load_cols = [COLUMN_MAPPING['s_time_col'], COLUMN_MAPPING['a_time_col'],
                              COLUMN_MAPPING['lat_col'], COLUMN_MAPPING['lon_col'],
                              COLUMN_MAPPING['age_col'], COLUMN_MAPPING['trd_col']] + COLUMN_MAPPING['vec_cols']
        missing_cols = [col for col in required_load_cols if col not in processed_df.columns]
        if missing_cols: raise ValueError(f"数据中缺少以下必需列: {missing_cols}")

        # 2. 构建 **无向** 图 (因为 build_multi_view_graphs_knn 内部会调用 to_undirected)
        print("\n步骤 2: 构建多视图 KNN 无向图...")
        list_of_undirected_graphs_pyg = build_multi_view_graphs_knn(
            processed_df,
            k=K_NEIGHBORS,
            output_format='pyg',
            make_undirected=True, # <-- 确保调用转换
            **COLUMN_MAPPING
        )

        # 3. 保存 **无向** 图数据
        print("\n步骤 3: 保存无向图数据 (PyG 格式)...")
        time_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_time_k{K_NEIGHBORS}.pt")
        space_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_space_k{K_NEIGHBORS}.pt")
        attr_graph_path = os.path.join(OUTPUT_DIR, f"{GRAPH_PREFIX}_attr_k{K_NEIGHBORS}.pt")

        torch.save(list_of_undirected_graphs_pyg[0], time_graph_path)
        print(f"  无向时间图已保存到: {time_graph_path}")
        torch.save(list_of_undirected_graphs_pyg[1], space_graph_path)
        print(f"  无向空间图已保存到: {space_graph_path}")
        torch.save(list_of_undirected_graphs_pyg[2], attr_graph_path)
        print(f"  无向属性图已保存到: {attr_graph_path}")

        print("\n--- 图构建任务完成 ---")

    except FileNotFoundError as fnf:
        print(f"\n错误: {fnf}")
    except ValueError as ve:
        print(f"\n处理错误: {ve}")
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        import traceback
        traceback.print_exc()
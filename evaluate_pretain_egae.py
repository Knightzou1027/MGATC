import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # 用于加载数据时可能需要的标准化

# 导入自定义模块
from models.mgat import MultiViewGAT # 使用你修改后的 mgat 模块
from models.clustering_egae import EGACECLoss # 导入新的EGAE聚类损失模块
from utils.metrics import evaluate_clustering, evaluate_clustering_comprehensive, compute_clustering_quality # 导入无标签评估函数

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_pretrain_egae(args):
    """
    加载预训练模型，获取嵌入，计算指示矩阵并使用KMeans聚类，进行评估。
    """
    logging.info("Starting evaluation of pretrain embeddings using EGAE clustering logic...")

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. 加载节点特征 (需要与训练时一致，确保标准化等操作相同)
    logging.info(f"Loading node features from: {args.feature_data_path}")
    try:
        if args.feature_data_path.endswith('.xlsx'):
            df_features = pd.read_excel(args.feature_data_path)
        else:
             raise ValueError(f"Unsupported feature file format: {args.feature_data_path}")

        logging.info(f"Feature data loaded successfully, shape: {df_features.shape}")

        FEATURE_COLUMN_NAMES = [
            # --- 96 个时间块特征列名 --- (这里应该与 train.py 中一致)
             'se_block_0', 'se_block_1', 'se_block_2', 'se_block_3', 'se_block_4', 'se_block_5', 'se_block_6','se_block_7',
            'se_block_8', 'se_block_9', 'se_block_10', 'se_block_11', 'se_block_12', 'se_block_13', 'se_block_14','se_block_15',
            'se_block_16', 'se_block_17', 'se_block_18', 'se_block_19', 'se_block_20', 'se_block_21', 'se_block_22','se_block_23',
            'se_block_24', 'se_block_25', 'se_block_26', 'se_block_27', 'se_block_28', 'se_block_29', 'se_block_30','se_block_31',
            'se_block_32', 'se_block_33', 'se_block_34', 'se_block_35', 'se_block_36', 'se_block_37', 'se_block_38','se_block_39',
            'se_block_40', 'se_block_41', 'se_block_42', 'se_block_43', 'se_block_44', 'se_block_45', 'se_block_46','se_block_47',
            'ea_block_0', 'ea_block_1', 'ea_block_2', 'ea_block_3', 'ea_block_4', 'ea_block_5', 'ea_block_6','ea_block_7',
            'ea_block_8', 'ea_block_9', 'ea_block_10', 'ea_block_11', 'ea_block_12', 'ea_block_13', 'ea_block_14','ea_block_15',
            'ea_block_16', 'ea_block_17', 'ea_block_18', 'ea_block_19', 'ea_block_20', 'ea_block_21', 'ea_block_22','ea_block_23',
            'ea_block_24', 'ea_block_25', 'ea_block_26', 'ea_block_27', 'ea_block_28', 'ea_block_29', 'ea_block_30','ea_block_31',
            'ea_block_32', 'ea_block_33', 'ea_block_34', 'ea_block_35', 'ea_block_36', 'ea_block_37', 'ea_block_38','ea_block_39',
            'ea_block_40', 'ea_block_41', 'ea_block_42', 'ea_block_43', 'ea_block_44', 'ea_block_45', 'ea_block_46','ea_block_47',
            # --- 19 个吸引力特征列名 ---
            '事务所_G_attractiveness', '产业园工厂_G_attractiveness','体育休闲_G_attractiveness','公司企业_G_attractiveness','医疗_G_attractiveness',
            '商务楼_G_attractiveness','固定性住宿_G_attractiveness','展馆_G_attractiveness','度假_G_attractiveness','政府机构_G_attractiveness',
            '流动性住宿_G_attractiveness','社会团体_G_attractiveness','考试培训_G_attractiveness','购物_G_attractiveness','酒店会议中心_G_attractiveness',
            '金融保险_G_attractiveness','风景_G_attractiveness','餐饮_G_attractiveness','高校科研_G_attractiveness'
        ]

        features_np = df_features[FEATURE_COLUMN_NAMES].values.astype(np.float32)

        # !!! 确保这里是否标准化与训练时一致 !!!
        # 根据你的日志，训练时 scale_features=False，所以这里不标准化
        # if args.scale_features:
        #     logging.info("Applying StandardScaler to features...")
        #     scaler = StandardScaler()
        #     features_np = scaler.fit_transform(features_np)
        #     logging.info("Features scaled.")

        features = torch.FloatTensor(features_np).to(device)
        logging.info(f"Node features tensor created, shape: {features.shape}")
        num_nodes = features.shape[0]

    except Exception as e:
         logging.error(f"Error loading features: {e}")
         return


    # 2. 加载图数据 (需要与训练时一致)
    logging.info(f"Loading pre-built graphs from directory: {args.graph_dir}")
    try:
        adj_list = []
        graph_views = ['time', 'space', 'attr']
        for view in graph_views:
            graph_file = f"{args.graph_prefix}_{view}_k{args.knn_k}.pt"
            graph_path = os.path.join(args.graph_dir, graph_file) # 路径相对于 CWD
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            graph_data = torch.load(graph_path, map_location=device)
            graph_data['edge_weight'] = graph_data['edge_weight'].to(device).to(torch.float32)
            if 'edge_index' not in graph_data or 'edge_weight' not in graph_data:
                 raise ValueError(f"Graph file {graph_path} is missing 'edge_index' or 'edge_weight'.")

            graph_data['edge_index'] = graph_data['edge_index'].to(device)
            graph_data['edge_weight'] = graph_data['edge_weight'].to(device)

            adj_list.append(graph_data)
            logging.info(f"  Loaded {view} graph: {graph_data['edge_index'].shape[1]} edges.")
    except Exception as e:
        logging.error(f"Error loading graphs: {e}")
        return


    # 3. 初始化模型 (只需要编码器)
    logging.info("Initializing encoder model...")
    encoder = MultiViewGAT(
        in_features=features.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout, # 评估时dropout无效
        alpha=args.alpha,
        num_views=len(adj_list)
    ).to(device)
    encoder.float()

    # 4. 加载预训练模型的参数
    pretrain_model_path = os.path.join(args.output_path, 'pretrain_encoder.pth')
    if not os.path.exists(pretrain_model_path):
        # 如果运行独立脚本，需要先确定 output_path 的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir_abs = os.path.join(script_dir, args.output_path)
        pretrain_model_path_abs = os.path.join(output_dir_abs, 'pretrain_encoder.pth')
        if not os.path.exists(pretrain_model_path_abs):
             logging.error(f"Pretrain model not found at {pretrain_model_path} or {pretrain_model_path_abs}")
             return
        else:
            pretrain_model_path = pretrain_model_path_abs # 使用绝对路径加载

    logging.info(f"Loading pretrain model from: {pretrain_model_path}")
    try:
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        logging.info("Pretrain model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading pretrain model: {e}")
        return


    # 5. 获取预训练嵌入
    encoder.eval() # 评估模式
    with torch.no_grad():
        pretrain_embeddings, _, pretrain_view_weights = encoder(features, adj_list)
    logging.info(f"Pretrain embeddings shape: {pretrain_embeddings.shape}")


    # 6. 计算指示矩阵并使用 K-Means 聚类 (EGAE 逻辑)
    logging.info("Evaluating pretrain embeddings using EGAE clustering logic (SVD + KMeans on P)...")

    # 初始化 EGAE 聚类损失模块 (用于计算 P 和 K-Means)
    clustering_loss_module = EGACECLoss(
        embedding_dim=args.embedding_dim,
        n_clusters=args.n_clusters,
        device=device
    ).to(device)

    # 更新指示矩阵 P (这将执行 ZZ^T 和 SVD)
    logging.info("Updating indicator matrix P based on pretrain embeddings...")
    clustering_loss_module.update_indicator(pretrain_embeddings)
    logging.info(f"Indicator matrix P shape: {clustering_loss_module.indicator.shape}")


    # 在指示矩阵 P 上运行 K-Means 获取聚类分配
    logging.info("Running KMeans on indicator matrix P...")
    cluster_ids_egae, _ = clustering_loss_module.get_cluster_assignments(pretrain_embeddings) # embeddings 参数用于获取数据大小和设备
    logging.info("KMeans on indicator matrix P finished.")

    # 7. 评估聚类结果
    logging.info("Evaluating clustering results from KMeans on indicator matrix P:")
    embeddings_np = pretrain_embeddings.cpu().numpy() # 评估指标需要 numpy 格式的嵌入
    cluster_ids_egae_np = cluster_ids_egae

    # 计算并打印评估指标
    metrics = evaluate_clustering(embeddings_np, cluster_ids_egae_np)
    quality_score = compute_clustering_quality(embeddings_np, cluster_ids_egae_np)

    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    logging.info(f"Quality Score: {quality_score:.4f}")

    # 计算簇大小并打印，以检查是否已经出现大簇
    unique_clusters, cluster_counts = np.unique(cluster_ids_egae_np, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, cluster_counts))
    logging.info(f"Cluster sizes: {cluster_sizes}")
    logging.info(f"Max cluster percentage: {np.max(cluster_counts) / num_nodes * 100:.2f}%")


    # 8. (可选) 保存结果
    logging.info("Saving evaluation results...")
    results_df = df_features.copy()
    results_df['cluster'] = cluster_ids_egae_np

    # 添加嵌入向量 (可选，如果想保存预训练嵌入)
    # for i in range(args.embedding_dim):
    #     results_df[f'embedding_{i}'] = embeddings_np[:, i]

    # 添加视图权重 (可选)
    view_weights_np = pretrain_view_weights.cpu().numpy() if pretrain_view_weights is not None else np.full((num_nodes, args.num_views), np.nan)
    for v in range(view_weights_np.shape[1]):
         results_df[f'weight_view_{v}'] = view_weights_np[:, v]

    output_csv_path = os.path.join(args.output_path, 'pretrain_egae_kmeans_clustering_results.csv')
    results_df.to_csv(output_csv_path, index=False)
    logging.info(f"Evaluation results saved to {output_csv_path}")


# 如果作为独立脚本运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pretrain Model with EGAE Clustering Logic')
    # 复制 train.py 中的参数定义，只需要模型和数据相关的参数
    parser.add_argument('--feature_data_path', type=str, default='./data/od_idv_train_sz_1022.xlsx')
    parser.add_argument('--graph_dir', type=str, default='./data')
    parser.add_argument('--graph_prefix', type=str, default='adaptive_undir_knn_graph')
    parser.add_argument('--knn_k', type=int, default=100)
    parser.add_argument('--output_path', type=str, default='./results/') # 输出路径用于加载模型和保存结果
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1) # 评估时dropout无效
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--n_clusters', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--scale_features', action='store_true') # 需要与训练时一致
    # 注意：这里的参数值必须与你运行 train.py 时使用的参数值一致，特别是模型相关的！

    args = parser.parse_args()

    # 确保 output_path 是绝对路径以便加载模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_path = os.path.join(script_dir, args.output_path)

    evaluate_pretrain_egae(args)
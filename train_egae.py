import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# 导入自定义模块
from models.mgat import MultiViewGAT # 使用你修改后的 mgat 模块
from models.clustering_egae import EGACECLoss # 导入新的EGAE聚类损失模块
from utils.metrics import evaluate_clustering_comprehensive# 导入无标签评估函数

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-View Deep Graph Clustering for Trip Purpose Inference (EGAE variant)')

    # 数据参数
    parser.add_argument('--feature_data_path', type=str, default='./data/od_idv_train_ptype0_gz_1022.xlsx',
                        help='Path to the Excel file containing node features.')
    # 图文件相关参数
    parser.add_argument('--graph_dir', type=str, default='./data',
                        help='Directory where pre-built graph files are stored.')
    parser.add_argument('--graph_prefix', type=str, default='adaptive_undir_knn_graph_gz_ptype0', help='Prefix for the graph file names.')
    parser.add_argument('--knn_k', type=int, default=100,
                        help='K value used for building the KNN graphs.')
    parser.add_argument('--output_path', type=str, default='./results/',
                        help='Directory to save results and models.')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2, help='GAT attention leaky_relu negative slope.')
    parser.add_argument('--n_clusters', type=int, default=6, help='Fixed number of clusters for trip purpose inference.')
    parser.add_argument('--lambda_reg', type=float, default=0.0001, help='Weight for GAT parameter regularization loss.')
    # EGAE聚类损失的权重参数
    parser.add_argument('--alpha_cluster', type=float, default=0.005, help='Weight for EGAE clustering loss (alpha in J = Jr + alpha*Jc).')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=80, help='Number of main training (outer) epochs.')
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretraining epochs.')
    # 评估和指示矩阵更新的间隔 (对应论文的 Outer-iterations 频率)
    parser.add_argument('--update_interval', type=int, default=10, help='Interval (in epochs) to update indicator matrix P and evaluate.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    # 是否标准化特征
    parser.add_argument('--scale_features', action='store_true', default=True, help='Apply StandardScaler to node features.')
    # 内循环的迭代次数 (优化编码器参数)
    parser.add_argument('--inner_iterations', type=int, default=3, help='Number of inner iterations to update encoder within each outer iteration.')

    return parser.parse_args()

def main():
    args = parse_args()
    logging.info(f"Starting training with arguments: {args}")

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        logging.info(f"CUDA available. Using device: {args.device}")
    else:
        logging.info("CUDA not available. Using CPU.")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1. 加载节点特征
    logging.info(f"Loading node features from: {args.feature_data_path}")
    try:
        if args.feature_data_path.endswith('.xlsx'):
            df_features_full = pd.read_excel(args.feature_data_path)
        elif args.feature_data_path.endswith('.csv'):  # 新增对CSV文件的支持
            df_features_full = pd.read_csv(args.feature_data_path)
        else:
            raise ValueError(f"Unsupported feature file format: {args.feature_data_path}. Please use .xlsx or .csv.")

        logging.info(f"Full feature data loaded successfully, shape: {df_features_full.shape}")
        if df_features_full.empty:
            raise ValueError("Feature data is empty!")

        # --- 定义数值特征列名 (你原有的特征) ---
        NUMERICAL_FEATURE_COLUMN_NAMES = [
            'se_block_0', 'se_block_1', 'se_block_2', 'se_block_3', 'se_block_4', 'se_block_5', 'se_block_6',
            'se_block_7',
            'se_block_8', 'se_block_9', 'se_block_10', 'se_block_11', 'se_block_12', 'se_block_13', 'se_block_14',
            'se_block_15',
            'se_block_16', 'se_block_17', 'se_block_18', 'se_block_19', 'se_block_20', 'se_block_21', 'se_block_22',
            'se_block_23',
            'se_block_24', 'se_block_25', 'se_block_26', 'se_block_27', 'se_block_28', 'se_block_29', 'se_block_30',
            'se_block_31',
            'se_block_32', 'se_block_33', 'se_block_34', 'se_block_35', 'se_block_36', 'se_block_37', 'se_block_38',
            'se_block_39',
            'se_block_40', 'se_block_41', 'se_block_42', 'se_block_43', 'se_block_44', 'se_block_45', 'se_block_46',
            'se_block_47',
            'ea_block_0', 'ea_block_1', 'ea_block_2', 'ea_block_3', 'ea_block_4', 'ea_block_5', 'ea_block_6',
            'ea_block_7',
            'ea_block_8', 'ea_block_9', 'ea_block_10', 'ea_block_11', 'ea_block_12', 'ea_block_13', 'ea_block_14',
            'ea_block_15',
            'ea_block_16', 'ea_block_17', 'ea_block_18', 'ea_block_19', 'ea_block_20', 'ea_block_21', 'ea_block_22',
            'ea_block_23',
            'ea_block_24', 'ea_block_25', 'ea_block_26', 'ea_block_27', 'ea_block_28', 'ea_block_29', 'ea_block_30',
            'ea_block_31',
            'ea_block_32', 'ea_block_33', 'ea_block_34', 'ea_block_35', 'ea_block_36', 'ea_block_37', 'ea_block_38',
            'ea_block_39',
            'ea_block_40', 'ea_block_41', 'ea_block_42', 'ea_block_43', 'ea_block_44', 'ea_block_45', 'ea_block_46',
            'ea_block_47',
            '事务所_G_attractiveness', '产业园工厂_G_attractiveness', '体育休闲_G_attractiveness',
            '公司企业_G_attractiveness', '医疗_G_attractiveness',
            '商务楼_G_attractiveness', '固定性住宿_G_attractiveness', '展馆_G_attractiveness', '度假_G_attractiveness',
            '政府机构_G_attractiveness',
            '流动性住宿_G_attractiveness', '社会团体_G_attractiveness', '考试培训_G_attractiveness',
            '购物_G_attractiveness', '酒店会议中心_G_attractiveness',
            '金融保险_G_attractiveness', '风景_G_attractiveness', '餐饮_G_attractiveness', '高校科研_G_attractiveness'
        ]

        # --- 定义新的分类特征列名 ---
        CATEGORICAL_COLUMNS = ['age', 'gender']

        # 检查所有需要的列是否存在
        all_required_columns = NUMERICAL_FEATURE_COLUMN_NAMES + CATEGORICAL_COLUMNS
        missing_in_file = [col for col in all_required_columns if col not in df_features_full.columns]
        if missing_in_file:
            raise ValueError(f"Feature data is missing required columns: {missing_in_file}")

        # 提取数值特征
        numerical_features_df = df_features_full[NUMERICAL_FEATURE_COLUMN_NAMES].copy()

        # 处理和独热编码分类特征
        categorical_features_one_hot_list = []
        for col in CATEGORICAL_COLUMNS:
            # 填充缺失值 (例如用 'Unknown')，确保数据类型为字符串以便 get_dummies
            df_features_full[col] = df_features_full[col].fillna('Unknown').astype(str)
            one_hot_encoded = pd.get_dummies(df_features_full[col], prefix=col,
                                             dummy_na=False)  # dummy_na=False 因为我们已经填充了
            categorical_features_one_hot_list.append(one_hot_encoded)
            logging.info(f"Column '{col}' one-hot encoded into {one_hot_encoded.shape[1]} features.")

        # 合并所有独热编码的特征
        if categorical_features_one_hot_list:
            categorical_features_df = pd.concat(categorical_features_one_hot_list, axis=1)
        else:
            categorical_features_df = pd.DataFrame()  # 如果没有分类特征

        # 标准化数值特征 (如果启用)
        if args.scale_features:
            logging.info("Applying StandardScaler to NUMERICAL features...")
            scaler = StandardScaler()
            numerical_features_scaled_np = scaler.fit_transform(numerical_features_df.values.astype(np.float32))
            numerical_features_final_df = pd.DataFrame(numerical_features_scaled_np,
                                                       columns=NUMERICAL_FEATURE_COLUMN_NAMES,
                                                       index=numerical_features_df.index)
            logging.info("Numerical features scaled.")
        else:
            numerical_features_final_df = numerical_features_df.astype(np.float32)
            logging.info("Skipping feature scaling for numerical features.")

        # 合并数值特征和独热编码的分类特征
        if not categorical_features_df.empty:
            features_final_df = pd.concat([numerical_features_final_df, categorical_features_df], axis=1)
        else:
            features_final_df = numerical_features_final_df

        features_np = features_final_df.values.astype(np.float32)
        features = torch.FloatTensor(features_np).to(device)
        logging.info(f"Node features tensor created, shape: {features.shape}")
        num_nodes = features.shape[0]

        # 保存一份原始的 df_features 用于后续结果保存 (不包含独热编码，但包含原始 age/gender)
        # 或者你也可以选择保存 features_final_df 以包含所有处理过的特征
        df_features_for_results = df_features_full.copy()


    except FileNotFoundError:
        logging.error(f"Error: Feature data file not found at {args.feature_data_path}")
        return
    except ValueError as ve:
        logging.error(f"Error loading or processing feature data: {ve}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during feature loading: {e}")
        import traceback
        traceback.print_exc()  # 打印更详细的错误追踪
        return


    # 2. 加载预构建的图
    logging.info(f"Loading pre-built graphs from directory: {args.graph_dir}")
    adj_list = []
    graph_views = ['time', 'space', 'attr']
    try:
        for view in graph_views:
            graph_file = f"{args.graph_prefix}_{view}_k{args.knn_k}.pt"
            graph_path = os.path.join(args.graph_dir, graph_file)
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

    except FileNotFoundError as fnf:
        logging.error(f"Error: {fnf}")
        logging.error("Please ensure you have run graph_builder.py first to generate the .pt files,")
        logging.error(f"and that --graph_dir ('{args.graph_dir}'), --graph_prefix ('{args.graph_prefix}'), and --knn_k ({args.knn_k}) match.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during graph loading: {e}")
        return


    # 3. 初始化模型组件
    logging.info("Initializing model components...")

    # 多视图注意力编码器 (使用你修改后的 mgat.py 中的类)
    encoder = MultiViewGAT(
        in_features=features.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        num_views=len(adj_list)
    ).to(device)
    encoder.float()

    # EGAE聚类损失模块 (无可学习参数，用于计算 Jc 和更新 P)
    clustering_loss_module = EGACECLoss(
        embedding_dim=args.embedding_dim,
        n_clusters=args.n_clusters,
        device=device
    ).to(device)


    # 优化器 (只优化编码器参数)
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    def compute_reconstruction_loss(reconstructed, adj_list):
        """计算重构损失"""
        total_loss = torch.tensor(0.0, dtype=reconstructed.dtype, device=reconstructed.device)

        # 计算与每个视图的邻接矩阵的重构损失
        for view_data in adj_list:
            # 从边索引和边权重重建邻接矩阵(仅用于计算损失)
            edge_index = view_data['edge_index']
            edge_weight = view_data['edge_weight']
            n = reconstructed.size(0)

            # 计算这些边的重构值
            recon_values = reconstructed[edge_index[0], edge_index[1]]

            # 计算这些边的MSE损失
            edge_loss = F.mse_loss(recon_values, edge_weight)
            total_loss += edge_loss

        return total_loss / len(adj_list)


    # 4. 预训练多视图编码器
    logging.info("Pretraining encoder...")
    # 预训练阶段只使用图重构损失 (采样的) 和参数正则化损失
    pretrain_optimizer = optim.Adam(
         encoder.parameters(),
         lr=args.lr,
         weight_decay=args.weight_decay
     )

    # 提示用户关于内存的潜在问题，尤其是主训练阶段
    logging.warning("The main training loop might still encounter OutOfMemoryError due to the N x N matrix calculation")
    logging.warning("in the clustering loss module (ZZ^T for SVD). This is inherent to the standard EGAE Jc loss formulation.")
    logging.warning("If OOM occurs in the main loop, consider using a GPU with more memory or exploring approximations if available.")


    for epoch in range(args.pretrain_epochs):
        encoder.train()
        start_time = time.time()

        # 前向传播 (编码器返回 embeddings (z), recon (sigmoid), view_weights)
        embeddings, recon_full, _ = encoder(features, adj_list) # recon_full 是 N x N 矩阵

        # 计算重构损失 (使用基于MSE的函数，输入 recon_full 和 adj_list)
        recon_loss = compute_reconstruction_loss(recon_full, adj_list)

        # 计算参数正则化损失
        reg_loss_raw = encoder.compute_regularization_loss()
        reg_loss = reg_loss_raw * args.lambda_reg

        # 总损失 (预训练阶段)
        loss = recon_loss + reg_loss

        # 反向传播
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
        end_time = time.time()
        epoch_time = end_time - start_time

        if (epoch + 1) % 10 == 0 or epoch == args.pretrain_epochs - 1:
            logging.info(f"Pretrain Epoch {epoch + 1}/{args.pretrain_epochs} - Time: {epoch_time:.2f}s: "
                         f"Recon Loss  = {recon_loss.item():.4f}, Reg Loss = {reg_loss.item():.4f}, Total Loss = {loss.item():.4f}")


    # 预训练结束后，使用 K-Means 对预训练嵌入进行聚类并保存结果
    logging.info("\nPretraining finished. Evaluating pretraining results with standard K-Means...")
    encoder.eval()
    with torch.no_grad():
         # 获取预训练结束后的嵌入 (z) 和视图权重
         pretrain_embeddings, _, pretrain_view_weights = encoder(features, adj_list)

    # 使用 **标准 K-Means** 直接对预训练嵌入进行聚类 (在 CPU 上进行)
    pretrain_embeddings_np = pretrain_embeddings.cpu().numpy()
    logging.info(f"Running standard KMeans on pretrain embeddings ({pretrain_embeddings_np.shape})...")
    kmeans_pretrain = KMeans(n_clusters=args.n_clusters, init='k-means++', n_init='auto', random_state=42)
    pretrain_cluster_ids = kmeans_pretrain.fit_predict(pretrain_embeddings_np)
    logging.info("Standard KMeans finished.")

    # 评估预训练聚类结果 (使用无标签评估函数)
    pretrain_comprehensive_results = evaluate_clustering_comprehensive(pretrain_embeddings_np, pretrain_cluster_ids)
    logging.info("Pretraining Standard K-Means Clustering Evaluation:")
    # 打印所有详细指标
    for metric, value in pretrain_comprehensive_results.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric}: {value:.4f}")
        elif isinstance(value, np.ndarray):
             logging.info(f"{metric}: {value.tolist()}")
        else:
            logging.info(f"{metric}: {value}")

    # 保存预训练结果
    pretrain_results_df = df_features_for_results.copy()
    pretrain_results_df['cluster'] = pretrain_cluster_ids
    pretrain_view_weights_np = pretrain_view_weights.cpu().numpy() if pretrain_view_weights is not None else np.full((pretrain_embeddings_np.shape[0], args.num_views if hasattr(args, 'num_views') else 3), np.nan)
    num_views = pretrain_view_weights_np.shape[1]
    for v in range(num_views):
        pretrain_results_df[f'pretrain_weight_view_{v}'] = pretrain_view_weights_np[:, v]

    #预训练结果保存
    pretrain_results_csv_filename = f"n{args.n_clusters}_pretrain_standard_kmeans_clustering_gz_ptype0.csv"
    pretrain_results_csv_path = os.path.join(args.output_path, pretrain_results_csv_filename)
    pretrain_results_df.to_csv(pretrain_results_csv_path, index=False)
    logging.info(f"Pretraining standard KMeans++ results saved to {pretrain_results_csv_path}")

    # 单独保存预训练评估的详细指标到一个文件
    pretrain_eval_metrics_filename = f"n{args.n_clusters}_pretrain_standard_kmeans_evaluation_metrics_gz_ptype0.txt"
    pretrain_eval_metrics_path = os.path.join(args.output_path, pretrain_eval_metrics_filename)
    with open(pretrain_eval_metrics_path, 'w') as f:
        for metric, value in pretrain_comprehensive_results.items():
            f.write(f"{metric}: {value}\n")
    logging.info(f"Pretraining standard KMeans++ evaluation metrics saved to {pretrain_eval_metrics_path}")

    # 保存预训练模型 (只保存 encoder)
    pretrain_encoder_filename = f"n{args.n_clusters}_pretrain_encoder_gz_pytpe0.pth"
    pretrain_encoder_path = os.path.join(args.output_path, pretrain_encoder_filename)
    torch.save({
        'encoder': encoder.state_dict(),
    }, pretrain_encoder_path)
    logging.info(f"Pretraining encoder saved to {pretrain_encoder_path}")


    # 5. 主训练循环 - 深度图聚类 (交替优化)
    logging.info("\nStarting main training loop (Alternating Optimization)...")
    last_inner_loop_loss = float('inf')

    # 初始化聚类指示矩阵 P (在主训练开始前使用预训练的嵌入初始化 P)
    logging.info("Initializing indicator matrix P for main training...")
    encoder.eval() # 切换到评估模式获取嵌入
    with torch.no_grad():
        # 使用预训练结束后的编码器获取初始嵌入 (z)
        initial_embeddings_main, _, _ = encoder(features, adj_list)
        # 使用这些嵌入初始化 EGACECLoss 中的指示矩阵 P
        clustering_loss_module.update_indicator(initial_embeddings_main)

        # --- 打印第一轮主训练的聚类指标结果 ---
        logging.info("Evaluating initial main training clustering results (SVD + KMeans++ on P)...")
        initial_cluster_ids_main, _ = clustering_loss_module.get_cluster_assignments(initial_embeddings_main)
        initial_embeddings_main_np = initial_embeddings_main.cpu().numpy()
        initial_comprehensive_results_main = evaluate_clustering_comprehensive(initial_embeddings_main_np, initial_cluster_ids_main)

        logging.info("Initial Main Training SVD + KMeans++ Clustering Evaluation:")
        for metric, value in initial_comprehensive_results_main.items():
             if isinstance(value, (int, float)):
                 logging.info(f"  {metric}: {value:.4f}")
             elif isinstance(value, np.ndarray):
                 logging.info(f"  {metric}: {value.tolist()}") # 修正打印方式
             else:
                 logging.info(f"  {metric}: {value} ")


    encoder.train() # 切换回训练模式

    for epoch in range(args.epochs):
        start_time = time.time()

        # 内循环：优化编码器参数 (固定指示矩阵 P)
        encoder.train() # 确保模型在训练模式

        for inner_iter in range(args.inner_iterations):
            optimizer.zero_grad()

            # 前向传播 (编码器返回 embeddings (z), recon (sigmoid), view_weights)
            embeddings, recon_full, _ = encoder(features, adj_list) # embeddings 是 Z

            # 计算图重构损失 (使用基于MSE的**采样**函数，输入 embeddings (z) 和 adj_list)
            recon_loss = compute_reconstruction_loss(recon_full, adj_list)

            # 计算聚类损失 (放松的 K-Means 损失 Jc)
            cluster_loss = clustering_loss_module(embeddings) * args.alpha_cluster

            # 计算参数正则化损失
            reg_loss_raw = encoder.compute_regularization_loss()
            reg_loss = reg_loss_raw * args.lambda_reg

            # 总损失 (主训练阶段)
            loss = recon_loss + cluster_loss + reg_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录最后一个内循环的损失
            if inner_iter == args.inner_iterations - 1:
                 last_inner_loop_loss = loss.item()

            # logging.info(f"  Epoch {epoch + 1}/{args.epochs}, Inner Iter {inner_iter + 1}/{args.inner_iterations}: "
            #              f"Recon Loss={recon_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f}, Reg Loss={reg_loss.item():.4f}, Total Loss={loss.item():.4f}")
            # # --- 打印结束 ---

        # 外循环：更新聚类指示矩阵 P 并进行评估 (每隔 update_interval 轮进行)
        # 打印内循环损失和评估结果
        if (epoch + 1) % args.update_interval == 0 or epoch == args.epochs - 1:
             eval_start_time = time.time() # 记录评估开始时间
             encoder.eval() # 评估时切换到 eval 模式
             with torch.no_grad():
                 # 获取当前嵌入 (z) 和视图权重
                 current_embeddings, _, current_view_weights = encoder(features, adj_list)

                 # 更新聚类指示矩阵 P (根据当前嵌入)
                 clustering_loss_module.update_indicator(current_embeddings)

                 # 获取聚类分配结果 (使用 KMeans 在最新的指示矩阵 P 上进行)
                 cluster_ids, _ = clustering_loss_module.get_cluster_assignments(current_embeddings)

             # 评估聚类结果 (使用无标签评价指标)
             embeddings_np = current_embeddings.cpu().numpy()
             cluster_ids_np = cluster_ids  # 确保是 numpy 数组以便传递给 metrics
             comprehensive_results = evaluate_clustering_comprehensive(embeddings_np, cluster_ids_np)


             # 打印外循环的评估指标和最后一个内循环的总损失
             end_time = time.time()
             epoch_time = end_time - start_time
             eval_time = end_time - eval_start_time
             logging.info(f"Epoch {epoch + 1}/{args.epochs} (Eval) - Time: {epoch_time:.2f}s (Eval Time: {eval_time:.2f}s):")
             # 打印所有详细指标
             logging.info("  Evaluation Metrics:")
             for metric, value in comprehensive_results.items():
                 if isinstance(value, (int, float)):
                      logging.info(f"    {metric}: {value:.4f}")
                 elif isinstance(value, np.ndarray):
                      logging.info(f"    {metric}: {value.tolist()}") # 修正打印 numpy 数组的方式
                 else:
                     logging.info(f"    {metric}: {value}")

        else: # 如果不是评估间隔的 epoch，只打印内循环损失
             end_time = time.time()
             epoch_time = end_time - start_time
             logging.info(f"Epoch {epoch + 1}/{args.epochs} - Time: {epoch_time:.2f}s:" f"Recon Loss={recon_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f},Reg Loss={reg_loss.item():.4f},Last Inner Loss={last_inner_loop_loss:.4f}")


    logging.info("\nTraining completed!")

    # 保存最后训练完成的模型 (只保存 encoder)
    logging.info("\nSaving final trained model...")
    final_model_filename = f"n{args.n_clusters}_final_model_gz_ptype0.pth"
    final_model_path = os.path.join(args.output_path, final_model_filename)
    torch.save({
        'encoder': encoder.state_dict(),
        'n_clusters': args.n_clusters,
        'epoch': args.epochs,
    }, final_model_path)
    logging.info(f"Final trained model saved to {final_model_path}")


    # 加载最后训练完成的模型进行最终评估
    logging.info("\nEvaluating final trained model...")
    encoder.eval() # 确保模型处于评估模式

    final_indicator_p_torch = None  # 初始化
    final_view_weights = None

    with torch.no_grad():
        # 获取最后训练完成的嵌入 (z)
        final_embeddings, _, final_view_weights = encoder(features, adj_list)
        # 在最终评估时，根据最后的嵌入更新指示矩阵 P
        clustering_loss_module.update_indicator(final_embeddings)
        # 然后根据最后的指示矩阵 P 获取聚类分配
        cluster_ids_final, final_indicator_p_torch = clustering_loss_module.get_cluster_assignments(final_embeddings)


    # 进行全面评估 (使用无标签评估函数)
    embeddings_np_final = final_embeddings.cpu().numpy()
    cluster_ids_final_np = cluster_ids_final # 确保是 numpy 数组以便传递给 metrics
    view_weights_np_final = final_view_weights.cpu().numpy() if final_view_weights is not None else np.full((embeddings_np_final.shape[0], args.num_views if hasattr(args, 'num_views') else 3), np.nan)
    final_indicator_p_np = final_indicator_p_torch.cpu().numpy()

    comprehensive_results_final = evaluate_clustering_comprehensive(embeddings_np_final, cluster_ids_final_np)
    # 打印详细评估结果
    logging.info("\nDetailed Clustering Evaluation of Final Model:")
    for metric, value in comprehensive_results_final.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric}: {value:.4f}")
        else:
            logging.info(f"{metric}: {value}")


    logging.info("\n--- Detailed Clustering Evaluation of Final Model (based on P indicator matrix as embeddings) ---")
    comprehensive_results_final_on_P = evaluate_clustering_comprehensive(final_indicator_p_np, cluster_ids_final_np)
    for metric, value in comprehensive_results_final_on_P.items():
        if isinstance(value, (int, float)):
            logging.info(f"P_based_{metric}: {value:.4f}")
        elif isinstance(value, (np.ndarray, list)):
            logging.info(f"P_based_{metric}: {value.tolist() if isinstance(value, np.ndarray) else value}")
        else:
            logging.info(f"P_based_{metric}: {value}")

    # 保存最后训练完成模型的聚类结果、嵌入和视图权重
    logging.info("Saving final results with embeddings and view weights...")
    final_results_df = df_features_for_results.copy()
    final_results_df['cluster'] = cluster_ids_final

    # 添加嵌入向量
    for i in range(args.embedding_dim):
        final_results_df[f'embedding_{i}'] = embeddings_np_final[:, i]

    # 添加视图权重
    num_views = view_weights_np_final.shape[1]
    for v in range(num_views):
        final_results_df[f'weight_view_{v}'] = view_weights_np_final[:, v]

    #保存最终结果
    final_results_csv_filename = f"n{args.n_clusters}_final_clustering_gz_pytpe0.csv"
    final_results_csv_path = os.path.join(args.output_path, final_results_csv_filename)
    final_results_df.to_csv(final_results_csv_path, index=False)
    logging.info(f"Final results saved to {final_results_csv_path}")

    # 单独保存最终评估的详细指标到一个文件
    final_eval_metrics_filename = f"n{args.n_clusters}_final_evaluation_metrics_gz_ptype0.txt"
    final_eval_metrics_path = os.path.join(args.output_path, final_eval_metrics_filename)
    with open(final_eval_metrics_path, 'w') as f:
         for metric, value in comprehensive_results_final.items():
              f.write(f"{metric}: {value}\n")
    logging.info(f"Final evaluation metrics saved to {final_eval_metrics_path}")


if __name__ == "__main__":
    main()
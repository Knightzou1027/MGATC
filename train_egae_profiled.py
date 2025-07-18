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
import torch.profiler  # 导入 Profiler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# 导入自定义模块
from models.mgat import MultiViewGAT
from models.clustering_egae import EGACECLoss
from utils.metrics import evaluate_clustering_comprehensive


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profiler Version: Multi-View Deep Graph Clustering for Trip Purpose Inference (EGAE variant)')

    # 数据参数
    parser.add_argument('--feature_data_path', type=str, default='./data/od_idv_train_gz_1022.xlsx',
                        help='Path to the Excel file containing node features.')
    parser.add_argument('--graph_dir', type=str, default='./data',
                        help='Directory where pre-built graph files are stored.')
    parser.add_argument('--graph_prefix', type=str, default='adaptive_undir_knn_graph_gz',
                        help='Prefix for the graph file names.')
    parser.add_argument('--knn_k', type=int, default=100,
                        help='K value used for building the KNN graphs.')
    parser.add_argument('--output_path', type=str, default='./results_profiled/',  # 修改输出路径以区分
                        help='Directory to save results, models, and profiler traces.')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2, help='GAT attention leaky_relu negative slope.')
    parser.add_argument('--n_clusters', type=int, default=12,
                        help='Fixed number of clusters for trip purpose inference.')
    parser.add_argument('--lambda_reg', type=float, default=0.0001,
                        help='Weight for GAT parameter regularization loss.')
    parser.add_argument('--alpha_cluster', type=float, default=0.01,
                        help='Weight for EGAE clustering loss (alpha in J = Jr + alpha*Jc).')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100, help='Number of main training (outer) epochs.')
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretraining epochs.')
    parser.add_argument('--update_interval', type=int, default=10,
                        help='Interval (in epochs) to update indicator matrix P and evaluate.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--scale_features', action='store_true', default=True,
                        help='Apply StandardScaler to node features.')
    parser.add_argument('--inner_iterations', type=int, default=3,
                        help='Number of inner iterations to update encoder within each outer iteration.')

    # Profiler specific arguments (optional, could be hardcoded for a profiling script)
    parser.add_argument('--profile_pretrain_epochs', type=int, default=4,
                        help="Number of pretrain epochs to run under profiler's active phase (needs enough total pretrain_epochs).")
    parser.add_argument('--profile_main_epochs', type=int, default=12,
                        help="Number of main training epochs to run for profiling analysis.")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"Starting PROFILING training with arguments: {args}")

    # --- Profiler setup ---
    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler_trace_dir = os.path.join(args.output_path, 'profiler_traces')
    os.makedirs(profiler_trace_dir, exist_ok=True)
    logging.info(f"Profiler traces will be saved to: {profiler_trace_dir}")

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
        elif args.feature_data_path.endswith('.csv'):
            df_features_full = pd.read_csv(args.feature_data_path)
        else:
            raise ValueError(f"Unsupported feature file format: {args.feature_data_path}. Please use .xlsx or .csv.")

        logging.info(f"Full feature data loaded successfully, shape: {df_features_full.shape}")
        if df_features_full.empty:
            raise ValueError("Feature data is empty!")

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
        CATEGORICAL_COLUMNS = ['age', 'gender']
        all_required_columns = NUMERICAL_FEATURE_COLUMN_NAMES + CATEGORICAL_COLUMNS
        missing_in_file = [col for col in all_required_columns if col not in df_features_full.columns]
        if missing_in_file:
            raise ValueError(f"Feature data is missing required columns: {missing_in_file}")

        numerical_features_df = df_features_full[NUMERICAL_FEATURE_COLUMN_NAMES].copy()
        categorical_features_one_hot_list = []
        for col in CATEGORICAL_COLUMNS:
            df_features_full[col] = df_features_full[col].fillna('Unknown').astype(str)
            one_hot_encoded = pd.get_dummies(df_features_full[col], prefix=col, dummy_na=False)
            categorical_features_one_hot_list.append(one_hot_encoded)
            logging.info(f"Column '{col}' one-hot encoded into {one_hot_encoded.shape[1]} features.")

        if categorical_features_one_hot_list:
            categorical_features_df = pd.concat(categorical_features_one_hot_list, axis=1)
        else:
            categorical_features_df = pd.DataFrame()

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

        if not categorical_features_df.empty:
            features_final_df = pd.concat([numerical_features_final_df, categorical_features_df], axis=1)
        else:
            features_final_df = numerical_features_final_df

        features_np = features_final_df.values.astype(np.float32)
        features = torch.FloatTensor(features_np).to(device)
        logging.info(f"Node features tensor created, shape: {features.shape}")
        num_nodes = features.shape[0]
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
        traceback.print_exc()
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
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during graph loading: {e}")
        return

    # 3. 初始化模型组件
    logging.info("Initializing model components...")
    encoder = MultiViewGAT(
        in_features=features.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        num_views=len(adj_list)
    ).to(device)
    encoder.float()
    clustering_loss_module = EGACECLoss(
        embedding_dim=args.embedding_dim,
        n_clusters=args.n_clusters,
        device=device
    ).to(device)
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    def compute_reconstruction_loss(reconstructed, adj_list):
        total_loss = torch.tensor(0.0, dtype=reconstructed.dtype, device=reconstructed.device)
        for view_data in adj_list:
            edge_index = view_data['edge_index']
            edge_weight = view_data['edge_weight']
            recon_values = reconstructed[edge_index[0], edge_index[1]]
            edge_loss = F.mse_loss(recon_values, edge_weight)
            total_loss += edge_loss
        return total_loss / len(adj_list)

    # 4. 预训练多视图编码器
    logging.info("Pretraining encoder...")
    pretrain_optimizer = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logging.warning("Note: Pretraining uses sampled reconstruction loss to save memory.")
    logging.warning("The main training loop might still encounter OutOfMemoryError due to the N x N matrix calculation")
    logging.warning(
        "in the clustering loss module (ZZ^T for SVD). This is inherent to the standard EGAE Jc loss formulation.")
    logging.warning(
        "If OOM occurs in the main loop, consider using a GPU with more memory or exploring approximations if available.")

    # --- Profile a few epochs of Pre-training ---
    num_pretrain_profile_total_epochs = args.profile_pretrain_epochs
    if args.pretrain_epochs < num_pretrain_profile_total_epochs:
        logging.warning(
            f"Total pretrain_epochs ({args.pretrain_epochs}) is less than num_pretrain_profile_total_epochs ({num_pretrain_profile_total_epochs}). Profiling all pretrain_epochs.")
        num_pretrain_profile_total_epochs = args.pretrain_epochs

    pretrain_schedule_obj = None  # Initialize
    active_steps_pretrain = 0

    if num_pretrain_profile_total_epochs > 0:
        # Calculate schedule steps
        wait_steps_pretrain = 1 if num_pretrain_profile_total_epochs > 1 else 0
        # Ensure active_steps is at least 1 if there's any profiling to be done after wait
        if num_pretrain_profile_total_epochs - wait_steps_pretrain > 0:
            warmup_steps_pretrain = 1 if num_pretrain_profile_total_epochs - wait_steps_pretrain > 1 else 0
        else:  # Not enough epochs for wait, let alone warmup/active
            warmup_steps_pretrain = 0

        active_steps_pretrain = num_pretrain_profile_total_epochs - wait_steps_pretrain - warmup_steps_pretrain
        if active_steps_pretrain <= 0 and num_pretrain_profile_total_epochs > 0:  # if total is 1, wait=0, warmup=0, active=1
            active_steps_pretrain = num_pretrain_profile_total_epochs  # if only 1 or 2 epochs, make them all active after minimal wait/warmup
            warmup_steps_pretrain = 0
            wait_steps_pretrain = 0
            if num_pretrain_profile_total_epochs == 2:  # e.g. wait 0, warmup 1, active 1
                warmup_steps_pretrain = 1
                active_steps_pretrain = 1

        pretrain_schedule_obj = torch.profiler.schedule(
            wait=wait_steps_pretrain,
            warmup=warmup_steps_pretrain,
            active=active_steps_pretrain,
            repeat=1
        )
        logging.info(f"Type of pretrain_schedule_obj: {type(pretrain_schedule_obj)}")
        if hasattr(pretrain_schedule_obj, 'wait_steps'):
            logging.info(
                f"Profiling pretraining. Schedule: wait={pretrain_schedule_obj.wait_steps}, "
                f"warmup={pretrain_schedule_obj.warmup_steps}, "
                f"active={pretrain_schedule_obj.active_steps}"
            )
        else:  # Should not happen if torch.profiler.schedule is called correctly
            logging.error(f"pretrain_schedule_obj is not a valid schedule object. Type: {type(pretrain_schedule_obj)}")
            pretrain_schedule_obj = None
    else:
        logging.info("Skipping pretraining profiling as num_pretrain_profile_total_epochs is 0.")

    profiler_context_pretrain = None
    if num_pretrain_profile_total_epochs > 0 and pretrain_schedule_obj is not None and active_steps_pretrain > 0:
        profiler_context_pretrain = torch.profiler.profile(
            activities=profiler_activities,
            schedule=pretrain_schedule_obj,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(profiler_trace_dir, 'pretrain')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler_context_pretrain.__enter__()  # Manually enter context

    for epoch in range(args.pretrain_epochs):
        # For profiling run, we might want to break early after profiled epochs
        if profiler_context_pretrain is None and epoch >= num_pretrain_profile_total_epochs and num_pretrain_profile_total_epochs < args.pretrain_epochs:
            logging.info(
                f"Stopping pretraining profiling run after {num_pretrain_profile_total_epochs} epochs (profiler was not active).")
            break
            # If profiler was active, its schedule will handle when to stop collecting data.
        # We break here if this is a dedicated profiling run and we don't want to continue unprofiled.
        if profiler_context_pretrain is not None and epoch >= (
                wait_steps_pretrain + warmup_steps_pretrain + active_steps_pretrain):
            if num_pretrain_profile_total_epochs < args.pretrain_epochs:  # Only break if it's a shorter profiling run
                logging.info(f"Finished profiling pretrain active steps. Stopping pretrain at epoch {epoch}.")
                break

        encoder.train()
        start_time_epoch = time.time()
        embeddings, recon_full, _ = encoder(features, adj_list)
        recon_loss = compute_reconstruction_loss(recon_full, adj_list)
        reg_loss_raw = encoder.compute_regularization_loss()
        reg_loss = reg_loss_raw * args.lambda_reg
        loss = recon_loss + reg_loss
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
        epoch_time_epoch = time.time() - start_time_epoch

        if (epoch + 1) % 10 == 0 or epoch == args.pretrain_epochs - 1 or epoch < num_pretrain_profile_total_epochs:
            logging.info(
                f"Pretrain Epoch {epoch + 1}/{args.pretrain_epochs} (Profiled active up to {active_steps_pretrain} steps) - Time: {epoch_time_epoch:.2f}s: "
                f"Recon Loss  = {recon_loss.item():.4f}, Reg Loss = {reg_loss.item():.4f}, Total Loss = {loss.item():.4f}")

        if profiler_context_pretrain is not None:
            profiler_context_pretrain.step()  # Signal profiler step

    if profiler_context_pretrain is not None:
        profiler_context_pretrain.__exit__(None, None, None)  # Manually exit context
        logging.info("Pre-training profiling with schedule completed.")
        # Tensorboard trace is saved by on_trace_ready. Additional manual export/print can be done:
        # print(profiler_context_pretrain.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        # profiler_context_pretrain.export_chrome_trace(os.path.join(profiler_trace_dir, "pretrain_manual_trace.json"))

    # --- 预训练结束后的评估和模型保存 ---
    # 在分析脚本中，这些部分可以根据需要保留或注释掉，以节省时间
    logging.info(
        "\nPretraining finished (or profiled segment finished). Evaluating pretraining results with standard K-Means...")
    encoder.eval()
    with torch.no_grad():
        pretrain_embeddings, _, pretrain_view_weights = encoder(features, adj_list)
    pretrain_embeddings_np = pretrain_embeddings.cpu().numpy()
    kmeans_pretrain = KMeans(n_clusters=args.n_clusters, init='k-means++', n_init='auto', random_state=42)
    pretrain_cluster_ids = kmeans_pretrain.fit_predict(pretrain_embeddings_np)
    pretrain_comprehensive_results = evaluate_clustering_comprehensive(pretrain_embeddings_np, pretrain_cluster_ids)
    logging.info("Pretraining Standard K-Means Clustering Evaluation:")
    for metric, value in pretrain_comprehensive_results.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric}: {value:.4f}")
        else:
            logging.info(f"{metric}: {value}")
    # ... (此处省略了预训练结果和模型的保存代码，与原脚本一致)

    # --- Profile Initial P Calculation for Main Training ---
    logging.info("\nProfiling Initial P Calculation for main training...")
    profiler_context_init_p = torch.profiler.profile(
        activities=profiler_activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(profiler_trace_dir, 'init_P'))
    )
    profiler_context_init_p.__enter__()
    with torch.profiler.record_function("Initial_P_Calculation_Block"):
        encoder.eval()
        with torch.no_grad():
            initial_embeddings_main, _, _ = encoder(features, adj_list)
            with torch.profiler.record_function("P_Update_SVD_Initial"):
                clustering_loss_module.update_indicator(initial_embeddings_main)
            with torch.profiler.record_function("KMeans_on_P_Initial"):
                initial_cluster_ids_main, _ = clustering_loss_module.get_cluster_assignments(initial_embeddings_main)
        initial_embeddings_main_np = initial_embeddings_main.cpu().numpy()
        initial_comprehensive_results_main = evaluate_clustering_comprehensive(initial_embeddings_main_np,
                                                                               initial_cluster_ids_main)
        logging.info("Initial Main Training SVD + KMeans++ Clustering Evaluation:")
        for metric, value in initial_comprehensive_results_main.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {metric}: {value:.4f}")
            elif isinstance(value, np.ndarray):
                logging.info(f"  {metric}: {value.tolist()}")
            else:
                logging.info(f"  {metric}: {value} ")
    profiler_context_init_p.__exit__(None, None, None)
    logging.info("Initial P calculation profiling completed.")
    encoder.train()

    # --- Profile Main Training Loop ---
    profiled_main_epochs = args.profile_main_epochs
    logging.info(
        f"\nStarting main training loop (Alternating Optimization) - Profiling up to {profiled_main_epochs} epochs...")

    if profiled_main_epochs > 0:
        profiler_context_main = torch.profiler.profile(
            activities=profiler_activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(profiler_trace_dir, 'main_train')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler_context_main.__enter__()

        for epoch in range(args.epochs):
            if epoch >= profiled_main_epochs:
                logging.info(f"Stopping main training profiling run after {profiled_main_epochs} epochs.")
                break

            with torch.profiler.record_function(f"Epoch_{epoch + 1}_Full_Main"):
                start_time_epoch_main = time.time()
                encoder.train()
                last_inner_loop_loss = float('inf')  # Reset for current epoch

                with torch.profiler.record_function("Inner_Loop_Optimization_Main"):
                    for inner_iter in range(args.inner_iterations):
                        optimizer.zero_grad()
                        embeddings, recon_full, _ = encoder(features, adj_list)  # Corrected variable name
                        recon_loss = compute_reconstruction_loss(recon_full, adj_list)
                        with torch.profiler.record_function("Clustering_Loss_Main"):
                            cluster_loss_val = clustering_loss_module(embeddings)  # Renamed to avoid conflict
                            cluster_loss = cluster_loss_val * args.alpha_cluster
                        reg_loss_raw = encoder.compute_regularization_loss()
                        reg_loss = reg_loss_raw * args.lambda_reg
                        loss = recon_loss + cluster_loss + reg_loss
                        loss.backward()
                        optimizer.step()
                        if inner_iter == args.inner_iterations - 1:
                            last_inner_loop_loss = loss.item()

                current_eval_time = 0.0  # Initialize
                is_eval_epoch = (
                                            epoch + 1) % args.update_interval == 0 or epoch == args.epochs - 1 or epoch == profiled_main_epochs - 1

                if is_eval_epoch:
                    with torch.profiler.record_function("Outer_Loop_Evaluation_Main"):
                        eval_start_time_section = time.time()
                        encoder.eval()
                        with torch.no_grad():
                            current_embeddings, _, current_view_weights = encoder(features, adj_list)
                            with torch.profiler.record_function("P_Update_SVD_Main"):
                                clustering_loss_module.update_indicator(current_embeddings)
                            with torch.profiler.record_function("KMeans_on_P_Main"):
                                cluster_ids, _ = clustering_loss_module.get_cluster_assignments(current_embeddings)
                        embeddings_np = current_embeddings.cpu().numpy()
                        cluster_ids_np = cluster_ids
                        comprehensive_results = evaluate_clustering_comprehensive(embeddings_np, cluster_ids_np)
                        current_eval_time = time.time() - eval_start_time_section
                        logging.info(f"  Evaluation Metrics for Epoch {epoch + 1}:")
                        for metric, value in comprehensive_results.items():
                            if isinstance(value, (int, float)):
                                logging.info(f"    {metric}: {value:.4f}")
                            elif isinstance(value, np.ndarray):
                                logging.info(f"    {metric}: {value.tolist()}")
                            else:
                                logging.info(f"    {metric}: {value}")
                    encoder.train()

                epoch_time_main = time.time() - start_time_epoch_main
                if is_eval_epoch:
                    logging.info(
                        f"Profiled Main Epoch {epoch + 1}/{args.epochs} (Eval) - Time: {epoch_time_main:.2f}s (Eval Section Time: {current_eval_time:.2f}s): Last Inner Loss={last_inner_loop_loss:.4f}")
                else:
                    logging.info(
                        f"Profiled Main Epoch {epoch + 1}/{args.epochs} - Time: {epoch_time_main:.2f}s: Recon Loss={recon_loss.item():.4f}, Cluster Loss={cluster_loss.item():.4f},Reg Loss={reg_loss.item():.4f},Last Inner Loss={last_inner_loop_loss:.4f}")

            if profiler_context_main is not None:  # Step only if profiler is active
                profiler_context_main.step()  # This is for schedule, but here we are manually breaking

        if profiler_context_main is not None:
            profiler_context_main.__exit__(None, None, None)
            logging.info("Main training profiling with schedule completed.")
    else:
        logging.info("Skipping main training profiling as profiled_main_epochs is 0.")

    logging.info(f"\n--- PROFILING RUN COMPLETED (or reached profiled epoch limit) ---")
    logging.info(f"--- Traces (if generated by on_trace_ready) are in {profiler_trace_dir} ---")
    logging.info(
        "--- For a full training run, use the original script or remove/adjust profiling limits and early exits ---")

    # For a profiling script, usually we don't run the full saving/final eval unless also profiling them.
    # Adding return here to make it clear this is a profiling-focused run.
    return


if __name__ == "__main__":
    main()
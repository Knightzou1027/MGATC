import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler #用于特征标准化

# 导入自定义模块
from models.mgat import MultiViewGAT
from models.clustering import ClusteringModule, target_distribution  # 直接导入
#from utils.graph_builder import build_multi_view_graphs
#from utils.data_processor import load_and_preprocess_data
from utils.metrics import evaluate_clustering, evaluate_clustering_comprehensive, compute_clustering_quality  # 导入无标签评估函数

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-View Deep Graph Clustering for Trip Purpose Inference')

    # 数据参数
    parser.add_argument('--feature_data_path', type=str, default='./data/od_idv_train_sz_1022.xlsx',
                        help='Path to the Excel file containing node features.')
    # !! 新增: 图文件相关参数 !!
    parser.add_argument('--graph_dir', type=str, default='./data',
                        help='Directory where pre-built graph files are stored.')
    parser.add_argument('--graph_prefix', type=str, default='adaptive_undir_knn_graph', help='Prefix for the graph file names.')
    parser.add_argument('--knn_k', type=int, default=100,
                        help='K value used for building the KNN graphs.')
    parser.add_argument('--output_path', type=str, default='./results/',
                        help='Directory to save results and models.')

    # 图构建参数
    #parser.add_argument('--time_threshold', type=float, default=0.8)
    #parser.add_argument('--space_threshold', type=float, default=2.0)
    #parser.add_argument('--attr_threshold', type=float, default=0.7)

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--n_clusters', type=int, default=6)  # 固定为6类出行目的
    parser.add_argument('--lambda_reg', type=float, default=0.001, help='Weight for GAT parameter regularization loss.')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)  # 训练轮数
    parser.add_argument('--pretrain_epochs', type=int, default=100)  # 预训练轮数，先基于重构和正则化损失来获取嵌入进行聚类获得初始聚类结果
    parser.add_argument('--update_interval', type=int, default=20)  # 每5个epoch更新一次目标分布
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    # 是否标准化特征
    parser.add_argument('--scale_features', action='store_true', help='Apply StandardScaler to node features.')

    return parser.parse_args()

def main():
    # 1. 解析参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 加载节点特征
    print(f"Loading node features from: {args.feature_data_path}")
    try:
        if args.feature_data_path.endswith('.xlsx'):
            df_features = pd.read_excel(args.feature_data_path)
        else:
             raise ValueError(f"Unsupported feature file format: {args.feature_data_path}")

        print(f"Feature data loaded successfully, shape: {df_features.shape}")
        if df_features.empty:
            raise ValueError("Feature data is empty!")

        # !! ************************************************* !!
        # !! ** 在这里定义你的 115 个特征列的列名列表 ** !!
        # !! ************************************************* !!
        FEATURE_COLUMN_NAMES = [
            # --- 96 个时间块特征列名 ---
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

        # 检查特征列是否存在
        missing_features = [col for col in FEATURE_COLUMN_NAMES if col not in df_features.columns]
        if missing_features:
            raise ValueError(f"Feature data is missing required columns: {missing_features}")

        # 提取特征数据
        features_np = df_features[FEATURE_COLUMN_NAMES].values.astype(np.float32)

        # (可选但推荐) 特征标准化
        if args.scale_features:
            print("Applying StandardScaler to features...")
            scaler = StandardScaler()
            features_np = scaler.fit_transform(features_np)
            print("Features scaled.")

        # 转换为 PyTorch 张量
        features = torch.FloatTensor(features_np).to(device)
        print(f"Node features tensor created, shape: {features.shape}")

    except FileNotFoundError:
        print(f"Error: Feature data file not found at {args.feature_data_path}")
        return
    except ValueError as ve:
        print(f"Error loading or processing feature data: {ve}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during feature loading: {e}")
        return

    # 3. 加载预构建的图
    print(f"Loading pre-built graphs from directory: {args.graph_dir}")
    adj_list = []
    graph_views = ['time', 'space', 'attr']
    try:
        for view in graph_views:
            graph_file = f"{args.graph_prefix}_{view}_k{args.knn_k}.pt"
            graph_path = os.path.join(args.graph_dir, graph_file)
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            graph_data = torch.load(graph_path, map_location=device) # 直接加载到目标设备
            graph_data['edge_weight'] = graph_data['edge_weight'].to(device).to(torch.float32)
            # 确保加载的数据包含 edge_index 和 edge_weight
            if 'edge_index' not in graph_data or 'edge_weight' not in graph_data:
                 raise ValueError(f"Graph file {graph_path} is missing 'edge_index' or 'edge_weight'.")

            # 确保 edge_index 和 edge_weight 在正确的设备上 (虽然 map_location 应该已经处理了)
            graph_data['edge_index'] = graph_data['edge_index'].to(device)
            graph_data['edge_weight'] = graph_data['edge_weight'].to(device)

            adj_list.append(graph_data)
            print(f"  Loaded {view} graph: {graph_data['edge_index'].shape[1]} edges.")

    except FileNotFoundError as fnf:
        print(f"Error: {fnf}")
        print("Please ensure you have run graph_builder.py first to generate the .pt files,")
        print(f"and that --graph_dir ('{args.graph_dir}'), --graph_prefix ('{args.graph_prefix}'), and --knn_k ({args.knn_k}) match.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during graph loading: {e}")
        return

    # 4. 初始化模型组件
    print("Initializing model components...")

    # 多视图注意力编码器
    encoder = MultiViewGAT(
        in_features=features.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        num_views=len(adj_list)
    ).to(device)

    encoder.float()  # 确保编码器参数为 float 类型

    # 聚类模块 - 固定为6类出行目的
    clustering = ClusteringModule(
        embedding_dim=args.embedding_dim,
        n_clusters=args.n_clusters
    ).to(device)

    clustering.float()  # 确保聚类模块参数为 float 类型

    # 5. 定义优化器
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    optimizer_clustering = torch.optim.Adam(
        clustering.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 6. 预训练多视图编码器
    print("Pretraining encoder...")
    for epoch in range(args.pretrain_epochs):
        encoder.train()

        # 前向传播
        embeddings, recon, _ = encoder(features, adj_list)

        # 计算重构损失
        recon_loss = compute_reconstruction_loss(recon, adj_list)

        # 计算参数正则化损失
        reg_loss_raw = encoder.compute_regularization_loss()
        reg_loss = reg_loss_raw * args.lambda_reg

        # 总损失
        loss = recon_loss + reg_loss

        # 反向传播
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        if (epoch + 1) % 10 == 0:
            print(f"Pretrain Epoch {epoch + 1}/{args.pretrain_epochs}: "
                  f"Recon Loss = {recon_loss.item():.4f}, Reg Loss = {reg_loss.item():.4f}")

    # 7. 初始化聚类中心
    print("Initializing clustering centers...")
    with torch.no_grad():
        embeddings, _, view_weights = encoder(features, adj_list)

    # 使用K-means初始化聚类中心
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    cluster_ids = kmeans.fit_predict(embeddings.cpu().numpy())
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)

    pretrain_results = df_features.copy()
    pretrain_results['cluster'] = cluster_ids
    view_weights_np = view_weights.cpu().numpy()
    for v in range(view_weights_np.shape[1]):
        pretrain_results[f'pretain_weight_view_{v}'] = view_weights_np[:, v]

    # 保存预训练结果
    pretrain_results.to_csv(os.path.join(args.output_path, 'pretrain_results.csv'), index=False)

    clustering.init_cluster_centers(cluster_centers)

    # 8. 主训练循环 - 深度图聚类
    print("Starting main training loop...")
    best_silhouette = -1  # 轮廓系数范围为[-1,1]，越大越好
    best_quality = -float('inf')  # 综合质量分数

    # 初始化目标分布
    with torch.no_grad():
        q = clustering(embeddings)
        p = target_distribution(q)

    for epoch in range(args.epochs):
        encoder.train()
        clustering.train()

        # 前向传播
        embeddings, recon, _ = encoder(features, adj_list)
        q = clustering(embeddings)

        # 计算损失
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        recon_loss = compute_reconstruction_loss(recon, adj_list)
        reg_loss = encoder.compute_regularization_loss()

        # 总损失
        loss = kl_loss + recon_loss + reg_loss

        # 反向传播
        optimizer_encoder.zero_grad()
        optimizer_clustering.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_clustering.step()

        # 每隔指定轮次更新目标分布，论文里提到的是5次，但需要根据情况来进行调整
        if (epoch + 1) % args.update_interval == 0:
            with torch.no_grad():
                q = clustering(embeddings)
                p = target_distribution(q)

        # 评估聚类结果
        cluster_ids = torch.argmax(q, dim=1).cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()

        # 使用无标签评价指标
        metrics = evaluate_clustering(embeddings_np, cluster_ids)
        silhouette = metrics['silhouette']
        davies_bouldin = metrics['davies_bouldin']
        calinski_harabasz = metrics['calinski_harabasz']

        # 计算综合质量得分
        quality_score = compute_clustering_quality(embeddings_np, cluster_ids)

        # 如果当前轮次效果最好，则保存结果
        if quality_score > best_quality:
            best_quality = quality_score
            best_silhouette = silhouette
            best_db = davies_bouldin
            best_ch = calinski_harabasz

            # 保存最佳模型
            torch.save({
                'encoder': encoder.state_dict(),
                'clustering': clustering.state_dict(),
                'n_clusters': args.n_clusters
            }, os.path.join(args.output_path, 'best_model.pth'))

            # 保存聚类结果
            np.save(os.path.join(args.output_path, 'cluster_assignments.npy'), cluster_ids)

        # 打印结果
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}: "
                  f"Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}, CH={calinski_harabasz:.2f}, "
                  f"Quality={quality_score:.4f}, Loss={loss.item():.4f}")

    print("\nTraining completed!")
    print(f"Best result: Silhouette={best_silhouette:.4f}, DB={best_db:.4f}, CH={best_ch:.2f}")
    print(f"Quality Score={best_quality:.4f}, n_clusters={args.n_clusters}")

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(args.output_path, 'best_model.pth'))
    encoder.load_state_dict(checkpoint['encoder'])
    clustering.load_state_dict(checkpoint['clustering'])

    encoder.eval()
    clustering.eval()

    final_view_weights = None #初始化变量
    with torch.no_grad():
        final_embeddings, _, final_view_weights = encoder(features, adj_list)
        q = clustering(embeddings)
        cluster_ids = torch.argmax(q, dim=1).cpu().numpy()

    # 进行全面评估
    embeddings_np = final_embeddings.cpu().numpy()
    # ** 将 view_weights 转为 numpy 数组 **
    if final_view_weights is not None:
        view_weights_np = final_view_weights.cpu().numpy()
    else:
        # 如果出于某种原因没有得到 view_weights，创建一个占位符或发出警告
        print("Warning: Could not retrieve final view weights.")
        view_weights_np = np.full((embeddings_np.shape[0], args.num_views if hasattr(args, 'num_views') else 3),
                                  np.nan)  # 假设 num_views=3
    comprehensive_results = evaluate_clustering_comprehensive(embeddings_np, cluster_ids)

    # 打印详细评估结果
    print("\nDetailed Clustering Evaluation:")
    for metric, value in comprehensive_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # 保存每个样本的嵌入和聚类结果
    print("Saving final results with embeddings and view weights...")
    results_df = df_features.copy()
    results_df['cluster'] = cluster_ids

    # 添加嵌入向量
    for i in range(args.embedding_dim):
        results_df[f'embedding_{i}'] = embeddings_np[:, i]

    # 添加视图权重
    num_views = view_weights_np.shape[1]  # 获取视图数量 (应该是 3)
    for v in range(num_views):
        # 为每个视图创建一个列，例如 'weight_view_0', 'weight_view_1', 'weight_view_2'
        results_df[f'weight_view_{v}'] = view_weights_np[:, v]

    # 保存结果
    results_df.to_csv(os.path.join(args.output_path, 'clustering_results.csv'), index=False)

    print(f"Results saved to {args.output_path}")


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


if __name__ == "__main__":
    main()

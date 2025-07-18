import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# --- 配置区域 ---
DATA_FILE_PATH = 'data/od_idv_train_sz_1022.xlsx'
# !! 输出图表保存目录 !!
OUTPUT_VIS_DIR = './feature_analysis_plots'

# !! 特征列名列表 (96 时间块 + 19 吸引力) !!
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
# -------------------------------------------------

# 设置 Matplotlib/Seaborn 样式 (可选)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者其他支持中文的字体如 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 创建输出目录
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
print(f"可视化图表将保存到: {OUTPUT_VIS_DIR}")

# 1. 加载数据
try:
    print(f"加载数据从: {DATA_FILE_PATH}")
    if DATA_FILE_PATH.endswith('.xlsx'):
        df = pd.read_excel(DATA_FILE_PATH)
    else:
        raise ValueError("仅支持 .xlsx 文件")
    print(f"数据加载成功，原始形状: {df.shape}")
    if df.empty: raise ValueError("数据为空")

    # 检查特征列是否存在
    missing_cols = [col for col in FEATURE_COLUMN_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少以下特征列: {missing_cols}")

    # 提取特征子集
    features_df = df[FEATURE_COLUMN_NAMES].copy()

except FileNotFoundError:
    print(f"错误: 文件未找到 {DATA_FILE_PATH}")
    exit()
except ValueError as ve:
    print(f"错误: {ve}")
    exit()
except Exception as e:
    print(f"加载或检查数据时发生意外错误: {e}")
    exit()

# 2. 描述性统计
print("\n--- 描述性统计 ---")
# 为了更好地显示，可以转置
desc_stats = features_df.describe().T
print(desc_stats)
desc_stats.to_csv(os.path.join(OUTPUT_VIS_DIR, 'feature_descriptive_stats.csv'))
print("描述性统计已保存到 feature_descriptive_stats.csv")

# 3. 缺失值检查
print("\n--- 缺失值检查 ---")
missing_values = features_df.isnull().sum()
missing_cols_with_na = missing_values[missing_values > 0]
if not missing_cols_with_na.empty:
    print("以下特征列存在缺失值:")
    print(missing_cols_with_na)
    # 你可能需要在这里添加处理缺失值的逻辑 (例如填充或删除)
    # features_df.fillna(features_df.mean(), inplace=True) # 示例：用均值填充
else:
    print("所有 115 个特征列均无缺失值。")

# 4. 分布可视化 (分批绘制)
print("\n--- 特征分布可视化 ---")

# --- 图 1: 时间块特征 ---
time_block_cols = [f'se_block_{i}' for i in [6, 12, 18, 24, 30, 36, 42]] + \
                  [f'ea_block_{i}' for i in [6, 12, 18, 24, 30, 36, 42]]
time_block_cols = [col for col in time_block_cols if col in features_df.columns] # 确保存在

if time_block_cols:
    print(f"  绘制时间块特征分布图 ({len(time_block_cols)} 个特征)...")
    num_plots = len(time_block_cols)
    num_cols_grid = 4
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    plt.figure(figsize=(num_cols_grid * 4, num_rows_grid * 3))
    for i, col in enumerate(time_block_cols):
        plt.subplot(num_rows_grid, num_cols_grid, i + 1)
        sns.histplot(features_df[col], kde=True, bins=30)
        plt.title(f'{col} 分布', fontsize=10)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    plt.suptitle('时间块特征分布', fontsize=14, y=1.02) # 添加总标题
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局防止标题重叠
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'feature_dist_time_blocks.png'))
    print(f"  时间块特征分布图已保存到 {os.path.join(OUTPUT_VIS_DIR, 'feature_dist_time_blocks.png')}")
    plt.close()
else:
    print("  未找到用于绘制的时间块特征列。")

# --- 图 2: 吸引力特征 ---
attractiveness_cols = [
    '事务所_G_attractiveness', '产业园工厂_G_attractiveness','体育休闲_G_attractiveness','公司企业_G_attractiveness','医疗_G_attractiveness',
    '商务楼_G_attractiveness','固定性住宿_G_attractiveness','展馆_G_attractiveness','度假_G_attractiveness','政府机构_G_attractiveness',
    '流动性住宿_G_attractiveness','社会团体_G_attractiveness','考试培训_G_attractiveness','购物_G_attractiveness','酒店会议中心_G_attractiveness',
    '金融保险_G_attractiveness','风景_G_attractiveness','餐饮_G_attractiveness','高校科研_G_attractiveness'
]
attractiveness_cols = [col for col in attractiveness_cols if col in features_df.columns] # 确保存在

if attractiveness_cols:
    print(f"  绘制吸引力特征分布图 ({len(attractiveness_cols)} 个特征)...")
    num_plots = len(attractiveness_cols)
    num_cols_grid = 4
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    # 检查计算出的格子数是否符合预期
    print(f"    吸引力特征图网格: {num_rows_grid} 行 x {num_cols_grid} 列 = {num_rows_grid * num_cols_grid} 个格子")
    plt.figure(figsize=(num_cols_grid * 4, num_rows_grid * 3))
    for i, col in enumerate(attractiveness_cols):
        # 再次检查索引是否超界 (理论上不应发生)
        if i + 1 > num_rows_grid * num_cols_grid:
            print(f"警告: 尝试访问第 {i+1} 个子图，但总格子数只有 {num_rows_grid * num_cols_grid}")
            break
        plt.subplot(num_rows_grid, num_cols_grid, i + 1)
        sns.histplot(features_df[col], kde=True, bins=30)
        plt.title(f'{col} 分布', fontsize=8) # 缩小标题字号
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(fontsize=7) # 缩小刻度字号
        plt.yticks(fontsize=7)
    plt.suptitle('吸引力特征分布', fontsize=14, y=1.02) # 添加总标题
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'feature_dist_attractiveness.png'))
    print(f"  吸引力特征分布图已保存到 {os.path.join(OUTPUT_VIS_DIR, 'feature_dist_attractiveness.png')}")
    plt.close()
else:
    print("  未找到用于绘制的吸引力特征列。")

# 5. 相关性分析 (分开绘制热力图)
# --- 5.1 时间块特征相关性 ---
print("  计算并绘制时间块特征相关性热力图...")
# 定义时间块特征列名 (96个)
time_block_cols = [
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
    'ea_block_40', 'ea_block_41', 'ea_block_42', 'ea_block_43', 'ea_block_44', 'ea_block_45', 'ea_block_46','ea_block_47'
]
# 确保列存在
time_block_cols_present = [col for col in time_block_cols if col in features_df.columns]

if time_block_cols_present:
    try:
        correlation_matrix_time = features_df[time_block_cols_present].corr()
        plt.figure(figsize=(24, 20)) # 96x96 的热力图需要较大尺寸
        sns.heatmap(correlation_matrix_time, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('时间块特征 (96维) 相关性热力图', fontsize=16)
        # 对于太多标签，可以选择不显示刻度标签或只显示部分
        plt.xticks(ticks=np.arange(len(time_block_cols_present)) + 0.5, labels=time_block_cols_present, fontsize=5, rotation=90)
        plt.yticks(ticks=np.arange(len(time_block_cols_present)) + 0.5, labels=time_block_cols_present, fontsize=5, rotation=0)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_VIS_DIR, 'feature_correlation_heatmap_time_blocks.png')
        plt.savefig(save_path, dpi=200) # 提高分辨率以便放大查看
        print(f"  时间块特征相关性热力图已保存到: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  计算或绘制时间块特征热力图时出错: {e}")
else:
    print("  未找到用于绘制的时间块特征列。")


# --- 5.2 吸引力特征相关性 ---
print("\n  计算并绘制吸引力特征相关性热力图...")
# 定义吸引力特征列名 (19个)
attractiveness_cols = [
    '事务所_G_attractiveness', '产业园工厂_G_attractiveness','体育休闲_G_attractiveness','公司企业_G_attractiveness','医疗_G_attractiveness',
    '商务楼_G_attractiveness','固定性住宿_G_attractiveness','展馆_G_attractiveness','度假_G_attractiveness','政府机构_G_attractiveness',
    '流动性住宿_G_attractiveness','社会团体_G_attractiveness','考试培训_G_attractiveness','购物_G_attractiveness','酒店会议中心_G_attractiveness',
    '金融保险_G_attractiveness','风景_G_attractiveness','餐饮_G_attractiveness','高校科研_G_attractiveness'
]
# 确保列存在
attractiveness_cols_present = [col for col in attractiveness_cols if col in features_df.columns]

if attractiveness_cols_present:
    try:
        correlation_matrix_attr = features_df[attractiveness_cols_present].corr()
        plt.figure(figsize=(10, 8)) # 19x19 的热力图尺寸可以小一些
        sns.heatmap(correlation_matrix_attr, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=.5, square=True, annot_kws={"size": 7}) # 显示数值
        plt.title('吸引力特征 (19维) 相关性热力图', fontsize=14)
        plt.xticks(fontsize=8, rotation=45, ha='right')
        plt.yticks(fontsize=8, rotation=0)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_VIS_DIR, 'feature_correlation_heatmap_attractiveness.png')
        plt.savefig(save_path, dpi=150)
        print(f"  吸引力特征相关性热力图已保存到: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  计算或绘制吸引力特征热力图时出错: {e}")
else:
    print("  未找到用于绘制的吸引力特征列。")


# 6. 降维与可视化 (PCA 和 t-SNE)
print("\n--- 降维可视化 (PCA & t-SNE) ---")
# 降维前建议先标准化特征
print("  应用 StandardScaler 进行标准化...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df.fillna(0)) # 再次处理可能的 NaN

# PCA
print("  执行 PCA...")
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)
print(f"  PCA 完成。解释方差比例: {pca.explained_variance_ratio_}")

# t-SNE
# 注意: t-SNE 对大数据集计算较慢，可以考虑先用 PCA 降维（例如到 50 维）再用 t-SNE
print("  执行 t-SNE (如果数据量大，可能需要较长时间)...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42) # 可以调整 perplexity 和 n_iter
features_tsne = tsne.fit_transform(features_scaled) # 直接在 scaled features 上运行
# 或者先 PCA 再 t-SNE:
# pca_50 = PCA(n_components=50)
# features_pca_50 = pca_50.fit_transform(features_scaled)
# features_tsne = tsne.fit_transform(features_pca_50)
print("  t-SNE 完成。")

# 绘图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(features_pca[:, 0], features_pca[:, 1], s=5, alpha=0.7)
plt.title('PCA 降维可视化 (前两个主成分)')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], s=5, alpha=0.7)
plt.title('t-SNE 降维可视化')
plt.xlabel('t-SNE 维度 1')
plt.ylabel('t-SNE 维度 2')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'feature_dimensionality_reduction.png'))
print("PCA 和 t-SNE 可视化结果已保存到 feature_dimensionality_reduction.png")
# plt.show()
plt.close()

print("\n--- 特征探索分析完成 ---")
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from collections import Counter
import glob

# --- 配置参数 ---
OD_FILE_PATH = r'data/station_od_idv_1022_attributes.xlsx'
# 与您 poi_embedding.py 和 activity_POI_gravity.py 中一致的POI数据根目录
POI_DATA_ROOT = r'Q:/LEARNNING-POSTGRADUATE/BaiduSyncdisk/LEARNNING-POSTGRADUATE/站点深化/ATP inferred/MGGDC/data/POI'
OUTPUT_CSV_FILE = 'od_points_poi_shannon_entropy.csv'

# OD 文件列名
OD_ORDER_COL = 'order'
OD_LON_COL = 'd_lon'
OD_LAT_COL = 'd_lat'
OD_AGE_COL = 'age'  # 确保此列存在或按需调整

# POI 文件列名
POI_LON_COL = 'wgs84_x'
POI_LAT_COL = 'wgs84_y'
POI_SMALL_CATEGORY_COL = '中类'  # 这是计算熵的基础

# 空间参数
BUFFER_RADIUS_METERS = 500
GEOGRAPHIC_CRS = "EPSG:4326"
# 使用与 activity_POI_gravity.py 一致的投影坐标系
PROJECTED_CRS = "EPSG:32650"

# POI 文件发现逻辑 (基于 poi_embedding.py)
CITIES = ['广州市', '佛山市', '东莞市', '深圳市']
# 这些是大类，用于匹配文件名
POI_FILENAME_CATEGORIES = [
    '餐饮服务', '道路附属设施', '地名地址信息', '风景名胜', '公共设施', '公司企业',
    '购物服务', '交通设施服务', '金融保险服务', '科教文化服务', '摩托车服务', '汽车服务',
    '汽车维修', '汽车销售', '商务住宅', '生活服务', '事件活动', '体育休闲服务', '虚拟数据',
    '医疗保健服务', '政府机构及社会团体', '住宿服务'
]


def calculate_shannon_entropy(series_of_categories):
    """为POI类别序列计算香农熵"""
    counts = Counter(series_of_categories)
    total_count = len(series_of_categories)
    if total_count == 0:
        return 0.0  # 如果缓冲区内没有POI，熵为0

    probabilities = [count / total_count for count in counts.values()]
    # 仅对概率大于0的项计算，避免 log2(0)
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


# --- 1. 加载OD点数据 ---
print("--- 1. 加载OD点数据 ---")
points_df = pd.read_excel(OD_FILE_PATH)
# 确保关键列存在且无缺失值
points_df = points_df.dropna(subset=[OD_ORDER_COL, OD_LON_COL, OD_LAT_COL]).reset_index(drop=True)

# 处理年龄列
if OD_AGE_COL in points_df.columns:
    if points_df[OD_AGE_COL].dtype == 'object':
        points_df[OD_AGE_COL].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    points_df.dropna(subset=[OD_AGE_COL], inplace=True)  # 如果需要年龄信息，则去除缺失行
else:
    print(f"警告: 列 '{OD_AGE_COL}' 在OD数据中未找到。")

points_df[OD_LON_COL] = pd.to_numeric(points_df[OD_LON_COL], errors='coerce')
points_df[OD_LAT_COL] = pd.to_numeric(points_df[OD_LAT_COL], errors='coerce')
points_df = points_df.dropna(subset=[OD_LON_COL, OD_LAT_COL]).reset_index(drop=True)

if points_df.empty:
    print("错误: OD点数据处理后为空。脚本退出。")
    exit()

points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df[OD_LON_COL], points_df[OD_LAT_COL]),
    crs=GEOGRAPHIC_CRS
)
print(f"加载并处理了 {len(points_gdf)} 个有效的OD点。")

# --- 2. 加载并合并POI数据 ---
print("\n--- 2. 加载并合并POI数据 ---")
poi_files_to_load = []
for city in CITIES:
    for category_name in POI_FILENAME_CATEGORIES:
        filename = os.path.join(POI_DATA_ROOT, f"{city}_广东_202407_{category_name}.csv")
        if os.path.exists(filename):
            poi_files_to_load.append(filename)

if not poi_files_to_load:
    print(f"错误: 在 '{POI_DATA_ROOT}' 未找到任何POI文件。脚本退出。")
    exit()

all_pois_list = []
print(f"找到 {len(poi_files_to_load)} 个POI文件，正在加载...")
for file_path in poi_files_to_load:
    try:
        df_poi_temp = pd.read_csv(file_path, usecols=[POI_LON_COL, POI_LAT_COL, POI_SMALL_CATEGORY_COL],
                                  encoding='utf8', low_memory=False)
    except UnicodeDecodeError:
        df_poi_temp = pd.read_csv(file_path, usecols=[POI_LON_COL, POI_LAT_COL, POI_SMALL_CATEGORY_COL], encoding='gbk',
                                  low_memory=False)
    except ValueError:  # usecols指定的列可能不存在
        print(
            f"警告: 文件 {os.path.basename(file_path)} 可能缺少必要的列 ({POI_LON_COL}, {POI_LAT_COL}, {POI_SMALL_CATEGORY_COL})。已跳过。")
        continue

    df_poi_temp[POI_SMALL_CATEGORY_COL] = df_poi_temp[POI_SMALL_CATEGORY_COL].astype(str).fillna('未知类别')
    df_poi_temp[POI_LON_COL] = pd.to_numeric(df_poi_temp[POI_LON_COL], errors='coerce')
    df_poi_temp[POI_LAT_COL] = pd.to_numeric(df_poi_temp[POI_LAT_COL], errors='coerce')
    df_poi_temp = df_poi_temp.dropna(subset=[POI_LON_COL, POI_LAT_COL])

    if not df_poi_temp.empty:
        all_pois_list.append(df_poi_temp)

if not all_pois_list:
    print("错误: 未能从任何文件中加载有效的POI数据。脚本退出。")
    exit()

all_pois_df = pd.concat(all_pois_list, ignore_index=True)
poi_gdf = gpd.GeoDataFrame(
    all_pois_df,
    geometry=gpd.points_from_xy(all_pois_df[POI_LON_COL], all_pois_df[POI_LAT_COL]),
    crs=GEOGRAPHIC_CRS
)
# 将用于熵计算的POI中类列重命名，以清晰区分
poi_gdf = poi_gdf.rename(columns={POI_SMALL_CATEGORY_COL: 'poi_small_category_for_entropy'})
print(f"加载并合并了 {len(poi_gdf)} 个有效的POI记录。")

# --- 3. 数据投影 ---
print("\n--- 3. 数据投影 ---")
points_projected_gdf = points_gdf.to_crs(PROJECTED_CRS)
poi_projected_gdf = poi_gdf.to_crs(PROJECTED_CRS)
print(f"数据已投影到 {PROJECTED_CRS}。")

# --- 4. 空间连接与香农熵计算 ---
print("\n--- 4. 空间连接与香农熵计算 ---")
# 为OD点创建缓冲区
points_buffers_gdf = points_projected_gdf.copy()
points_buffers_gdf['geometry'] = points_buffers_gdf.geometry.buffer(BUFFER_RADIUS_METERS)
# 仅保留OD点标识符和缓冲区几何图形用于连接
points_buffers_gdf = points_buffers_gdf[[OD_ORDER_COL, 'geometry']]

# 执行空间连接：找到每个OD点缓冲区内的POI
# 仅选择POI的几何图形和用于熵计算的中类列
pois_for_join_gdf = poi_projected_gdf[['geometry', 'poi_small_category_for_entropy']].copy()
pois_in_buffers = gpd.sjoin(
    pois_for_join_gdf,
    points_buffers_gdf,
    how="inner",
    predicate='within'  # 精确查找缓冲区内的POI
)

if pois_in_buffers.empty:
    print("警告: 所有OD点的缓冲区内均未找到POI。所有点的香农熵将为0。")
    points_df['poi_shannon_entropy'] = 0.0
else:
    print(f"在OD点缓冲区内共找到 {len(pois_in_buffers)} 个POI实例。")
    # sjoin后，来自points_buffers_gdf的列名可能会被加上后缀，如 'order_right'
    # 我们需要确定正确的OD点标识符列名
    od_id_col_in_join = OD_ORDER_COL + '_right' if OD_ORDER_COL + '_right' in pois_in_buffers.columns else OD_ORDER_COL

    if od_id_col_in_join not in pois_in_buffers.columns:
        print(
            f"错误: 无法在空间连接结果中找到OD标识符列 (尝试过 '{od_id_col_in_join}' 和 '{OD_ORDER_COL}')。可用列: {pois_in_buffers.columns}")
        points_df['poi_shannon_entropy'] = np.nan  # 标记为错误或未知
    else:
        print(f"按 '{od_id_col_in_join}' 分组计算香农熵...")
        # 对每个OD点缓冲区内的POI小类列表计算香农熵
        entropy_series = pois_in_buffers.groupby(od_id_col_in_join)['poi_small_category_for_entropy'].apply(
            calculate_shannon_entropy)
        entropy_df = entropy_series.reset_index()
        # 将分组的ID列重命名回原始的OD标识符列名，以便合并
        entropy_df.columns = [OD_ORDER_COL, 'poi_shannon_entropy']

        # 将熵结果合并回原始的OD DataFrame (points_df)
        points_df = points_df.merge(entropy_df, on=OD_ORDER_COL, how='left')
        # 对于在缓冲区内没有POI的OD点，其熵在合并后会是NaN，填充为0
        points_df['poi_shannon_entropy'] = points_df['poi_shannon_entropy'].fillna(0.0)
        print("香农熵计算并合并完成。")

# --- 5. 保存结果 ---
print("\n--- 5. 保存结果 ---")
if 'poi_shannon_entropy' not in points_df.columns:
    points_df['poi_shannon_entropy'] = np.nan  # 如果因错误未计算，则添加NaN列

# 选择原始OD数据的所有列，并确保熵列也在其中
columns_to_save = [col for col in points_df.columns if col != 'geometry']  # 移除可能存在的geometry列
if 'poi_shannon_entropy' not in columns_to_save and 'poi_shannon_entropy' in points_df.columns:
    columns_to_save.append('poi_shannon_entropy')

final_output_df = points_df[columns_to_save]

final_output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
print(f"包含POI香农熵的结果已保存至: '{OUTPUT_CSV_FILE}'")
print("\n输出文件前5行预览 (部分列):")
preview_cols = [OD_ORDER_COL, OD_LON_COL, OD_LAT_COL, 'poi_shannon_entropy']
# 确保预览的列都存在
preview_cols = [col for col in preview_cols if col in final_output_df.columns]
print(final_output_df[preview_cols].head())

print("\n脚本执行完毕。")
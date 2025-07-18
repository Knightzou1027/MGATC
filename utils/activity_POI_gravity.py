# 导入所需的库
import pandas as pd
import numpy as np
import glob
import os
import geopandas as gpd
from shapely.geometry import Point
import warnings

# --- 配置参数 ---
# OD(目的地)点数据文件路径
od_file = r'data/station_od_idv_1018_attributes.xlsx'
# 包含 POI Excel 文件的目录路径
poi_dir = 'Q:/LEARNNING-POSTGRADUATE/BaiduSyncdisk/LEARNNING-POSTGRADUATE/站点深化/ATP inferred/MGGDC/data/node-POI' # 使用 './' 表示当前目录
# 缓冲区半径 (米)
buffer_radius_meters = 500
# 目标投影坐标系 (Projected CRS) - 例如 UTM Zone 50N for GZ/SZ/DG/FS
PROJECTED_CRS = 'EPSG:32650'
# 原始地理坐标系 (Geographic CRS) - 通常是 WGS84
GEOGRAPHIC_CRS = 'EPSG:4326'

# --- 列名配置 (请根据你的实际文件修改!) ---
# OD 文件相关列名
od_lon_col = 'd_lon'     # OD 文件中的目的地经度列
od_lat_col = 'd_lat'     # OD 文件中的目的地纬度列
od_city_col = 'd_city'   # OD 文件中标识目的地城市的列 (预期为拼音)

# POI 文件相关列名
poi_lon_col = 'wgs84_x'  # POI 文件中的经度列
poi_lat_col = 'wgs84_y'  # POI 文件中的纬度列
poi_city_col = 'cityname' # POI 文件中标识城市名称的列 (预期为中文)

# --- 核心修改：定义拼音到中文的映射 ---
# !!! 请根据你的 d_city 列中的实际拼音和对应的中文名进行补充和修改 !!!
pinyin_to_chinese_map = {
    'guangzhou': '广州', # 假设 POI cityname 是 '广州市'
    'shenzhen': '深圳',
    'foshan': '佛山',
    'dongguan': '东莞'
    # 添加你数据中可能出现的其他城市映射...
}
# --- ---

# --- 1. 加载和预处理数据 ---

# --- 1a. 加载 OD 数据、转换城市名为中文、创建 GeoDataFrame ---
print(f"===== 步骤 1a: 加载 OD 数据 ({od_file}) =====")
try:
    od_df = pd.read_excel(od_file)
    print(f"  原始 OD 记录数: {len(od_df)}")
    required_od_cols = [od_lon_col, od_lat_col, od_city_col]
    if not all(col in od_df.columns for col in required_od_cols):
        missing_cols = [col for col in required_od_cols if col not in od_df.columns]
        raise ValueError(f"OD 文件缺少必要的列: {', '.join(missing_cols)}")

    initial_len = len(od_df)
    od_df.dropna(subset=required_od_cols, inplace=True)
    if initial_len != len(od_df): print(f"  已删除 {initial_len - len(od_df)} 行，因为必需列信息缺失。")

    # 清理拼音城市名
    od_df[od_city_col] = od_df[od_city_col].astype(str).str.strip().str.lower()
    od_df = od_df[od_df[od_city_col].str.len() > 0]

    # --- 核心修改：转换 OD 城市名为中文 ---
    print(f"  正在将 '{od_city_col}' 列 (拼音) 映射为中文...")
    od_df['od_city_chinese'] = od_df[od_city_col].map(pinyin_to_chinese_map)
    # 检查未成功映射的城市
    unmapped_cities = od_df[od_df['od_city_chinese'].isna()][od_city_col].unique()
    if len(unmapped_cities) > 0:
        print(f"  警告：以下城市拼音未能映射到中文，相关行将被删除: {', '.join(unmapped_cities)}")
        print(f"  请检查并更新代码中的 'pinyin_to_chinese_map' 字典。")
        initial_len = len(od_df)
        od_df.dropna(subset=['od_city_chinese'], inplace=True)
        print(f"  已删除 {initial_len - len(od_df)} 行，因为城市名无法映射。")
    # --------------------------------------

    # 清洗和转换坐标
    od_df[od_lon_col] = pd.to_numeric(od_df[od_lon_col], errors='coerce')
    od_df[od_lat_col] = pd.to_numeric(od_df[od_lat_col], errors='coerce')
    od_df.dropna(subset=[od_lon_col, od_lat_col], inplace=True)

    if od_df['age'].dtype == 'object':
        od_df['age'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    od_df.dropna(subset=['age'], inplace=True)

    # 创建 GeoDataFrame
    print("  正在创建 OD 点的 GeoDataFrame...")
    geometry_od = [Point(xy) for xy in zip(od_df[od_lon_col], od_df[od_lat_col])]
    od_gdf = gpd.GeoDataFrame(od_df, geometry=geometry_od, crs=GEOGRAPHIC_CRS)
    print(f"  已创建 OD GeoDataFrame，包含 {len(od_gdf)} 个有效点。")
    od_gdf['original_od_index'] = od_gdf.index # 保留原始索引

    cities_in_od_chinese = sorted(od_gdf['od_city_chinese'].unique())
    print(f"  OD 数据中包含的城市 (中文): {', '.join(cities_in_od_chinese)}")

except FileNotFoundError:
    print(f"错误：未在 {od_file} 找到 OD 文件。程序退出。")
    exit()
except ImportError:
     print("错误：需要安装 'geopandas' 和 'shapely' 库。程序退出。")
     exit()
except Exception as e:
    print(f"加载 OD 文件或创建 GeoDataFrame 时出错: {e}。程序退出。")
    exit()

# --- 1b. 加载 POI 数据、创建 GeoDataFrame、并按中文城市统计全局数量 ---
poi_global_counts_by_city = {} # 存储按城市(中文)和类别分组的 POI 全局总数
all_poi_dfs_processed = []

print(f"\n===== 步骤 1b: 加载和处理 POI 数据 (来自 {poi_dir}) =====")
# 查找 Excel 文件
poi_files = [f for f in glob.glob(os.path.join(poi_dir, '*.xlsx')) if not os.path.basename(f).startswith('~') and os.path.basename(f) != os.path.basename(od_file).replace('.csv','.xlsx')]
print(f"  找到 {len(poi_files)} 个潜在的 POI Excel 文件。")
if not poi_files: print("错误：在指定目录未找到任何 POI Excel (.xlsx) 文件。程序退出。"); exit()

for f in poi_files:
    filename = os.path.basename(f)
    category_name = filename.split('.')[0] # 提取类别名称
    if not category_name: continue # 跳过无法解析名称的文件

    print(f"  > 正在处理: {filename} (类别: {category_name})")
    try:
        # 读取 Excel 文件
        df = pd.read_excel(f)
        required_poi_cols = [poi_lon_col, poi_lat_col, poi_city_col]
        if not all(col in df.columns for col in required_poi_cols):
             missing_cols = [col for col in required_poi_cols if col not in df.columns]
             print(f"    警告：跳过。缺少必要的列: {', '.join(missing_cols)}。")
             continue

        # 数据清洗
        initial_poi_len = len(df)
        df.dropna(subset=required_poi_cols, inplace=True)
        # 清理中文城市名中的空格（如果有的话）
        df[poi_city_col] = df[poi_city_col].astype(str).str.strip()
        df = df[df[poi_city_col].str.len() > 0]
        if initial_poi_len != len(df): print(f"    已删除 {initial_poi_len - len(df)} 行，因为必需列信息缺失或城市名无效。")

        # 转换坐标类型
        df[poi_lon_col] = pd.to_numeric(df[poi_lon_col], errors='coerce')
        df[poi_lat_col] = pd.to_numeric(df[poi_lat_col], errors='coerce')
        df.dropna(subset=[poi_lon_col, poi_lat_col], inplace=True)

        if len(df) > 0:
            df['poi_category'] = category_name
            # 创建 GeoDataFrame
            geometry_poi = [Point(xy) for xy in zip(df[poi_lon_col], df[poi_lat_col])]
            temp_gdf = gpd.GeoDataFrame(
                df[[poi_city_col, 'poi_category']], # 使用原始中文城市名列
                geometry=geometry_poi,
                crs=GEOGRAPHIC_CRS
            )
            all_poi_dfs_processed.append(temp_gdf)
            print(f"    -> 成功加载并处理 {len(temp_gdf)} 个有效 POI。")
        else:
            print(f"    -> 警告：文件中未找到有效的 POI 数据。")

    except Exception as e:
        print(f"    加载或处理 POI 文件 {filename} 时出错: {e}")

# 检查是否成功加载了任何 POI 数据
if not all_poi_dfs_processed: print("\n错误：未能加载任何有效的 POI 数据。程序退出。"); exit()

# 合并所有处理后的 POI GeoDataFrame
print("\n  合并所有 POI GeoDataFrame...")
poi_gdf_all = pd.concat(all_poi_dfs_processed, ignore_index=True)
del all_poi_dfs_processed, temp_gdf
print(f"  合并完成，共 {len(poi_gdf_all)} 个 POI。")

# 按中文城市和类别计算全局数量
print("  正在按中文城市统计全局 POI 数量...")
# 使用原始中文 poi_city_col 列进行分组
city_category_counts = poi_gdf_all.groupby([poi_city_col, 'poi_category']).size()
poi_global_counts_by_city = city_category_counts.unstack(fill_value=0).to_dict('index')

print("  全局 POI 数量统计 (按中文城市):")
for city_chinese, counts in poi_global_counts_by_city.items():
    print(f"  - {city_chinese}: {sum(counts.values())} total")

# --- 2. 使用 GeoPandas进行空间处理 ---

# --- 2a. 投影到米制坐标系 ---
print(f"\n===== 步骤 2a: 投影到目标坐标系 ({PROJECTED_CRS}) =====")
try:
    print("  正在投影 OD 点...")
    od_gdf_proj = od_gdf.to_crs(PROJECTED_CRS)
    print("  正在投影 POI 点...")
    poi_gdf_all_proj = poi_gdf_all.to_crs(PROJECTED_CRS)
    print("  投影完成。")
except Exception as e: print(f"错误：坐标系投影失败: {e}。程序退出。"); exit()

# --- 2b. 创建 OD 点缓冲区 ---
print(f"\n===== 步骤 2b: 创建 OD 点缓冲区 ({buffer_radius_meters}米) =====")
od_gdf_proj['buffer_geometry'] = od_gdf_proj.geometry.buffer(buffer_radius_meters)
od_gdf_proj = od_gdf_proj.set_geometry('buffer_geometry')
print("  缓冲区创建完成。")

# --- 2c. 执行空间连接 (Spatial Join) ---
print("\n===== 步骤 2c: 执行空间连接查找缓冲区内的 POI =====")
joined_gdf = gpd.sjoin(od_gdf_proj, poi_gdf_all_proj, how='left', predicate='contains', lsuffix='od', rsuffix='poi')
del od_gdf_proj, poi_gdf_all_proj # 清理内存
print(f"  空间连接完成。找到 {joined_gdf['index_poi'].notna().sum()} 个 OD-POI 匹配关系。")

# --- 2d. 计算精确距离 (向量化，无需重新投影和合并几何列) ---
print("\n===== 步骤 2d: 计算匹配的 OD-POI 距离 (向量化优化 v2) =====")

# 筛选出实际发生连接的行
mask = joined_gdf['index_poi'].notna()
joined_gdf_matched = joined_gdf[mask].copy() # 使用副本操作

# 检查是否有匹配项
if not joined_gdf_matched.empty:
    print("  准备用于距离计算的几何对象 (使用已投影的点)...")
    # 获取匹配行的 OD 点原始索引和 POI 点索引
    od_indices = joined_gdf_matched['original_od_index']
    poi_indices = joined_gdf_matched['index_poi'].astype(int)

    # --- 获取对齐的、已在步骤 2a 投影过的点几何对象 ---
    # 1. 获取 OD 点的投影后几何 (从原始 od_gdf 投影一次得到)
    #    确保索引是原始 OD 索引 (original_od_index)
    print("    提取对齐的 OD 点投影几何...")
    od_points_proj_geom = od_gdf.set_index('original_od_index').to_crs(PROJECTED_CRS).geometry
    od_geoms_aligned = od_points_proj_geom.loc[od_indices]

    # 2. 获取 POI 点的投影后几何 (从原始 poi_gdf_all 投影一次得到)
    #    确保索引是 POI 的原始索引 (0, 1, 2...)
    print("    提取对齐的 POI 点投影几何...")
    poi_points_proj_geom = poi_gdf_all.set_index(poi_gdf_all.index).to_crs(PROJECTED_CRS).geometry
    poi_geoms_aligned = poi_points_proj_geom.loc[poi_indices]
    # ---

    # --- 关键修改：向量化距离计算 ---
    # 重设 POI 几何序列的索引，使其与 OD 几何序列 (以及 joined_gdf_matched) 的索引对齐
    # joined_gdf_matched 的索引是 sjoin 操作生成的，我们需要用这个索引来对齐
    od_geoms_aligned.index = joined_gdf_matched.index
    poi_geoms_aligned.index = joined_gdf_matched.index

    print("  计算距离 (向量化)...")
    # 直接在两个对齐的 GeoSeries 之间计算距离
    distances = od_geoms_aligned.distance(poi_geoms_aligned)

    # 将计算得到的距离赋值回 joined_gdf
    print("  将距离结果赋值回主表...")
    joined_gdf.loc[mask, 'distance_meters'] = distances
    print("  距离计算完成。")

else:
    print("  没有找到匹配的 OD-POI 对，跳过距离计算。")
    if 'distance_meters' not in joined_gdf.columns:
         joined_gdf['distance_meters'] = np.nan # 确保列存在

# --- 3. 聚合结果并计算 Rho, C, G ---
print("\n===== 步骤 3: 聚合结果并计算 Rho, C, G 指标 =====")
# 处理 sjoin 可能添加的后缀
poi_category_col_name = 'poi_category_poi' if 'poi_category_poi' in joined_gdf.columns else 'poi_category'
if poi_category_col_name not in joined_gdf.columns:
    print(f"错误：在连接结果中找不到 POI 类别列 ('{poi_category_col_name}' 或 'poi_category')。")
    exit()

print("  正在分组聚合统计...")
agg_funcs = {'distance_meters': ['count', 'mean']}
grouped_stats = joined_gdf.dropna(subset=['distance_meters']).groupby(['original_od_index', poi_category_col_name]).agg(agg_funcs)
grouped_stats.columns = ['local_count', 'avg_dist_local']
grouped_stats['avg_dist_local'] = grouped_stats['avg_dist_local'].replace(0, 1e-6)

print("  正在整理为宽表格式...")
final_stats = grouped_stats.unstack(level=poi_category_col_name)
if isinstance(final_stats.columns, pd.MultiIndex):
     final_stats.columns = [f'{col[1]}_{col[0]}' for col in final_stats.columns]

poi_categories_found = poi_gdf_all['poi_category'].unique()
for cat in poi_categories_found:
    count_col, dist_col = f'{cat}_local_count', f'{cat}_avg_dist_local'
    if count_col in final_stats.columns: final_stats[count_col] = final_stats[count_col].fillna(0).astype(int)
    else: final_stats[count_col] = 0
    if dist_col not in final_stats.columns: final_stats[dist_col] = np.nan

# 计算 Rho, C, G
print("  正在计算 Rho, C, G...")
metric_cols = []
for cat in poi_categories_found: metric_cols.extend([f'{cat}_rho', f'{cat}_C', f'{cat}_G_attractiveness'])
for col in metric_cols:
     if col not in final_stats.columns: final_stats[col] = 0.0

# 合并 OD 的中文城市名以查找全局数量
final_stats = final_stats.merge(od_gdf[['original_od_index', 'od_city_chinese']].set_index('original_od_index'),
                                left_index=True, right_index=True, how='left')

for od_idx, row in final_stats.iterrows():
    od_city_chinese_name = row['od_city_chinese'] # 获取中文城市名
    # 使用中文城市名查找全局数量
    city_poi_counts = poi_global_counts_by_city.get(od_city_chinese_name, {})
    total_rho_sum_for_point = 0.0
    rho_values_point = {}

    for category in poi_categories_found:
        local_count = row.get(f'{category}_local_count', 0)
        global_count_in_city = city_poi_counts.get(category, 0)
        rho = float(local_count) / global_count_in_city if global_count_in_city > 0 else 0.0
        rho_values_point[category] = rho
        final_stats.loc[od_idx, f'{category}_rho'] = rho
        total_rho_sum_for_point += rho

    for category in poi_categories_found:
        rho = rho_values_point.get(category, 0.0)
        c = rho / total_rho_sum_for_point if total_rho_sum_for_point > 1e-9 else 0.0
        final_stats.loc[od_idx, f'{category}_C'] = c
        avg_dist = row.get(f'{category}_avg_dist_local', np.nan)
        g = c / (avg_dist**2) if c > 0 and not pd.isna(avg_dist) and avg_dist > 1e-9 else 0.0
        final_stats.loc[od_idx, f'{category}_G_attractiveness'] = g

# 删除临时的中文城市名列
final_stats.drop(columns=['od_city_chinese'], inplace=True)
print("  指标计算完成。")

# --- 4. 合并最终结果到原始 OD 数据 ---
print("\n===== 步骤 4: 合并最终指标到原始 OD 数据 =====")
# 使用原始 OD DataFrame (od_df) 和它的索引进行合并
od_df_final = od_df.merge(final_stats, left_index=True, right_index=True, how='left')

# 填充缺失值
print("  正在填充缺失的指标值...")
fill_values = {}
for col in final_stats.columns:
    if '_count' in col or '_rho' in col or '_C' in col or '_attractiveness' in col: fill_values[col] = 0.0
    elif '_dist' in col: fill_values[col] = np.nan
od_df_final.fillna(fill_values, inplace=True)
print("  填充完成。")

# --- 5. 保存输出文件 ---
output_filename = 'od_points_with_poi_attractiveness_chinese_city_geopandas.csv'
print(f"\n===== 步骤 5: 保存最终结果到 {output_filename} =====")
od_df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("\n===== 处理完成 =====")
print(f"最终输出的 DataFrame 维度: {od_df_final.shape}")
print("\n输出文件的前 5 行预览:")
# 增加显示的列数，以便看到新添加的列
pd.set_option('display.max_columns', None) # 显示所有列
print(od_df_final.head())
pd.reset_option('display.max_columns') # 恢复默认设置
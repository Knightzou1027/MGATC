import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import os

# --- 1. 加载点位数据 ---
print("--- 1. 加载点位数据 ---")

# 加载点位数据Excel文件
points_df = pd.read_excel(r'data/station_od_idv_1018_attributes.xlsx')
# 使用 'order' 列作为点位ID，'d_lon' 和 'd_lat' 作为坐标
# 去除 order, d_lon, d_lat 列中有空值的行
points_df = points_df.dropna(subset=['order', 'd_lon', 'd_lat']).reset_index(drop=True)

#过滤掉没有年龄信息的行
if points_df['age'].dtype == 'object':
     points_df['age'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
points_df.dropna(subset=['age'], inplace=True)

# 确保 order 列是唯一的 ID
if not points_df['order'].is_unique:
    print("警告：点位数据中的 'order' 列包含重复值，这可能会影响结果的唯一性。")

# 将经纬度列转换为数值类型，处理可能的非数值数据
points_df['d_lon'] = pd.to_numeric(points_df['d_lon'], errors='coerce')
points_df['d_lat'] = pd.to_numeric(points_df['d_lat'], errors='coerce')
# 再次去除因转换错误产生的空值行
points_df = points_df.dropna(subset=['d_lon', 'd_lat']).reset_index(drop=True)

# 检查转换后是否还有点位
if points_df.empty:
    print("错误：点位数据加载或处理后为空，没有有效的点位坐标。")
    exit()

points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df['d_lon'], points_df['d_lat']),
    crs="EPSG:4326" # 假设原始数据是WGS 84地理坐标系
)
print(f"已加载并处理 {len(points_gdf)} 个有效点位。")


# --- 2. 自动生成POI文件列表，加载并合并数据 ---
print("\n--- 2. 自动生成POI文件列表，加载并合并数据 ---")

# 定义城市列表
cities = ['广州市', '佛山市', '东莞市', '深圳市']

# 定义POI类别列表 (根据您的文件命名推测和补充，请根据实际情况调整)
categories = [
    '餐饮服务', '道路附属设施', '地名地址信息', '风景名胜', '公共设施', '公司企业',
    '购物服务', '交通设施服务', '金融保险服务', '科教文化服务', '摩托车服务', '汽车服务',
    '汽车维修', '汽车销售', '商务住宅', '生活服务', '事件活动', '体育休闲服务', '虚拟数据',
    '医疗保健服务','政府机构及社会团体','住宿服务'
]

poi_files = []
# 定义POI文件所在的根目录
poi_data_root = r'Q:\LEARNNING-POSTGRADUATE\BaiduSyncdisk\LEARNNING-POSTGRADUATE\站点深化\ATP inferred\MGGDC_1018\data\POI'

# 根据城市和类别自动生成文件名列表
for city in cities:
    for category in categories:
        # 构建完整的文件路径
        filename = os.path.join(poi_data_root, f"{city}_广东_202407_{category}.csv")
        # 检查文件是否存在才添加到列表 (这里不移除 except FileNotFoundError，只做 exists 检查)
        if os.path.exists(filename):
             poi_files.append(filename)


all_pois_list = []

# 假定所有POI文件都有 'wgs84_x', 'wgs84_y', '小类' 列，且坐标是WGS84
poi_lon_col = 'wgs84_x'
poi_lat_col = 'wgs84_y'
poi_category_col = '小类' # 使用 '小类' 列作为类别信息

print(f"正在尝试加载并合并 {len(poi_files)} 个POI文件...")
loaded_poi_count = 0
for file in poi_files:
    # 尝试多种编码读取CSV
    # 注意：这里没有错误处理，如果文件读取失败，程序将中断
    try:
        df = pd.read_csv(file, encoding='utf8')
    except:
        df = pd.read_csv(file, encoding='gbk')

    # 检查是否存在必要的列 (这里假定必要的列肯定存在，如果不存在将直接KeyError中断)
    df = df[[poi_lon_col, poi_lat_col, poi_category_col]]
    # 将类别转换为字符串并处理空值
    df[poi_category_col] = df[poi_category_col].astype(str).fillna('未知类别')

    # 将坐标列转换为数值类型，处理可能的非数值数据
    df[poi_lon_col] = pd.to_numeric(df[poi_lon_col], errors='coerce')
    df[poi_lat_col] = pd.to_numeric(df[poi_lat_col], errors='coerce')
    # 去除因转换错误产生的空值行
    df = df.dropna(subset=[poi_lon_col, poi_lat_col]).reset_index(drop=True)

    # 如果去除空值后DataFrame为空，跳过此文件
    if df.empty:
        continue

    # 将检测到的列名重命名为标准名称
    df = df.rename(columns={poi_lon_col: 'longitude', poi_lat_col: 'latitude', poi_category_col: 'category'})

    all_pois_list.append(df)
    loaded_poi_count += len(df)


# 合并所有POI数据
# 注意：如果 all_pois_list 为空，pd.concat 将报错
all_pois_df = pd.concat(all_pois_list, ignore_index=True)

if all_pois_df.empty:
     print("错误：合并后的POI数据为空，或者所有行都缺少必要的坐标/类别信息。")
     exit()

# 创建POI GeoDataFrame
poi_gdf = gpd.GeoDataFrame(
    all_pois_df,
    geometry=gpd.points_from_xy(all_pois_df['longitude'], all_pois_df['latitude']),
    crs="EPSG:4326" # 原始数据是WGS 84地理坐标系
)
print(f"已从所有找到并成功读取的文件中加载并合并 {len(poi_gdf)} 个有效POI。")

# --- 3. 将数据投影到合适的投影坐标系 ---
print("\n--- 3. 将数据投影到合适的投影坐标系 ---")
# 用户指定使用 49N，对应 EPSG:2381
projected_crs = "EPSG:2381" # WGS 84 / UTM zone 49N

# 注意：这里没有错误处理，如果投影失败，程序将中断
points_projected_gdf = points_gdf.to_crs(projected_crs)
poi_projected_gdf = poi_gdf.to_crs(projected_crs)
print(f"数据已投影到 {projected_crs}。")


# --- 4. 矢量化搜索周边POI，按距离排序，并构建“文档” ---
print("\n--- 4. 矢量化搜索周边POI，按距离排序，并构建“文档” ---")

search_radius = 500 # 搜索半径，单位米

# 过滤掉无效或空的POI几何对象
print("正在过滤无效或空的POI几何对象...")
# 确保 geometry 列存在且几何对象有效且非空
valid_poi_projected_gdf = poi_projected_gdf[
    (poi_projected_gdf.geometry.is_valid) &
    (~poi_projected_gdf.geometry.is_empty)
].copy()
print(f"过滤无效/空几何对象后剩余 {len(valid_poi_projected_gdf)} 个有效POI。")

# --- 矢量化操作：为所有点位创建缓冲区 ---
print("正在为所有点位创建缓冲区...")
# points_projected_gdf 已经是投影后的点位 GeoDataFrame
# .buffer() 方法可以矢量化地为所有点位创建缓冲区
# 确保点位 GeoDataFrame 有用于识别的列（这里是 'order'）
points_buffers_gdf = points_projected_gdf.copy()
# 创建缓冲区几何对象 (没有错误处理，如果点位几何对象无效，这里可能出错)
points_buffers_gdf['geometry'] = points_buffers_gdf.geometry.buffer(search_radius)
# 只保留点位ID和缓冲区几何对象
points_buffers_gdf = points_buffers_gdf[['order', 'geometry']]
# 重命名点位ID列
points_buffers_gdf = points_buffers_gdf.rename(columns={'order': 'point_id'})
print(f"已创建 {len(points_buffers_gdf)} 个点位缓冲区。")


# --- 矢量化操作：执行空间连接 ---
print("正在执行POI与点位缓冲区的空间连接...")
# 对过滤后的POI数据和所有点位缓冲区进行空间连接
# 查找落在任一缓冲区内的POI
# predicate='intersects' 查找与缓冲区相交的POI
# 注意：这里没有错误处理，如果空间连接失败，程序将中断
pois_in_buffers = gpd.sjoin(valid_poi_projected_gdf, points_buffers_gdf, how="inner", predicate='intersects')

print(f"空间连接完成，找到 {len(pois_in_buffers)} 个POI落在点位缓冲区内。")

# 如果空间连接结果为空，则没有点位找到周边POI
if pois_in_buffers.empty:
    print("在指定半径内没有点位找到周边POI。无法构建Doc2Vec文档。请检查搜索半径和POI数据覆盖范围。")
    exit()

# --- 对空间连接结果进行分组和处理 ---
print("正在分组并处理空间连接结果以构建Doc2Vec文档...")

point_documents = []
point_original_ids_with_pois = [] # 记录哪些原始点位成功构建了文档

# 按 point_id 分组
grouped_pois = pois_in_buffers.groupby('point_id')

# 遍历每个点位分组
for point_id, group in grouped_pois:
    # group 是一个 DataFrame，包含属于当前 point_id 的所有周边 POI

    # 需要获取当前点位的原始几何对象，以便计算距离
    # 从原始投影后的点位 GeoDataFrame 中查找 (没有错误处理，如果找不到点位ID，这里可能中断)
    original_point_geometry = points_projected_gdf[points_projected_gdf['order'] == point_id].geometry.iloc[0]

    # 计算这些周边POI到当前点位的距离
    # 直接在分组的 DataFrame 上计算距离 (没有错误处理，如果几何对象无效，这里可能出错)
    group['distance'] = group.distance(original_point_geometry)

    # 按距离从小到大排序周边POI
    group = group.sort_values(by='distance')

    # 提取POI类别序列 (使用 'category' 列)
    poi_categories_sequence = group['category'].astype(str).tolist()
    # 过滤掉可能存在的空字符串和 '未知类别'
    poi_categories_sequence = [cat for cat in poi_categories_sequence if cat.strip() != '' and cat != '未知类别']


    # 如果有提取到类别，才创建文档
    if poi_categories_sequence:
        # 为Doc2Vec创建TaggedDocument，标签使用点位ID（标签必须是字符串列表）
        point_documents.append(TaggedDocument(words=poi_categories_sequence, tags=[str(point_id)]))
        point_original_ids_with_pois.append(point_id) # 记录成功构建文档的原始点位ID


print(f"已从具有周边POI的点位中创建 {len(point_documents)} 个Doc2Vec文档。")

if not point_documents:
    print("错误：没有点位找到周边POI来构建文档。无法训练Doc2Vec模型。请检查搜索半径和POI数据覆盖范围。")
    exit()

# --- 5. 训练Doc2Vec模型 ---
print("\n--- 5. 训练Doc2Vec模型 ---")

# 定义Doc2Vec模型参数
vector_size = 100  # 嵌入向量的维度
window = 10        # 上下文窗口大小
min_count = 1      # 忽略词频低于此值的词
dm = 1             # 训练算法: 1为PV-DM, 0为PV-DBOW
epochs = 100        # 训练迭代次数
workers = 4        # 训练并行线程数

# 注意：这里没有错误处理，如果Doc2Vec训练失败，程序将中断
model = Doc2Vec(documents=point_documents, vector_size=vector_size, window=window, min_count=min_count, dm=dm, epochs=epochs, workers=workers)

print("Doc2Vec 模型训练完成。")

# --- 6. 获取每个有周边POI的点位的嵌入向量并保存 ---
print("\n--- 6. 获取每个有周边POI的点位的嵌入向量并保存 ---")

point_embeddings = {}
# 遍历成功构建文档的原始点位ID
for original_id in point_original_ids_with_pois:
     # 使用字符串标签访问文档向量 model.dv[]
     # 注意：这里没有错误处理，如果标签不存在，将KeyError中断
     point_embeddings[original_id] = model.dv[str(original_id)]


# 将结果存储到DataFrame中
# 注意：如果 point_embeddings 为空，pd.DataFrame.from_dict 将可能出错
point_embeddings_df = pd.DataFrame.from_dict(point_embeddings, orient='index', columns=[f'vec_{i+1}' for i in range(vector_size)])
point_embeddings_df.index.name = 'point_id'

print("\n点位嵌入向量生成完毕（仅针对找到周边POI的点位）：")
print(point_embeddings_df.head())

# 保存结果
# 将 point_id (原始 order) 从索引转为列
point_embeddings_df_final = point_embeddings_df.reset_index()

# 导出到Excel文件
output_excel_file = 'dpoint_poi_embeddings_.xlsx' # 文件名可以自定义
# 注意：这里没有错误处理，如果导出失败，程序将中断
point_embeddings_df_final.to_excel(output_excel_file, index=False) # index=False 表示不导出DataFrame的索引
print(f"\n点位嵌入向量数据已成功导出到 '{output_excel_file}'")
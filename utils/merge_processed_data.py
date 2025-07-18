import pandas as pd
import re

# 定义文件路径
# 请确保这些文件与你的Python脚本在同一个目录下，或者在这里提供完整的文件路径
file1 = r'data/dpoint_poi_embeddings_.xlsx'
file2 = r'data/od_data_timeblock.csv'
file3 = r'data/od_points_with_poi_attractiveness_chinese_city_geopandas.csv'
file4 = r'data/station_od_idv_1018_attributes.xlsx'

try:
    # 加载数据框
    df1 = pd.read_excel(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df4 = pd.read_excel(file4)

    # 按照'order'列顺序合并数据框
    # 使用inner合并确保只有在所有表中都存在'order'值的行被保留
    merged_df = pd.merge(df1, df2, on='order', how='inner')
    merged_df = pd.merge(merged_df, df3, on='order', how='inner')
    merged_df = pd.merge(merged_df, df4, on='order', how='inner')

    # 处理合并后带有后缀的重复列
    cols = merged_df.columns
    # 用于存储最终保留的列名及其对应在merged_df中的实际列名
    final_columns_map = {}

    for col in cols:
        # 使用正则表达式移除 '_x' 或 '_y' 或我们设置的临时后缀，获取原始列名
        base_name = re.sub(r'(_x|_y|_y_temp1|_y_temp2|_y_temp3)$', '', col)

        # 如果这个原始列名还没有被添加到最终列映射中，则添加
        if base_name not in final_columns_map:
            final_columns_map[base_name] = col
        # 如果已经存在，说明这是同一个概念的重复列（带有不同的后缀），我们忽略后续的，只保留第一个遇到的

    # 构建最终的数据框，只包含选定的列，并重命名列
    # 从映射的值中获取要保留的实际列名
    cols_to_keep_actual_names = list(final_columns_map.values())

    # 从映射的键中获取最终想要的列名顺序（即不带后缀的原始列名）
    final_column_names_order = list(final_columns_map.keys())

    # 筛选数据框中的列
    final_df = merged_df[cols_to_keep_actual_names]

    # 重命名列为不带后缀的原始列名
    final_df.columns = final_column_names_order

    # 显示合并后的数据框的前几行和列名，以便检查结果
    print("合并后的数据框头部：")
    print(merged_df.head())
    print("\n合并后的数据框列名：")
    print(merged_df.columns)

    #数据导出
    final_df.to_csv('od_idv_train_1018.csv', index=False)
    print("\n合并后的数据框已保存为 'od_idv_train_1018.csv'")

except FileNotFoundError as e:
    print(f"错误：文件未找到 - {e}")
    print("请检查文件路径是否正确，并确保文件存在。")
except KeyError as e:
    print(f"错误：合并键'order'未找到 - {e}")
    print("请确保所有文件中都存在名为'order'的列。")
except Exception as e:
    print(f"发生了一个错误：{e}")
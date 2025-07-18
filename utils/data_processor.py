import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

#读取原始数据






def load_and_preprocess_data(data_path):
    """
    加载并预处理数据（无需划分训练集和测试集）

    参数:
        data_path: str, 数据文件路径

    返回:
        data: DataFrame, 预处理后的数据
        features: ndarray, 提取的特征矩阵
    """
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 根据文件扩展名确定加载方式
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext == '.csv':
        data = pd.read_csv(data_path)
    elif file_ext in ['.xls', '.xlsx']:
        data = pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_ext}")

    # 数据预处理
    data = preprocess_data(data)

    # 特征提取
    features = extract_features(data)

    return data, features


def preprocess_data(data):
    """
    数据预处理

    参数:
        data: DataFrame, 原始数据

    返回:
        data: DataFrame, 预处理后的数据
    """
    # 复制数据，避免修改原始数据
    data = data.copy()

    # 1. 处理缺失值
    for col in data.columns:
        # 对数值型特征，用均值填充
        if data[col].dtype in ['int64', 'float64']:
            if data[col].isna().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)
        # 对类别特征，用众数填充
        else:
            if data[col].isna().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)

    # 2. 处理时间特征
    if 'departure_time' in data.columns:
        data['departure_time'] = pd.to_datetime(data['departure_time'])
        data['departure_hour'] = data['departure_time'].dt.hour
        data['departure_weekday'] = data['departure_time'].dt.weekday

    if 'arrival_time' in data.columns:
        data['arrival_time'] = pd.to_datetime(data['arrival_time'])
        data['arrival_hour'] = data['arrival_time'].dt.hour

    # 3. 检测并处理异常值（使用IQR方法）
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将异常值替换为边界值
        data[col] = data[col].clip(lower_bound, upper_bound)

    return data


def extract_features(data):
    """
    特征提取和转换

    参数:
        data: DataFrame, 预处理后的数据

    返回:
        features: ndarray, 提取的特征矩阵
    """
    # 复制数据，避免修改原始数据
    data_copy = data.copy()

    # 1. 分离数值特征和类别特征
    num_features = data_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. 标准化数值特征
    if num_features:
        scaler = StandardScaler()
        data_copy[num_features] = scaler.fit_transform(data_copy[num_features])

    # 3. 对类别特征进行独热编码
    if cat_features:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(data_copy[cat_features])

        # 创建独热编码的DataFrame
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=encoder.get_feature_names_out(cat_features),
            index=data_copy.index
        )

        # 合并回原始数据
        data_copy = pd.concat([data_copy[num_features], encoded_df], axis=1)

    # 返回特征矩阵
    return data_copy.values

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationarity(data, max_diff=3):
    # 存储原始序列
    original_series = data.values
    series = original_series.copy()
    
    # 初始化结果字典
    results = {
        'Stationarity': False,  # 平稳性状态
        'Differencing': 0,      # 差分次数
        'p-value': None,        # ADF检验的p值
        'Stationary Series': original_series  # 初始设为原始序列
    }
    
    # 检查序列是否为常数
    if np.all(series == series[0]):
        results['p-value'] = 0.0
        results['Stationarity'] = True  # 常数序列视为平稳
        return results
    
    # 执行最多max_diff次差分
    for d in range(max_diff + 1):
        if d > 0:  # 应用差分
            series = np.diff(series, n=1)
            if np.all(series == series[0]):  # 再次检查是否为常数
                results['p-value'] = 0.0
                results['Stationarity'] = True
                results['Differencing'] = d
                results['Stationary Series'] = series
                return results
        
        # 执行ADF检验
        adf_test = adfuller(series, autolag='AIC')
        p_value = adf_test[1]
        
        # 检查序列是否平稳
        if p_value < 0.05:
            results['Stationarity'] = True
            results['Differencing'] = d
            results['p-value'] = p_value
            results['Stationary Series'] = series
            break
        
        if d == max_diff:
            results['p-value'] = p_value
            results['Stationary Series'] = series
    
    return results

def analyze_stationarity(data_path, output_path):
    # 加载数据
    data = pd.read_excel(data_path)
    
    # 准备分析所有产品的平稳性
    all_stationarity_results = []
    
    # 遍历数据集中的每一行（每行对应一个产品的时间序列）
    for index, row in data.iterrows():
        product_series = row[1:]  # 跳过产品编号列
        result = check_stationarity(product_series)
        result['Product'] = row['product_no']  # 将产品编号包含在结果中
        all_stationarity_results.append(result)
    
    # 将结果转换为DataFrame，以便更容易查看和分析
    results_df = pd.DataFrame(all_stationarity_results)
    
    # 将结果保存到Excel文件
    results_df.to_excel(output_path, index=False)

# 示例用法
data_path = 'datasets/Timeserious.xlsx'  # 替换为你的Excel文件路径
output_path = 'output/Differential.xlsx'  # 期望的输出Excel文件路径
analyze_stationarity(data_path, output_path)


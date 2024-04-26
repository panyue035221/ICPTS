import pandas as pd
import numpy as np

def standardize_series(series):
    #标准化时间序列数据。
    
    #参数:
    #series: pd.Series, 时间序列数据。
    
    #返回:
    #standardized_series: pd.Series, 标准化后的时间序列数据。
    mean = series.mean()
    std = series.std()
    if std == 0:  # 如果标准差为0，返回一个全0的序列
        return pd.Series(np.zeros_like(series), index=series.index)
    else:
        standardized_series = (series - mean) / std
        return standardized_series

def extract_baseline(series, span=12, adjust=True):
    #使用指数加权移动平均（EWMA）提取时间序列的基线。
    
    #参数:
    # series: pd.Series, 时间序列数据。
    # span: int, EWMA的跨度，控制平滑的程度。
    # adjust: bool, 是否调整EWMA的开始处的偏差。
    
    #返回:
    #baseline: pd.Series, 时间序列的基线。
    baseline = series.ewm(span=span, adjust=adjust).mean()
    return baseline

def main():
    # 加载数据
    data_path = 'datasets/Afterdiserious.xlsx'  # Excel文件路径
    data = pd.read_excel(data_path)

    # 初始化DataFrame用于保存基线数据
    baseline_data = pd.DataFrame()

    # 处理每个产品的时间序列
    for index, row in data.iterrows():
        product_no = row['product_no']
        time_series = pd.Series(row[1:].values, index=row.index[1:])
        
        # 标准化时间序列
        standardized_series = standardize_series(time_series)
        
        # 提取基线
        baseline_series = extract_baseline(standardized_series)
        
        # 保存结果
        baseline_data = pd.concat([baseline_data, pd.DataFrame({product_no: baseline_series}).T])

    # 设置基线数据的列名为日期
    baseline_data.columns = data.columns[1:]

    # 输出文件路径
    output_path = 'output/Baseline_Series.xlsx'
    # 将包含基线的数据写入到新的Excel文件中
    baseline_data.to_excel(output_path)

    print("基线数据已保存至:", output_path)

if __name__ == "__main__":
    main()

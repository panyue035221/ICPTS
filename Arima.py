import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from itertools import product
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义计算 MAPE 的函数
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 定义计算 SMAPE 的函数
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

# 定义执行 ARIMA 模型的函数
def run_arima_model(data_path, output_dir='output'):
    # 从指定路径读取数据集
    data = pd.read_excel(data_path, index_col=0)
    data.columns = pd.to_datetime(data.columns)  # 确保列名是日期格式
    model_params = []
    performance_metrics = []

    # 遍历数据集中的每一行（每个商品）
    for index, row in data.iterrows():
        series = row.dropna()
        best_score, best_cfg, best_model_info = float("inf"), None, None

        # 为 ARIMA 模型生成 (p, d, q) 参数的所有组合
        p = d = q = range(0, 4)
        pdq = list(product(p, range(0, 3), q))

        # 对每组参数进行模型拟合，并通过 AIC 寻找最佳模型
        for param in pdq:
            try:
                model = ARIMA(series, order=param)
                results = model.fit()
                if results.aic < best_score:
                    best_score = results.aic
                    best_cfg = param
                    best_model_info = results
            except Exception as e:
                logging.warning(f"处理参数 {param} 时失败: {e}")
                continue

        if best_model_info:
            # 为最优模型计算性能指标
            predictions = best_model_info.predict(start=0, end=len(series) - 1)
            mse = mean_squared_error(series, predictions)
            mae = mean_absolute_error(series, predictions)
            mape = mean_absolute_percentage_error(series, predictions)
            smape = symmetric_mean_absolute_percentage_error(series, predictions)
            rsq = r2_score(series, predictions)

            model_params.append((index, best_cfg))
            performance_metrics.append((index, rsq, mae, mape, smape, mse))

    # 将模型参数和性能指标保存为 DataFrame
    df_params = pd.DataFrame(model_params, columns=['Product', 'Best_ARIMA_Params'])
    df_performance = pd.DataFrame(performance_metrics, columns=['Product', 'R_squared', 'MAE', 'MAPE', 'SMAPE', 'MSE'])
    
    params_path = f'{output_dir}/Arima_params.xlsx'
    performance_path = f'{output_dir}/Arima_performance.xlsx'
    df_params.to_excel(params_path, index=False)
    df_performance.to_excel(performance_path, index=False)

    return params_path, performance_path

# 调用函数，运行模型
params_path, performance_path = run_arima_model('datasets/Timeserious.xlsx')
print(f"模型参数保存至 {params_path}")
print(f"性能指标保存至 {performance_path}")

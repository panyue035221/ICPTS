import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score
from scipy.stats import entropy

# 加载数据
data_path = 'datasets/Labelserious.xlsx'  # 修改为实际的文件路径
data = pd.read_excel(data_path)

# 提取时间序列数据，假设前三列是标识信息
time_series_data = data.iloc[:, 3:].values

# 检查并移除全NaN或长度为零的时间序列
valid_indices = np.array([not np.all(np.isnan(ts)) and len(ts) > 0 for ts in time_series_data])
time_series_data = time_series_data[valid_indices]

# 使用插值方法填充NaN值
for i in range(time_series_data.shape[0]):
    valid_idx = ~np.isnan(time_series_data[i])
    if np.any(valid_idx):
        time_series_data[i][~valid_idx] = np.interp(np.flatnonzero(~valid_idx), np.flatnonzero(valid_idx), time_series_data[i][valid_idx])

# 标准化时间序列数据
scaler = TimeSeriesScalerMeanVariance()
time_series_scaled = scaler.fit_transform(time_series_data)

# 使用DTW和TimeSeriesKMeans进行聚类
n_clusters =  2  # 可根据需求调整聚类的数量
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True, max_iter=10, random_state=0)
labels = model.fit_predict(time_series_scaled)

# 计算评价指标
silhouette_avg = silhouette_score(time_series_scaled.reshape(time_series_scaled.shape[0], -1), labels)
ari = adjusted_rand_score(data['seller_no'].iloc[valid_indices], labels)
nmi = normalized_mutual_info_score(data['seller_no'].iloc[valid_indices], labels)
dbi = davies_bouldin_score(time_series_scaled.reshape(time_series_scaled.shape[0], -1), labels)
chi = calinski_harabasz_score(time_series_scaled.reshape(time_series_scaled.shape[0], -1), labels)
fm_index = fowlkes_mallows_score(data['seller_no'].iloc[valid_indices], labels)
cluster_entropy = entropy(labels)

# 将聚类结果和评价指标添加回原始DataFrame
data = data.iloc[valid_indices]  # 确保只使用有效数据
data['Cluster_Labels'] = labels
data['Silhouette Score'] = silhouette_avg
data['ARI'] = ari
data['NMI'] = nmi
data['DBI'] = dbi
data['CHI'] = chi
data['FM Index'] = fm_index
data['Entropy'] = cluster_entropy

# 输出聚类结果
print(data.head())

# 保存到新的Excel文件

output_file_path = f'output/DTW_results{n_clusters}.xlsx'
data.to_excel(output_file_path, index=False)
print("聚类结果和评价指标已保存到:", output_file_path)

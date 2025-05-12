import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist

# --- 1. 读取子集文件 ---
meta_path = r"C:\Users\ryan\Desktop\ADVANCED Master\data\meta_subset_DANG_top200rows.csv"
expr_path = r"C:\Users\ryan\Desktop\ADVANCED Master\data\expr_subset_DANG_top200rows.csv"

meta_df = pd.read_csv(meta_path, index_col=0)   # 索引 = cid
expr_df = pd.read_csv(expr_path, index_col=0)   # 索引 = rid

# --- 2. 构建原始 X（one-hot）和 y（expression） ---
# 2.1 one-hot
cmap_series = meta_df["cmap_name"]
X = pd.get_dummies(cmap_series)                 # 索引 = cid

# 2.2 expression 转置
y = expr_df.T                                   # 索引 = cid

# 2.3 对齐
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

# （可选）保存原始 X,y
X.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\X_DANG_raw.csv")
y.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_DANG_raw.csv")

print("原始 X shape:", X.shape)
print("原始 y shape:", y.shape)

# --- 3. 按 cmap_name 聚合：对 y 取平均 ---
# 3.1 先把标签重并到 y
y["cmap_name"] = cmap_series.loc[y.index]

# 3.2 分组取平均
y_agg = y.groupby("cmap_name").mean()

# 3.3 生成聚合后 one-hot X_agg
X_agg = pd.get_dummies(y_agg.index)
X_agg.index = y_agg.index

print("聚合后 X_agg shape:", X_agg.shape)
print("聚合后 y_agg shape:", y_agg.shape)

# --- 3.4 计算每组内部的样本差异程度（信息损失量化） ---
group_variability = {}

for name, group in y.groupby("cmap_name"):
    group_expr = group.drop(columns=["cmap_name"])
    
    if group_expr.shape[0] < 2:
        variability = 0.0
    else:
        pairwise_dists = pdist(group_expr.values, metric='correlation')
        variability = pairwise_dists.mean()
    
    group_variability[name] = variability

# 转为 DataFrame 并保存
variability_df = pd.DataFrame.from_dict(group_variability, orient="index", columns=["avg_pairwise_distance"])
variability_df.index.name = "cmap_name"
variability_df.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_group_variability.csv")
print("每组内部差异程度已计算并保存。")

# --- 3.5 可视化差异程度 ---
plt.figure(figsize=(12, 6))
variability_df_sorted = variability_df.sort_values(by="avg_pairwise_distance", ascending=False)
sns.barplot(x=variability_df_sorted.index, y="avg_pairwise_distance", data=variability_df_sorted)
plt.xticks(rotation=90)
plt.xlabel("cmap_name")
plt.ylabel("Avg. Pairwise Distance")
plt.title("Group Expression Variability (Information Loss after Aggregation)")
plt.tight_layout()
plt.savefig(r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_group_variability_plot.png")
plt.close()
print("可视化图已保存。")

# --- 4. 保存聚合后的结果 ---
X_agg.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\X_DANG_agg.csv")
y_agg.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_DANG_agg.csv")
print("聚合后的 X_agg 和 y_agg 已保存。")

# --- 5. 基因名编码 ---
gene_names = X_agg.idxmax(axis=1)
gene_ids, gene_vocab = pd.factorize(gene_names)

gene_id_df = pd.DataFrame({'gene_id': gene_ids}, index=X_agg.index)
gene_id_df.to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\gene_ids.csv")
pd.Series(gene_vocab).to_csv(r"C:\Users\ryan\Desktop\ADVANCED Master\data\gene_vocab.csv", index=False)

print("已生成整数 gene_id 文件，shape:", gene_id_df.shape)

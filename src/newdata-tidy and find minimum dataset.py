# -*- coding: utf-8 -*-
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

# --- 第 1 步：读取 CRISPR 的表达数据 GCTX 文件 ---
gctx_path = r"D:\大文件下载\level5_beta_trt_xpr_n142901x12328.gctx"
print(f"正在读取 GCTX 文件: {gctx_path} ...")
try:
    gctoo = parse(gctx_path)
    print("GCTX 文件读取完成。")
    crispr_sig_ids = gctoo.col_metadata_df.index.tolist()
    print(f"从 GCTX 文件中提取了 {len(crispr_sig_ids)} 个 sig_id。")
except FileNotFoundError:
    print(f"错误：GCTX 文件未找到 {gctx_path}")
    raise
except Exception as e:
    print(f"读取或解析 GCTX 文件时发生错误: {e}")
    raise

# --- 第 2 步：读取已筛选的元数据 CSV 文件 ---
raw_siginfo_path = r"D:\大文件下载\siginfo_beta.txt"
filtered_siginfo_path = r"D:\大文件下载\siginfo_trt_xpr_only.csv"
print(f"正在读取原始 siginfo 文件: {raw_siginfo_path} ...")
try:
    raw_siginfo_df = pd.read_csv(raw_siginfo_path, sep="\t")
    print("原始 siginfo 文件读取完成。")
    print(f"总共读取到 {len(raw_siginfo_df)} 条记录。")
except FileNotFoundError:
    print(f"错误：文件未找到 {raw_siginfo_path}")
    raise
except Exception as e:
    print(f"读取 CSV 文件时发生错误: {e}")
    raise

# 筛选出 pert_type 为 'trt_xpr' 的行
filtered_df = raw_siginfo_df[raw_siginfo_df["pert_type"] == "trt_xpr"].copy()
print(f"筛选出 pert_type 为 'trt_xpr' 的记录数量: {len(filtered_df)}")

# 保存为 CSV
filtered_df.to_csv(filtered_siginfo_path, index=False)
print(f"筛选结果已保存为: {filtered_siginfo_path}")

print(f"正在读取已过滤的元数据 CSV 文件: {filtered_siginfo_path} ...")
try:
    siginfo_df = pd.read_csv(filtered_siginfo_path, sep=",")
    print("已过滤的元数据 CSV 文件读取完成。")
    print(f"读取到的数据包含 {len(siginfo_df)} 行。")
except FileNotFoundError:
    print(f"错误：文件未找到 {filtered_siginfo_path}")
    raise
except Exception as e:
    print(f"读取 CSV 文件时发生错误: {e}")
    raise

# --- 第 3 步：筛选 sig_id 并与 GCTX 匹配 ---
if "sig_id" not in siginfo_df.columns:
    raise KeyError("错误：'sig_id' 列不存在于元数据中。")

siginfo_df["sig_id"] = siginfo_df["sig_id"].astype(str)
print("'sig_id' 列已确保为字符串类型。")

# 保留与 GCTX 文件中 sig_id 匹配的元数据
crispr_meta = siginfo_df[siginfo_df["sig_id"].isin(crispr_sig_ids)].copy()
print(f"过滤后，保留 {len(crispr_meta)} 行元数据与 GCTX 文件匹配。")

# 设置索引为 sig_id
crispr_meta.set_index("sig_id", inplace=True)

# --- 第 4 步：合并元数据到 GCTX 对象的 col_metadata_df ---
print("正在将筛选后的元数据合并到 GCTX 对象的 col_metadata_df 中...")
gctoo.col_metadata_df = gctoo.col_metadata_df.join(crispr_meta, how="left")
print("元数据合并完成。")

# --- 第 5 步：清洗并对齐表达数据和元数据 ---
expr_df = gctoo.data_df
meta_df = gctoo.col_metadata_df

# 去除表达数据中没有匹配元数据的列，确保二者一致
expr_df = expr_df.loc[:, expr_df.columns.isin(meta_df.index)]
meta_df = meta_df.loc[expr_df.columns]

# 检查并断言一致性
assert list(expr_df.columns) == list(meta_df.index), "表达数据和元数据未成功对齐"

# 更新 GCTX 对象
gctoo.data_df = expr_df
gctoo.col_metadata_df = meta_df

# --- 最终报告 ---
print("\n--- ✅ 分析准备就绪 ---")
print(f"表达数据维度: {expr_df.shape} (基因 x 样本)")
print(f"元数据维度: {meta_df.shape} (样本 x 元数据字段)")
print(f"合并后的元数据列: {meta_df.columns.tolist()}")

# --- 选择细胞系 cell_iname ---
selected_cell = "A549"
print(f"\n选定的 cell_iname 为: {selected_cell}")

# --- 获取该 cell_iname 对应的所有 sig_id (cid) ---
selected_cids = meta_df[meta_df["cell_iname"] == selected_cell].index.tolist()
print(f"该 cell_iname 共包含 {len(selected_cids)} 个样本。")

# --- 提取表达数据子集：只选取 expr_df 中包含这些 cid 的列 ---
expr_subset = expr_df[selected_cids]


# --- 提取元数据子集：对应选定的 cids，列没变
meta_subset = meta_df.loc[selected_cids]

# --- 保存到文件（指定绝对路径）---
output_expr_path = r"C:\Users\ryan\Desktop\ADVANCED Master\data\expr_subset_DANG_top200rows.csv"
output_meta_path = r"C:\Users\ryan\Desktop\ADVANCED Master\data\meta_subset_DANG_top200rows.csv"

expr_subset.to_csv(output_expr_path)
meta_subset.to_csv(output_meta_path)

print("\n✅ 表达数据子集和元数据子集已保存：")
print(f"表达数据子集维度: {expr_subset.shape}")
print(f"元数据子集维度: {meta_subset.shape}")
print(f"文件保存为: {output_expr_path} 和 {output_meta_path}")

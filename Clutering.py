import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. Build feature matrix
# =========================
def build_features(df: pd.DataFrame):
    cat_cols = [c for c in ["Category","Style","Size"] if c in df.columns]
    num_cols = [c for c in ["Amount"] if c in df.columns]
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    X = ct.fit_transform(df[cat_cols + num_cols])
    return X, ct

# =========================
# 2. Train MiniBatchKMeans
# =========================
def train_cluster(df: pd.DataFrame, n_clusters=2, batch_size=1024, random_state=42):
    df = df.drop_duplicates(subset=["SKU"]).reset_index(drop=True)
    X, ct = build_features(df)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    labels = kmeans.fit_predict(X)
    df["Cluster"] = labels
    sil_score = silhouette_score(X, labels) if n_clusters > 1 else np.nan
    dbi_score = davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan
    return df, X, kmeans, ct, sil_score, dbi_score

# =========================
# 3. Recommend SKU using clustering
# =========================
def recommend_by_sku_cluster(sku, df, X, top_n=5, price_tol=10.0):
    row = df[df["SKU"] == sku]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    cluster = row["Cluster"]
    ref_amount = row["Amount"]

    # Candidate cùng cluster, khác SKU
    cand = df[(df["Cluster"] == cluster) & (df["SKU"] != sku)].copy()
    if cand.empty:
        return pd.DataFrame()

    # Lọc theo Category, Size, Core
    cand = cand[
        (cand["Category"] == row["Category"]) &
        (cand["Size"] == row["Size"]) &
        (cand["Core"] == row["Core"])
    ]
    if cand.empty:
        return pd.DataFrame()

    # Lọc ± Amount
    cand = cand[(cand["Amount"] >= ref_amount - price_tol) & (cand["Amount"] <= ref_amount + price_tol)]
    if cand.empty:
        return pd.DataFrame()

    # Cosine similarity
    ref_vec = X[row.name:row.name+1]
    cand_vecs = X[cand.index]
    cand["cosine_sim"] = cosine_similarity(ref_vec, cand_vecs).ravel()
    cand["amount_diff"] = abs(cand["Amount"] - ref_amount)

    # Sắp xếp
    # cand = cand.sort_values(by=["cosine_sim","amount_diff"], ascending=[False, True])
    cand = cand.sort_values(by=["amount_diff", "cosine_sim"], ascending=[True, False])


    return cand.head(top_n)[["SKU","Category","Core","Style","Size","Amount","Cluster","cosine_sim","amount_diff"]]

# =========================
# 4. Đánh giá 1 testcase
# =========================
def evaluate_single_testcase(file_path, df, X, top_n=5, price_tol=10.0):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return {
            "file": os.path.basename(file_path),
            "sku_main": None,
            "old_skus": [],
            "new_skus": [],
            "passed": False
        }

    # SKU chính
    sku_main = lines[0].split("(")[1].split(")")[0]

    # SKU gợi ý cũ
    old_skus = []
    for line in lines:
        if line.startswith("- "):
            sku = line.split("|")[0].strip()[2:]
            old_skus.append(sku)

    # SKU gợi ý mới từ clustering
    new_recs_df = recommend_by_sku_cluster(sku_main, df, X, top_n=top_n, price_tol=price_tol)
    new_skus = new_recs_df["SKU"].tolist() if not new_recs_df.empty else []

    testcase_passed = set(old_skus) == set(new_skus)

    return {
        "file": os.path.basename(file_path),
        "sku_main": sku_main,
        "old_skus": old_skus,
        "new_skus": new_skus,
        "passed": testcase_passed
    }

# =========================
# 5. Đánh giá tất cả testcase
# =========================
def evaluate_all_testcases(df, X, output_dir, top_n=5, price_tol=10.0):
    files = sorted([f for f in Path(output_dir).glob("*.txt")])
    results = []

    for f in files:
        result = evaluate_single_testcase(f, df, X, top_n=top_n, price_tol=price_tol)
        results.append(result)

    results_df = pd.DataFrame(results)
    total_cases = len(results_df)
    passed_cases = results_df["passed"].sum() if total_cases > 0 else 0
    accuracy = passed_cases / total_cases * 100 if total_cases > 0 else 0

    print(f"\nTổng testcase: {total_cases}")
    print(f"Số testcase đúng: {passed_cases}/{total_cases}")
    print(f"Tỷ lệ chính xác: {accuracy:.2f}%")

    # Lưu kết quả
    results_df.to_csv("cluster_testcase_results.csv", index=False)
    print("✔ Đã lưu kết quả chi tiết vào cluster_testcase_results.csv")
    return results_df

# =========================
# MAIN
# =========================
DATA_PATH = "new_data_to_analysis.csv"
OUTPUT_DIR = "output_total"  # thư mục chứa các file output cũ

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df, X, kmeans, ct, sil_score, dbi_score = train_cluster(df, n_clusters=4)
    print(f"✅ Clustering xong | Silhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")

    # Đánh giá tất cả testcase
    evaluate_all_testcases(df, X, OUTPUT_DIR, top_n=5, price_tol=10.0)
    print("Hoàn tất đánh giá tất cả testcase bằng clustering.")

# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     df = pd.read_csv(DATA_PATH)
    
#     # Build feature matrix
#     X, ct = build_features(df)

#     # Danh sách lưu inertia
#     inertia = []
#     ks = range(2, 11)

#     for k in ks:
#         # MiniBatchKMeans
#         kmeans = MiniBatchKMeans(
#             n_clusters=k,
#             batch_size=1024,      # kích thước batch, có thể chỉnh lớn hơn hoặc nhỏ hơn
#             random_state=42,
#             max_iter=300,
#             n_init=10             # số lần khởi tạo, thường nhỏ hơn KMeans chuẩn
#         )
#         kmeans.fit(X)             # chỉ fit, không cần lấy labels ngay
#         inertia.append(kmeans.inertia_)

#     # Vẽ Elbow plot
#     plt.figure(figsize=(8,5))
#     plt.plot(ks, inertia, marker='o', linestyle='-')
#     plt.title('Elbow Method – Xác định số cụm K tối ưu (MiniBatchKMeans)')
#     plt.xlabel('Số cụm (k)')
#     plt.ylabel('Inertia')
#     plt.grid(True)
#     plt.show()
   
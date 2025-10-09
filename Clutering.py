import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans

# === 1. Đọc dữ liệu ===
df = pd.read_csv("new_data_to_analysis.csv")
df = df.drop_duplicates(subset=["SKU"], keep="first")

# === 2. Encode các cột chuỗi ===
cols = ["Core", "Size", "Category"]
df_encoded = df.copy()
for col in cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

# === 3. Gom cụm bằng MiniBatchKMeans ===
X = df_encoded[["Amount", "Core", "Size", "Category"]]
X_scaled = StandardScaler().fit_transform(X)

kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=1024)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# === 4. Hàm gợi ý ===
def recommend_by_sku_cluster(sku, df, top_n=5):
    product = df[df["SKU"] == sku]
    if product.empty:
        print("⚠️ SKU không tồn tại.")
        return pd.DataFrame()
    
    cluster = product["Cluster"].iloc[0]
    amount = product["Amount"].iloc[0]
    size = product["Size"].iloc[0]
    core = product["Core"].iloc[0]
    category = product["Category"].iloc[0]

    # --- cùng cụm ---
    same_cluster = df[(df["Cluster"] == cluster) & (df["SKU"] != sku)].copy()

    # --- tính chênh lệch giá ---
    same_cluster["price_diff"] = abs(same_cluster["Amount"] - amount)

    # --- lọc ưu tiên cùng core, size, category ---
    same_cluster = same_cluster[
        (same_cluster["Core"] == core) &
        (same_cluster["Size"] == size) &
        (same_cluster["Category"] == category)
    ].sort_values("price_diff")

    return same_cluster.head(top_n)[["SKU", "Category", "Style", "Size", "Core", "Amount", "Cluster", "price_diff"]]

# === 5. Test thử ===
sku_input = "JNE2270-KR-487-A-M"
print(f"=== GỢI Ý CHO {sku_input} ===")
print(recommend_by_sku_cluster(sku_input, df))

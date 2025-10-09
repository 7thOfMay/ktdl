import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# === 1. Đọc dữ liệu ===
df = pd.read_csv("new_data_to_analysis.csv")

# === 2. Các cột dùng cho Apriori ===
cols = ["Category", "Style", "Size", "Core", "price_level"]
df_hot = pd.get_dummies(df[cols])

# === 3. Sinh tập phổ biến & luật kết hợp ===
frequent_itemsets = apriori(df_hot, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules.sort_values(by="lift", ascending=False)

print("=== LUẬT KẾT HỢP (tóm tắt) ===")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head())

# === 4. Hàm gợi ý theo SKU (ưu tiên cùng size) ===
def recommend_by_sku(sku, df, delta=10, top_n=5):
    """Gợi ý sản phẩm tương tự theo SKU, ưu tiên cùng Size."""
    product = df[df["SKU"] == sku]
    if product.empty:
        print("⚠️ SKU không tồn tại.")
        return pd.DataFrame()
    
    core = product["Core"].iloc[0]
    price_level = product["price_level"].iloc[0]
    size = product["Size"].iloc[0]
    amount = product["Amount"].iloc[0]
    category = product["Category"].iloc[0]

    print(amount, core, category)

    # --- B1: Gợi ý cùng Core, price_level, cùng Size ---
    mask_same_size = (
        (df["Core"] == core) &
        (df["price_level"] == price_level) &
        (df["Size"] == size) &
        (abs(df["Amount"] - amount) <= delta) &
        (df["SKU"] != sku)
    )
    recs_same_size = df[mask_same_size]

    # --- B2: Nếu chưa đủ, thêm sản phẩm cùng Core, price_level nhưng khác Size ---
    mask_diff_size = (
        (df["Core"] == core) &
        (df["price_level"] == price_level) &
        (df["Size"] != size) &
        (abs(df["Amount"] - amount) <= delta) &
        (df["SKU"] != sku)
    )
    recs_diff_size = df[mask_diff_size]

    # --- B3: Gộp kết quả, ưu tiên cùng size ---
    recs_same_size = recs_same_size.sort_values(by="Amount")
    recs_diff_size = recs_diff_size.sort_values(by="Amount")
    recs = pd.concat([recs_same_size, recs_diff_size])
    return recs.head(top_n)[["SKU", "Category", "Style", "Size", "Amount"]]

# === 5. Ví dụ test ===
sku_input = "JNE2270-KR-487-A-M"
print(f"\n=== GỢI Ý CHO SẢN PHẨM {sku_input} ===")
recommendations = recommend_by_sku(sku_input, df)
print(recommendations)

# apriori_recommend_fixed.py
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# =========================
# 1. Sinh rules Apriori
# =========================
def generate_apriori_rules(csv_path, min_support=0.1, min_lift=1):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="SKU", keep="first")

    cols = ["Category", "Style", "Size", "Core", "price_level"]
    df_hot = pd.get_dummies(df[cols])
    
    frequent_itemsets = apriori(df_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values(by="lift", ascending=False)
    return df, rules


# =========================
# 2. Recommend for a SKU
# =========================
def recommend_by_sku(sku, df, rules=None, top_n=5, price_tol=10.0):
    product = df[df["SKU"] == sku]
    if product.empty:
        print(f"SKU {sku} không tồn tại.")
        return pd.DataFrame()
    
    product = product.iloc[0]

    category = product["Category"]
    core = product["Core"]
    size = product["Size"]
    price_level = product["price_level"]
    amount = product["Amount"]

    candidates = df[
        (df["SKU"] != sku) &
        (df["Size"] == size) &
        (df['Core'] == core) &
        (df['Category'] == category)
        ].copy()

    candidates["similarity_score"] = (
        (candidates["Category"] == category).astype(int) +
        (candidates["Core"] == core).astype(int) +
        (candidates["Size"] == size).astype(int) +
        (candidates["price_level"] == price_level).astype(int)
    )

    candidates = candidates[candidates["similarity_score"] >= 2]
    
    # --- TÍNH CHÊNH LỆCH GIÁ ---
    candidates["amount_diff"] = abs(candidates["Amount"] - amount)
    
    # --- LỌC CHÊNH LỆCH GIÁ THEO THAM SỐ ---
    candidates = candidates[candidates["amount_diff"] <= price_tol]
    
    candidates = candidates.drop_duplicates(subset="SKU")

    candidates = candidates.sort_values(
        by=["similarity_score", "amount_diff"],
        ascending=[False, True]
    )

    return candidates.head(top_n)[
        ["SKU","Category","Core","Size","Amount","similarity_score","amount_diff"]
    ]



# =========================
# 3. Ghi file output test
# =========================
def write_output(file_path, sku, product_info, recommendations):
    """Ghi file txt"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"=== GỢI Ý SẢN PHẨM CHO SKU: {sku} ===\n\n")
        f.write(">>> THÔNG TIN SẢN PHẨM GỐC:\n")
        f.write(str(product_info) + "\n\n")

        f.write(">>> TOP GỢI Ý (Apriori + Similarity):\n")
        if recommendations.empty:
            f.write("KHÔNG CÓ GỢI Ý\n")
            return

        for idx, row in recommendations.iterrows():
            f.write(
                f"- {row['SKU']} | core={row['Core']} | size={row['Size']} "
                f"| amount={row['Amount']} | score={row['similarity_score']} "
                f"| diff={row['amount_diff']}\n"
            )

def save_apriori_rules(rules_df, filename="output_apriori/apriori_rules.csv"):
    # Chuyển antecedents và consequents thành dạng list/string
    rules_export = rules_df.copy()
    rules_export["antecedents"] = rules_export["antecedents"].apply(lambda x: ','.join(list(x)) if isinstance(x, frozenset) else str(x))
    rules_export["consequents"] = rules_export["consequents"].apply(lambda x: ','.join(list(x)) if isinstance(x, frozenset) else str(x))

    # Lưu file CSV
    rules_export.to_csv(filename, index=False, encoding="utf-8")
    print(f"✔ Đã lưu {len(rules_export)} luật Apriori vào {filename}")

# =========================
# 4. MAIN — chạy test 100 SKU
# =========================
if __name__ == "__main__":
    csv_path = "new_data_to_analysis.csv"

    print("🔍 Đang chạy Apriori attribute-based…")
    df, rules = generate_apriori_rules(csv_path)

    # --- Lưu toàn bộ luật Apriori ra file CSV ---
    os.makedirs("outputs_apriori", exist_ok=True)
    save_apriori_rules(rules, "output_apriori/apriori_rules.csv")

    print("➡ Lấy 100 SKU đầu tiên trong dataset để test…")
    sku_list = df["SKU"].unique()[:100]

    os.makedirs("output_apriori", exist_ok=True)

    for idx, sku in enumerate(sku_list, start=1):
        product_info = df[df["SKU"] == sku].iloc[0]

        recs = recommend_by_sku(sku, df, rules, top_n=5)

        output_path = f"output_apriori/output_{idx}.txt"
        write_output(output_path, sku, product_info, recs)

        print(f"✔ File {output_path} đã tạo xong cho SKU {sku}")

    print("\n🎉 HOÀN TẤT! ĐÃ TẠO 100 FILE TRONG THƯ MỤC output_apriori/")
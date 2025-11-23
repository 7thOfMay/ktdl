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
# 2. Gợi ý sản phẩm cho SKU
# =========================
def recommend_by_sku(sku, df, rules=None, top_n=5, price_tol=10.0):
    product = df[df["SKU"] == sku]
    if product.empty:
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
    candidates["amount_diff"] = abs(candidates["Amount"] - amount)
    candidates = candidates[candidates["amount_diff"] <= price_tol]
    candidates = candidates.drop_duplicates(subset="SKU")
    candidates = candidates.sort_values(by=["similarity_score","amount_diff"], ascending=[False, True])

    return candidates.head(top_n)[
        ["SKU","Category","Core","Size","Amount","similarity_score","amount_diff"]
    ]

# =========================
# 3. Đánh giá 1 testcase
# =========================
def evaluate_single_testcase(file_path, df, rules=None, top_n=5, price_tol=10.0):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # SKU chính
    sku_main = lines[0].split("(")[1].split(")")[0]

    # SKU gợi ý cũ
    old_skus = []
    for line in lines:
        if line.startswith("- "):
            sku = line.split("|")[0].strip()[2:]
            old_skus.append(sku)

    # SKU gợi ý mới từ Apriori
    new_recs_df = recommend_by_sku(sku_main, df, rules=rules, top_n=top_n, price_tol=price_tol)
    new_skus = new_recs_df["SKU"].tolist()

    # So sánh: tất cả SKU cũ có trùng hết với gợi ý mới => đúng
    testcase_passed = set(old_skus) == set(new_skus)

    return {
        "file": os.path.basename(file_path),
        "sku_main": sku_main,
        "old_skus": old_skus,
        "new_skus": new_skus,
        "passed": testcase_passed
    }

# =========================
# 4. Đánh giá tất cả testcase trong folder
# =========================
def evaluate_all_testcases(df, output_dir, rules=None, top_n=5, price_tol=10.0):
    files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
    files.sort()
    results = []

    for f in files:
        file_path = os.path.join(output_dir, f)
        result = evaluate_single_testcase(file_path, df, rules=rules, top_n=top_n, price_tol=price_tol)
        results.append(result)
        # status = "PASSED ✅" if result["passed"] else "FAILED ❌"
        # print(f"{f}: {status}")

    results_df = pd.DataFrame(results)
    total_cases = len(results_df)
    passed_cases = results_df["passed"].sum()
    accuracy = passed_cases / total_cases * 100 if total_cases > 0 else 0

    print(f"\nTổng testcase: {total_cases}")
    print(f"Số testcase đúng: {passed_cases}/{total_cases}")
    print(f"Tỷ lệ chính xác: {accuracy:.2f}%")

    # Lưu kết quả chi tiết
    results_df.to_csv("apriori_testcase_results.csv", index=False)
    print("✔ Đã lưu kết quả chi tiết vào apriori_testcase_results.csv")
    return results_df

# =========================
# MAIN
# =========================
DATA_PATH = "new_data_to_analysis.csv"
OUTPUT_DIR = "output_total"  # thư mục chứa file output cũ

# 1. Sinh rules Apriori
df, rules = generate_apriori_rules(DATA_PATH, min_support=0.1, min_lift=1)

# 2. Đánh giá tất cả testcase
results_df = evaluate_all_testcases(df, OUTPUT_DIR, rules=rules, top_n=5, price_tol=10.0)

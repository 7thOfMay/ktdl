import pandas as pd
import os

# ============================
# 1. Đọc dataset gốc
# ============================
df = pd.read_csv("new_data_to_analysis.csv")

# === BƯỚC 0: LOẠI BỎ TRÙNG SKU TRÊN TOÀN DATASET ===
df = df.drop_duplicates(subset="SKU", keep="first")

# Tạo thư mục output nếu chưa có
os.makedirs("outputs", exist_ok=True)

# Lấy 100 SKU đầu tiên (hoặc ít hơn nếu dataset nhỏ)
top_100 = df.head(100)

# ============================
# 2. HÀM LỌC SẢN PHẨM GỢI Ý
# ============================

def get_recommendations(row, df_full):
    category = row["Category"]
    size = row["Size"]
    core = row["Core"]
    price_level = row["price_level"]
    style = row["Style"]
    amount = row["Amount"]

    # --- Lọc sản phẩm có cùng Category, Size, Core, Pricelevel ---
    filtered = df_full[
        (df_full["Category"] == category) &
        (df_full["Size"] == size) &
        (df_full["Core"] == core) &
        (df_full["price_level"] == price_level) &
        (df_full["Style"] != style)              # khác style
    ]

    # Loại trùng SKU trong bộ lọc
    filtered = filtered.drop_duplicates(subset="SKU", keep="first")

    # =============================
    # ƯU TIÊN 1: Lấy đúng giá
    # =============================
    same_price = filtered[filtered["Amount"] == amount]

    # Nếu đủ 5 thì trả luôn
    if len(same_price) >= 5:
        return same_price.head(5)

    # =============================
    # ƯU TIÊN 2: Lấy thêm +-10
    # =============================
    around_price = filtered[
        (filtered["Amount"] >= amount - 10) &
        (filtered["Amount"] <= amount + 10)
    ]

    # Gộp hai tập
    merged = pd.concat([same_price, around_price]).drop_duplicates(subset="SKU")

    return merged.head(5)


# ============================
# 3. TẠO 100 FILE OUTPUT
# ============================

for idx, (_, row) in enumerate(top_100.iterrows(), start=1):
    recs = get_recommendations(row, df)

    file_path = f"outputs/output_{idx}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"=== PRODUCT #{idx} ({row['SKU']}) ===\n")
        f.write(f"Category: {row['Category']}\n")
        f.write(f"Size: {row['Size']}\n")
        f.write(f"Core: {row['Core']}\n")
        f.write(f"Price Level: {row['price_level']}\n")
        f.write(f"Style: {row['Style']}\n")
        f.write(f"Amount: {row['Amount']}\n\n")

        f.write("=== 5 GỢI Ý SẢN PHẨM ===\n")

        if len(recs) == 0:
            f.write("Không tìm thấy gợi ý phù hợp.\n")
        else:
            for _, r in recs.iterrows():
                f.write(f"- {r['SKU']} | {r['Style']} | {r['Amount']} VND\n")

    print("Created:", file_path)

print("✔ DONE! Đã tạo xong 100 file output trong thư mục /outputs/")

import os

# =========================
# 1. Hàm đọc SKU từ file output
# =========================
def read_skus(filepath):
    """Đọc danh sách SKU từ file output, bỏ qua các dòng header"""
    if not os.path.exists(filepath):
        return None
    skus = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                # SKU luôn là token thứ 2 bất kể định dạng khác nhau
                sku = line.split()[1]
                skus.append(sku)
    return skus

# =========================
# 2. Đánh giá một folder output
# =========================
def evaluate_folder(folder, label):
    correct = 0
    wrong = 0
    missing = 0
    print(f"\n=== ĐANG KIỂM TRA {label} ===")

    for i in range(1, 101):
        gt_skus = read_skus(f"outputs/output_{i}.txt")  # ground truth
        pred_skus = read_skus(f"{folder}/output_{i}.txt")

        if gt_skus is None:
            print(f"⚠️ Thiếu file ground truth: outputs/output_{i}.txt")
            continue
        if pred_skus is None:
            print(f"⚠️ Thiếu file: {folder}/output_{i}.txt")
            missing += 1
            continue

        if set(gt_skus) == set(pred_skus):
            correct += 1
        else:
            wrong += 1
            print(f"\n❌ Sai tại {folder}/output_{i}.txt")
            print("EXPECTED:", gt_skus)
            print("PREDICTED:", pred_skus)

    print(f"\n---- KẾT QUẢ {label} ----")
    print("✔ Đúng:", correct)
    print("❌ Sai:", wrong)
    print("⚠️ Thiếu file:", missing)
    print("--------------------------")


# =========================
# 3. Chạy kiểm tra cho cả Apriori và Clustering
# =========================
if __name__ == "__main__":
    # So sánh với Apriori
    evaluate_folder("output_apriori", "APRIORI")

    # So sánh với Clustering
    # evaluate_folder("outputs_cluster", "CLUSTERING")

    print("\n🎉 HOÀN TẤT SO SÁNH SKU")

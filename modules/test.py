import pandas as pd
import matplotlib.pyplot as plt

def visualize_data_info(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1️⃣ Số lượng NaN theo cột
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]  # Chỉ hiển thị cột có NaN
    if not nan_counts.empty:
        axes[0, 0].bar(nan_counts.index, nan_counts.values, color='red')
        axes[0, 0].set_title("Số lượng giá trị NaN theo cột")
        axes[0, 0].tick_params(axis='x', rotation=45)

    # 2️⃣ Phân phối dữ liệu số
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(ax=axes[0, 1], bins=20, edgecolor='black', grid=False)
        axes[0, 1].set_title("Phân phối dữ liệu số")

    # 3️⃣ Thống kê dữ liệu phân loại (chọn cột đầu tiên nếu có)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        unique_vals, counts = zip(*df[categorical_cols[0]].value_counts().items())
        axes[1, 0].bar(unique_vals, counts, color='blue')
        axes[1, 0].set_title(f"Phân bố của {categorical_cols[0]}")
        axes[1, 0].tick_params(axis='x', rotation=45)

    # 4️⃣ Pie Chart: Tỉ lệ dữ liệu thiếu
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    labels = ['Có dữ liệu', 'Thiếu dữ liệu']
    sizes = [1 - missing_ratio, missing_ratio]
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    axes[1, 1].set_title("Tỉ lệ dữ liệu thiếu")

    plt.tight_layout()
    plt.show()

# 📌 Test với dữ liệu giả lập
df_sample = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': ['a', 'b', 'c', 'a', 'b'],
    'C': [10, 20, 30, 40, None]
})

visualize_data_info(df_sample)

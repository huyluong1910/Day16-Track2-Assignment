import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import lightgbm as lgb

def main():
    print("=" * 60)
    print("      LAB 16 - ML BENCHMARK WITH LIGHTGBM (CPU)")
    print("=" * 60)

    csv_path = "creditcard.csv"
    
    # 1. Đo thời gian Load Data
    print("\n[1/5] Đang đọc dữ liệu từ dataset...", flush=True)
    t_load_start = time.time()
    df = pd.read_csv(csv_path)
    load_time_sec = time.time() - t_load_start
    print(f"-> Hoàn tất load data trong: {load_time_sec:.4f} giây")
    print(f"-> Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

    # 2. Tiền xử lý & Chia tập Train/Test
    print("\n[2/5] Phân chia tập Train/Test...", flush=True)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Train set: {X_train.shape[0]} dòng | Test set: {X_test.shape[0]} dòng")

    # 3. Huấn luyện mô hình LightGBM
    print("\n[3/5] Bắt đầu huấn luyện LightGBM...", flush=True)
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    t_train_start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    training_time_sec = time.time() - t_train_start
    best_iteration = int(model.best_iteration_) if model.best_iteration_ is not None else 100
    print(f"-> Training hoàn thành trong: {training_time_sec:.4f} giây")
    print(f"-> Best iteration: {best_iteration}")

    # 4. Đánh giá chất lượng mô hình trên tập Test
    print("\n[4/5] Đánh giá các chỉ số mô hình trên tập Test...", flush=True)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_pred_proba))
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))

    # 5. Đo Inference Latency (1 dòng) & Throughput (1000 dòng)
    print("\n[5/5] Đo Inference Benchmark (Latency & Throughput)...", flush=True)
    sample_single = X_test.iloc[[0]]
    # Warm-up
    for _ in range(10):
        _ = model.predict_proba(sample_single)

    # Đo latency 1 dòng (lấy trung bình qua 100 lần lặp)
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict_proba(sample_single)
        latencies.append(time.perf_counter() - t0)
    avg_latency_ms = float(np.mean(latencies) * 1000)

    # Đo throughput 1000 dòng
    sample_1000 = X_test.iloc[:1000]
    t0_tp = time.perf_counter()
    _ = model.predict_proba(sample_1000)
    time_1000_sec = time.perf_counter() - t0_tp
    throughput_rps = float(1000.0 / time_1000_sec)

    # Kết quả
    results = {
        "load_time_sec": round(load_time_sec, 4),
        "training_time_sec": round(training_time_sec, 4),
        "best_iteration": best_iteration,
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(acc, 6),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "inference_latency_1_row_ms": round(avg_latency_ms, 3),
        "inference_throughput_1000_rows_per_sec": round(throughput_rps, 1),
        "time_1000_rows_ms": round(time_1000_sec * 1000, 2)
    }

    # In bảng ra terminal
    print("\n" + "=" * 60)
    print("                    BẢNG KẾT QUẢ BENCHMARK")
    print("=" * 60)
    print(f"{'Metric':<35} | {'Kết quả':<20}")
    print("-" * 60)
    print(f"{'Thời gian load data':<35} | {results['load_time_sec']} s")
    print(f"{'Thời gian training':<35} | {results['training_time_sec']} s")
    print(f"{'Best iteration':<35} | {results['best_iteration']}")
    print(f"{'AUC-ROC':<35} | {results['auc_roc']}")
    print(f"{'Accuracy':<35} | {results['accuracy'] * 100:.2f}%")
    print(f"{'F1-Score':<35} | {results['f1_score']}")
    print(f"{'Precision':<35} | {results['precision']}")
    print(f"{'Recall':<35} | {results['recall']}")
    print(f"{'Inference latency (1 row)':<35} | {results['inference_latency_1_row_ms']} ms")
    print(f"{'Inference throughput (1000 rows)':<35} | {results['inference_throughput_1000_rows_per_sec']} rows/s ({results['time_1000_rows_ms']} ms)")
    print("=" * 60)

    # Lưu ra benchmark_result.json
    output_file = "benchmark_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Đã lưu kết quả chi tiết vào file: {output_file}\n")

if __name__ == "__main__":
    main()

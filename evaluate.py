"""Threshold ve pencere ayarlari icin otomatik degerlendirme scripti."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoder karar ayarlari icin grid degerlendirme.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifact klasoru")
    parser.add_argument("--model-path", type=str, default="model.h5", help="Egitilen model dosyasi")
    parser.add_argument("--target-recall", type=float, default=0.80, help="Secim icin hedef recall degeri")
    parser.add_argument("--max-rows", type=int, default=15, help="Konsolda yazdirilacak satir sayisi")
    return parser.parse_args()


def reconstruction_errors(model: tf.keras.Model, samples: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(samples, verbose=0)
    return np.mean(np.square(samples - reconstructed), axis=1)


def apply_window_filter(raw_predictions: np.ndarray, window_size: int, min_votes: int) -> np.ndarray:
    output = np.zeros_like(raw_predictions)
    for idx in range(len(raw_predictions)):
        start = max(0, idx - window_size + 1)
        votes = int(np.sum(raw_predictions[start : idx + 1]))
        output[idx] = int(votes >= min_votes)
    return output


def evaluate_configs(
    y_true: np.ndarray,
    mse_values: np.ndarray,
    base_threshold: float,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    threshold_scales = np.round(np.arange(0.80, 1.21, 0.05), 2)
    window_sizes = [1, 3, 5, 7]

    for scale in threshold_scales:
        active_threshold = base_threshold * float(scale)
        raw_predictions = (mse_values > active_threshold).astype(int)

        for window_size in window_sizes:
            for min_votes in range(1, window_size + 1):
                y_pred = apply_window_filter(raw_predictions, window_size, min_votes)
                records.append(
                    {
                        "threshold_scale": float(scale),
                        "active_threshold": float(active_threshold),
                        "window_size": int(window_size),
                        "min_votes": int(min_votes),
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    }
                )

    return pd.DataFrame(records)


def select_best_config(results: pd.DataFrame, target_recall: float) -> pd.Series:
    eligible = results[results["recall"] >= target_recall]
    if not eligible.empty:
        return eligible.sort_values(by=["f1", "precision"], ascending=False).iloc[0]
    return results.sort_values(by=["recall", "f1"], ascending=False).iloc[0]


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_path = Path(args.model_path)

    test_set_path = artifacts_dir / "test_set.csv"
    threshold_path = artifacts_dir / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")
    if not test_set_path.exists():
        raise FileNotFoundError(f"Test seti bulunamadi: {test_set_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold dosyasi bulunamadi: {threshold_path}")

    with threshold_path.open("r", encoding="utf-8") as file:
        threshold_metadata = json.load(file)
    base_threshold = float(threshold_metadata["threshold"])

    test_frame = pd.read_csv(test_set_path)
    y_true = test_frame["binary_label"].to_numpy(dtype=np.int32)
    x_test = test_frame.drop(columns=["binary_label"]).to_numpy(dtype=np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)
    mse_values = reconstruction_errors(model, x_test)

    results = evaluate_configs(y_true=y_true, mse_values=mse_values, base_threshold=base_threshold)
    best = select_best_config(results, target_recall=args.target_recall)

    results_sorted = results.sort_values(by=["f1", "recall", "precision"], ascending=False)
    print("\nEn iyi kombinasyonlar:")
    print(results_sorted.head(args.max_rows).to_string(index=False))

    best_config = {
        "selection_policy": "recall>=target ise en iyi f1, aksi halde en yuksek recall",
        "target_recall": args.target_recall,
        "base_threshold": base_threshold,
        "best_threshold_scale": float(best["threshold_scale"]),
        "best_active_threshold": float(best["active_threshold"]),
        "best_window_size": int(best["window_size"]),
        "best_min_votes": int(best["min_votes"]),
        "best_accuracy": float(best["accuracy"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
    }

    output_csv = artifacts_dir / "evaluation_results.csv"
    output_json = artifacts_dir / "best_config.json"
    results.to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(best_config, file, ensure_ascii=False, indent=2)

    print("\nSecilen en iyi konfig:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
    print(f"\nKaydedilen dosyalar: {output_csv} ve {output_json}")


if __name__ == "__main__":
    main()

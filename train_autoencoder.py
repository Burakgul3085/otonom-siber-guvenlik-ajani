"""Sifirdan kurulan dense autoencoder egitimi ve threshold hesaplama."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


@dataclass
class AutoencoderTrainer:
    """Autoencoder modelini egitir, threshold hesaplar ve modeli kaydeder."""

    artifacts_dir: Path = Path("artifacts")
    model_path: Path = Path("model.h5")
    random_state: int = 42
    epochs: int = 60
    batch_size: int = 256
    min_precision_for_recall_mode: float = 0.70
    recall_objective_weight: float = 0.65
    target_recall: float = 0.80
    seed_candidates: Tuple[int, ...] = (42, 77, 123)

    def load_dataset(self) -> pd.DataFrame:
        dataset_path = self.artifacts_dir / "processed_friday_balanced.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(
                "Islenmis veri bulunamadi. Once `python data_preprocessing.py` calistirin."
            )
        return pd.read_csv(dataset_path)

    @staticmethod
    def build_autoencoder(
        input_dim: int,
        enc_1: int = 128,
        enc_2: int = 64,
        latent_dim: int = 32,
        dropout_1: float = 0.2,
        dropout_2: float = 0.15,
        learning_rate: float = 1e-3,
        l2_lambda: float = 1e-5,
    ) -> Model:
        inputs = Input(shape=(input_dim,), name="input_layer")

        x = Dense(
            enc_1,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name="enc_dense_1",
        )(inputs)
        x = Dropout(dropout_1, name="enc_dropout_1")(x)
        x = Dense(
            enc_2,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name="enc_dense_2",
        )(x)
        x = Dropout(dropout_2, name="enc_dropout_2")(x)
        latent = Dense(
            latent_dim,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
            name="latent_space",
        )(x)

        x = Dense(enc_2, activation="relu", name="dec_dense_1")(latent)
        x = Dropout(dropout_2, name="dec_dropout_1")(x)
        x = Dense(enc_1, activation="relu", name="dec_dense_2")(x)
        outputs = Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

        model = Model(inputs, outputs, name="dense_autoencoder")
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    def prepare_splits(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        feature_columns = [col for col in data.columns if col not in {"Label", "binary_label"}]
        x = data[feature_columns].to_numpy(dtype=np.float32)
        y = data["binary_label"].to_numpy(dtype=np.int32)

        x_train_full, x_test, y_train_full, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=self.random_state,
        )

        x_train_normal = x_train[y_train == 0]
        return x_train_normal, x_val, y_val, x_test, y_test

    @staticmethod
    def reconstruction_errors(model: Model, samples: np.ndarray) -> np.ndarray:
        reconstructed = model.predict(samples, verbose=0)
        return np.mean(np.square(samples - reconstructed), axis=1)

    def find_optimal_thresholds(self, val_errors: np.ndarray, y_val: np.ndarray) -> dict:
        # Daha dusuk threshold degerleri daha fazla saldiri yakalama (recall) egilimindedir.
        candidate_thresholds = np.quantile(val_errors, np.linspace(0.50, 0.995, 300))

        best_f1_threshold = float(candidate_thresholds[0])
        best_f1 = -1.0

        best_recall_threshold = float(candidate_thresholds[0])
        best_recall_score = -1.0

        for threshold in candidate_thresholds:
            y_pred = (val_errors > threshold).astype(int)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            current_f1 = f1_score(y_val, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_f1_threshold = float(threshold)

            # Recall odakli mod: precision tamamen cokmesin diye alt sinir korunur.
            if precision >= self.min_precision_for_recall_mode:
                current_f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
                if current_f2 > best_recall_score:
                    best_recall_score = current_f2
                    best_recall_threshold = float(threshold)

        # Eger precision kosulunu saglayan aday cikmadiysa fallback olarak F1 threshold kullanilir.
        if best_recall_score < 0:
            best_recall_threshold = best_f1_threshold

        return {
            "balanced_threshold": best_f1_threshold,
            "recall_priority_threshold": best_recall_threshold,
        }

    @staticmethod
    def default_candidate_configs() -> List[Dict[str, float]]:
        # Yapiyi bozmadan kucuk bir hiperparametre taramasi yapilir.
        return [
            {
                "name": "baseline",
                "enc_1": 128,
                "enc_2": 64,
                "latent_dim": 32,
                "dropout_1": 0.20,
                "dropout_2": 0.15,
                "learning_rate": 1e-3,
                "l2_lambda": 1e-5,
            },
            {
                "name": "recall_dense",
                "enc_1": 160,
                "enc_2": 96,
                "latent_dim": 24,
                "dropout_1": 0.15,
                "dropout_2": 0.10,
                "learning_rate": 8e-4,
                "l2_lambda": 1e-5,
            },
            {
                "name": "regularized",
                "enc_1": 128,
                "enc_2": 64,
                "latent_dim": 16,
                "dropout_1": 0.25,
                "dropout_2": 0.20,
                "learning_rate": 1e-3,
                "l2_lambda": 5e-5,
            },
        ]

    def train_and_score_candidate(
        self,
        candidate_config: Dict[str, float],
        x_train_normal: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        seed: int,
    ) -> Dict[str, object]:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = self.build_autoencoder(
            input_dim=x_train_normal.shape[1],
            enc_1=int(candidate_config["enc_1"]),
            enc_2=int(candidate_config["enc_2"]),
            latent_dim=int(candidate_config["latent_dim"]),
            dropout_1=float(candidate_config["dropout_1"]),
            dropout_2=float(candidate_config["dropout_2"]),
            learning_rate=float(candidate_config["learning_rate"]),
            l2_lambda=float(candidate_config["l2_lambda"]),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        ]

        model.fit(
            x_train_normal,
            x_train_normal,
            validation_data=(x_val[y_val == 0], x_val[y_val == 0]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        val_errors = self.reconstruction_errors(model, x_val)
        thresholds = self.find_optimal_thresholds(val_errors, y_val)
        recall_threshold = thresholds["recall_priority_threshold"]
        y_pred = (val_errors > recall_threshold).astype(int)
        precision = float(precision_score(y_val, y_pred, zero_division=0))
        recall = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))

        weighted_score = (self.recall_objective_weight * recall) + ((1 - self.recall_objective_weight) * f1)
        return {
            "model": model,
            "config": candidate_config,
            "seed": seed,
            "thresholds": thresholds,
            "val_errors": val_errors,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1,
            "selection_score": weighted_score,
        }

    def select_best_result(self, results: List[Dict[str, object]]) -> Dict[str, object]:
        recall_eligible = [
            item for item in results if float(item["validation_recall"]) >= self.target_recall
        ]
        if recall_eligible:
            # Hedef recall saglandiysa en iyi F1 secilir.
            return max(recall_eligible, key=lambda item: float(item["validation_f1"]))
        # Hedef recall saglanamazsa en yuksek recall, esitlikte en iyi F1 secilir.
        return max(
            results,
            key=lambda item: (float(item["validation_recall"]), float(item["validation_f1"])),
        )

    def save_mse_plot(
        self,
        val_errors: np.ndarray,
        y_val: np.ndarray,
        balanced_threshold: float,
        recall_priority_threshold: float,
    ) -> None:
        plt.figure(figsize=(12, 6))
        plt.hist(val_errors[y_val == 0], bins=80, alpha=0.6, label="Normal", color="#2E8B57")
        plt.hist(val_errors[y_val == 1], bins=80, alpha=0.6, label="Saldiri", color="#CD5C5C")
        plt.axvline(
            balanced_threshold,
            color="#1E1E1E",
            linestyle="--",
            linewidth=2,
            label=f"Dengeli threshold={balanced_threshold:.6f}",
        )
        plt.axvline(
            recall_priority_threshold,
            color="#8B0000",
            linestyle="-.",
            linewidth=2,
            label=f"Recall oncelikli threshold={recall_priority_threshold:.6f}",
        )
        plt.title("Validation MSE Dagilimi", fontsize=14)
        plt.xlabel("MSE")
        plt.ylabel("Frekans")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "validation_mse_distribution.png", dpi=140)
        plt.close()

    def save_metadata(
        self,
        thresholds: dict,
        val_errors: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        balanced_threshold = thresholds["balanced_threshold"]
        recall_priority_threshold = thresholds["recall_priority_threshold"]

        y_val_pred_balanced = (val_errors > balanced_threshold).astype(int)
        y_val_pred_recall = (val_errors > recall_priority_threshold).astype(int)
        metadata = {
            "threshold": recall_priority_threshold,
            "threshold_strategy": "recall_priority",
            "balanced_threshold": balanced_threshold,
            "recall_priority_threshold": recall_priority_threshold,
            "validation_balanced_precision": float(
                precision_score(y_val, y_val_pred_balanced, zero_division=0)
            ),
            "validation_balanced_recall": float(
                recall_score(y_val, y_val_pred_balanced, zero_division=0)
            ),
            "validation_balanced_f1": float(f1_score(y_val, y_val_pred_balanced, zero_division=0)),
            "validation_recall_priority_precision": float(
                precision_score(y_val, y_val_pred_recall, zero_division=0)
            ),
            "validation_recall_priority_recall": float(
                recall_score(y_val, y_val_pred_recall, zero_division=0)
            ),
            "validation_recall_priority_f1": float(
                f1_score(y_val, y_val_pred_recall, zero_division=0)
            ),
        }

        with (self.artifacts_dir / "threshold.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        test_frame = pd.DataFrame(x_test)
        test_frame["binary_label"] = y_test
        test_frame.to_csv(self.artifacts_dir / "test_set.csv", index=False)

    def run(self) -> None:
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print("[1/5] Islenmis veri yukleniyor...")
        data = self.load_dataset()
        x_train_normal, x_val, y_val, x_test, y_test = self.prepare_splits(data)

        print("[2/5] Autoencoder adaylari 3 farkli seed ile egitiliyor...")
        candidate_results: List[Dict[str, object]] = []
        for candidate in self.default_candidate_configs():
            for seed in self.seed_candidates:
                print(f"-> Aday egitimi basladi: {candidate['name']} | seed={seed}")
                result = self.train_and_score_candidate(candidate, x_train_normal, x_val, y_val, seed=seed)
                candidate_results.append(result)
                print(
                    "-> Aday sonucu | "
                    f"precision={result['validation_precision']:.4f}, "
                    f"recall={result['validation_recall']:.4f}, "
                    f"f1={result['validation_f1']:.4f}, "
                    f"skor={result['selection_score']:.4f}"
                )

        best_result = self.select_best_result(candidate_results)
        model = best_result["model"]  # type: ignore[assignment]
        thresholds = best_result["thresholds"]  # type: ignore[assignment]
        val_errors = best_result["val_errors"]  # type: ignore[assignment]
        selected_config = best_result["config"]  # type: ignore[assignment]
        selected_seed = int(best_result["seed"])

        print("[3/5] En iyi aday secildi ve threshold hesaplandi...")
        balanced_threshold = thresholds["balanced_threshold"]
        recall_priority_threshold = thresholds["recall_priority_threshold"]
        self.save_mse_plot(val_errors, y_val, balanced_threshold, recall_priority_threshold)
        self.save_metadata(thresholds, val_errors, y_val, x_test, y_test)

        threshold_path = self.artifacts_dir / "threshold.json"
        with threshold_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["selected_model_config"] = selected_config
        metadata["selected_seed"] = selected_seed
        metadata["target_recall"] = self.target_recall
        metadata["selection_policy"] = "recall>=target ise en iyi f1, aksi halde en yuksek recall"
        metadata["candidate_results"] = [
            {
                "name": str(item["config"]["name"]),
                "seed": int(item["seed"]),
                "precision": float(item["validation_precision"]),
                "recall": float(item["validation_recall"]),
                "f1": float(item["validation_f1"]),
                "selection_score": float(item["selection_score"]),
            }
            for item in candidate_results
        ]
        with threshold_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        print("[4/5] Secilen model disa aktariliyor...")
        model.save(self.model_path)
        print("[5/5] Egitim raporu tamamlandi.")
        print(f"Egitim tamamlandi. Model: {self.model_path}")
        print(f"Secilen aday: {selected_config['name']} | seed={selected_seed}")
        print(f"Dengeli threshold (F1): {balanced_threshold:.6f}")
        print(f"Recall oncelikli threshold (F2): {recall_priority_threshold:.6f}")


if __name__ == "__main__":
    AutoencoderTrainer().run()

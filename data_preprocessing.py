"""CICIDS2017 Cuma verisi icin veri on isleme boru hatti."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@dataclass
class DataPreprocessor:
    """CICIDS2017 Cuma gunu dosyalarini okuyup model icin hazirlar."""

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    random_state: int = 42
    target_class_size: int = 100_000
    scaler: MinMaxScaler = field(default_factory=MinMaxScaler)
    categorical_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)

    def _friday_files(self) -> List[Path]:
        files = sorted(self.data_dir.glob("Friday-WorkingHours-*.csv"))
        if not files:
            raise FileNotFoundError("Cuma gunune ait CSV dosyalari bulunamadi.")
        return files

    def load_friday_data(self) -> pd.DataFrame:
        frames = []
        for csv_file in self._friday_files():
            frame = pd.read_csv(csv_file, low_memory=False)
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        data.columns = [col.strip() for col in data.columns]
        return data

    def clean_missing_and_infinite(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(axis=0).copy()
        return data

    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        object_columns = data.select_dtypes(include=["object"]).columns.tolist()
        object_columns = [col for col in object_columns if col != "Label"]

        for column in object_columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column].astype(str))
            self.categorical_encoders[column] = encoder
        return data

    @staticmethod
    def map_binary_label(raw_label: str) -> int:
        return 0 if str(raw_label).strip().upper() == "BENIGN" else 1

    def balance_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data["binary_label"] = data["Label"].apply(self.map_binary_label)

        normal_df = data[data["binary_label"] == 0]
        attack_df = data[data["binary_label"] == 1]

        normal_balanced = normal_df.sample(
            n=self.target_class_size,
            replace=len(normal_df) < self.target_class_size,
            random_state=self.random_state,
        )
        attack_balanced = attack_df.sample(
            n=self.target_class_size,
            replace=len(attack_df) < self.target_class_size,
            random_state=self.random_state,
        )

        balanced = pd.concat([normal_balanced, attack_balanced], ignore_index=True)
        balanced = balanced.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return balanced

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [col for col in data.columns if col not in {"Label", "binary_label"}]
        data[feature_columns] = self.scaler.fit_transform(data[feature_columns])
        return data

    def save_artifacts(self, data: pd.DataFrame) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_csv = self.artifacts_dir / "processed_friday_balanced.csv"
        output_columns = self.artifacts_dir / "feature_columns.json"
        output_scaler = self.artifacts_dir / "scaler_minmax.csv"

        data.to_csv(output_csv, index=False)

        feature_columns = [col for col in data.columns if col not in {"Label", "binary_label"}]
        with output_columns.open("w", encoding="utf-8") as file:
            json.dump(feature_columns, file, ensure_ascii=False, indent=2)

        # MinMax parametrelerini kaydederek tekrar uretilebilirlik saglanir.
        scaler_frame = pd.DataFrame(
            {
                "feature": feature_columns,
                "data_min": self.scaler.data_min_,
                "data_max": self.scaler.data_max_,
                "scale": self.scaler.scale_,
                "min": self.scaler.min_,
            }
        )
        scaler_frame.to_csv(output_scaler, index=False)

    def run(self) -> None:
        print("[1/5] Cuma gunu CSV dosyalari okunuyor...")
        data = self.load_friday_data()

        print("[2/5] NaN ve Infinity degerleri temizleniyor...")
        data = self.clean_missing_and_infinite(data)

        print("[3/5] Kategorik alanlar sayisallastiriliyor...")
        data = self.encode_categorical_features(data)

        print("[4/5] Veri seti 100k normal - 100k saldiri olarak dengeleniyor...")
        data = self.balance_dataset(data)

        print("[5/5] MinMaxScaler ile 0-1 normalizasyonu uygulaniyor...")
        data = self.scale_features(data)
        self.save_artifacts(data)

        class_counts = data["binary_label"].value_counts().to_dict()
        print("On isleme tamamlandi.")
        print(f"Sinif dagilimi: {class_counts}")
        print("Cikti dosyasi: artifacts/processed_friday_balanced.csv")


if __name__ == "__main__":
    DataPreprocessor().run()

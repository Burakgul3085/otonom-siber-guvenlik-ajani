"""Proje icin sunuma hazir Turkce gorsellestirme ciktilari uretir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass
class GorsellestirmeAraci:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("artifacts/gorseller")
    model_path: Path = Path("model.h5")
    random_state: int = 42
    pca_ornek_sayisi: int = 6000
    detay_ornek_sayisi: int = 12000
    secili_ozellik_sayisi: int = 6

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ham_veri_dir = self.output_dir / "ham_veri"
        self.model_oncesi_dir = self.output_dir / "model_oncesi"
        self.model_sirasi_dir = self.output_dir / "model_sirasi"
        self.model_sonrasi_dir = self.output_dir / "model_sonrasi"
        for folder in [
            self.ham_veri_dir,
            self.model_oncesi_dir,
            self.model_sirasi_dir,
            self.model_sonrasi_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _kaydet_yolu(self, asama: str, dosya_adi: str) -> Path:
        klasor_map = {
            "ham_veri": self.ham_veri_dir,
            "model_oncesi": self.model_oncesi_dir,
            "model_sirasi": self.model_sirasi_dir,
            "model_sonrasi": self.model_sonrasi_dir,
        }
        if asama not in klasor_map:
            raise ValueError(f"Bilinmeyen asama: {asama}")
        return klasor_map[asama] / dosya_adi

    @staticmethod
    def _etiket_binary(label: str) -> int:
        return 0 if str(label).strip().upper() == "BENIGN" else 1

    @staticmethod
    def _etiket_yazi(binary_label: int) -> str:
        return "normal" if int(binary_label) == 0 else "saldiri"

    def _cuma_dosyalari(self) -> List[Path]:
        files = sorted(self.data_dir.glob("Friday-WorkingHours-*.csv"))
        if not files:
            raise FileNotFoundError("Cuma gunu CSV dosyalari bulunamadi.")
        return files

    def _ham_veri(self) -> pd.DataFrame:
        frames = [pd.read_csv(path, low_memory=False) for path in self._cuma_dosyalari()]
        data = pd.concat(frames, ignore_index=True)
        data.columns = [col.strip() for col in data.columns]
        return data

    def _islenmis_veri(self) -> pd.DataFrame:
        path = self.artifacts_dir / "processed_friday_balanced.csv"
        if not path.exists():
            raise FileNotFoundError("Islenmis veri bulunamadi. Once data_preprocessing.py calistirin.")
        return pd.read_csv(path)

    def _test_verisi(self) -> pd.DataFrame:
        path = self.artifacts_dir / "test_set.csv"
        if not path.exists():
            raise FileNotFoundError("Test seti bulunamadi. Once train_autoencoder.py calistirin.")
        return pd.read_csv(path)

    def _threshold_bilgisi(self) -> dict:
        path = self.artifacts_dir / "threshold.json"
        if not path.exists():
            raise FileNotFoundError("threshold.json bulunamadi. Once train_autoencoder.py calistirin.")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _secili_ozellikler(self, islenmis: pd.DataFrame) -> List[str]:
        feature_cols = [c for c in islenmis.columns if c not in {"Label", "binary_label"}]
        variances = islenmis[feature_cols].var().sort_values(ascending=False)
        return variances.head(self.secili_ozellik_sayisi).index.tolist()

    def sinif_dagilimi_grafigi(self, ham: pd.DataFrame, islenmis: pd.DataFrame) -> None:
        ham_binary = ham["Label"].apply(self._etiket_binary)
        ham_counts = ham_binary.value_counts().reindex([0, 1], fill_value=0)
        islenmis_counts = islenmis["binary_label"].value_counts().reindex([0, 1], fill_value=0)

        frame = pd.DataFrame(
            {
                "asama": ["ham", "ham", "islenmis", "islenmis"],
                "sinif": ["normal", "saldiri", "normal", "saldiri"],
                "adet": [
                    int(ham_counts.loc[0]),
                    int(ham_counts.loc[1]),
                    int(islenmis_counts.loc[0]),
                    int(islenmis_counts.loc[1]),
                ],
            }
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=frame, x="asama", y="adet", hue="sinif", palette=["#2E8B57", "#CD5C5C"])
        plt.title("Sinif Dagilimi: Ham Veri vs Islenmis Veri", fontsize=13)
        plt.xlabel("Veri Asamasi")
        plt.ylabel("Kayit Sayisi")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("ham_veri", "01_sinif_dagilimi_ham_vs_islenmis.png"), dpi=150)
        plt.close()

    def eksik_bozuk_deger_grafigi(self, ham: pd.DataFrame) -> None:
        toplam_nan = int(ham.isna().sum().sum())
        numeric = ham.select_dtypes(include=[np.number])
        toplam_inf = int(np.isinf(numeric.to_numpy()).sum())

        temiz = ham.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        temiz_nan = int(temiz.isna().sum().sum())
        temiz_numeric = temiz.select_dtypes(include=[np.number])
        temiz_inf = int(np.isinf(temiz_numeric.to_numpy()).sum())

        frame = pd.DataFrame(
            {
                "metrik": ["nan", "inf", "nan", "inf"],
                "asama": ["temizlik oncesi", "temizlik oncesi", "temizlik sonrasi", "temizlik sonrasi"],
                "adet": [toplam_nan, toplam_inf, temiz_nan, temiz_inf],
            }
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(data=frame, x="metrik", y="adet", hue="asama", palette=["#A9A9A9", "#4682B4"])
        plt.title("NaN ve Infinity Degerlerinin Temizlik Oncesi/Sonrasi Durumu", fontsize=13)
        plt.xlabel("Deger Tipi")
        plt.ylabel("Adet")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("ham_veri", "02_nan_inf_temizlik_karsilastirmasi.png"), dpi=150)
        plt.close()

    def pca_dagilimi_grafigi(self, islenmis: pd.DataFrame) -> None:
        sample = islenmis.sample(
            n=min(self.pca_ornek_sayisi, len(islenmis)),
            random_state=self.random_state,
        ).copy()
        features = sample.drop(columns=["Label", "binary_label"])
        pca = PCA(n_components=2, random_state=self.random_state)
        components = pca.fit_transform(features.to_numpy(dtype=np.float32))

        plot_frame = pd.DataFrame(
            {
                "bilesen_1": components[:, 0],
                "bilesen_2": components[:, 1],
                "sinif": sample["binary_label"].apply(self._etiket_yazi).to_numpy(),
            }
        )

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=plot_frame,
            x="bilesen_1",
            y="bilesen_2",
            hue="sinif",
            palette={"normal": "#2E8B57", "saldiri": "#CD5C5C"},
            alpha=0.55,
            s=18,
        )
        plt.title("PCA ile Ozellik Uzayi Dagilimi (2 Boyut)", fontsize=13)
        plt.xlabel("PCA Bileseni 1")
        plt.ylabel("PCA Bileseni 2")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_oncesi", "03_pca_ozellik_dagilimi.png"), dpi=150)
        plt.close()

    def korelasyon_matrisi_grafigi(self, islenmis: pd.DataFrame, secili_ozellikler: List[str]) -> None:
        corr = islenmis[secili_ozellikler].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Secili Ozellikler Korelasyon Matrisi", fontsize=13)
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_oncesi", "04_secili_ozellikler_korelasyon_matrisi.png"), dpi=150)
        plt.close()

    def secili_ozellik_violin_grafigi(self, islenmis: pd.DataFrame, secili_ozellikler: List[str]) -> None:
        sample = islenmis.sample(
            n=min(self.detay_ornek_sayisi, len(islenmis)),
            random_state=self.random_state,
        ).copy()
        long_frame = sample[secili_ozellikler + ["binary_label"]].melt(
            id_vars="binary_label", var_name="ozellik", value_name="deger"
        )
        long_frame["sinif"] = long_frame["binary_label"].apply(self._etiket_yazi)

        plt.figure(figsize=(14, 7))
        sns.violinplot(
            data=long_frame,
            x="ozellik",
            y="deger",
            hue="sinif",
            split=True,
            inner="quart",
            palette={"normal": "#2E8B57", "saldiri": "#CD5C5C"},
        )
        plt.title("Secili Ozelliklerde Sinif Bazli Violin Dagilimi", fontsize=13)
        plt.xlabel("Ozellik")
        plt.ylabel("Normalize Deger")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_oncesi", "05_secili_ozellik_violin_sinif_dagilimi.png"), dpi=150)
        plt.close()

    def secili_ozellik_kutu_grafigi(self, islenmis: pd.DataFrame, secili_ozellikler: List[str]) -> None:
        sample = islenmis.sample(
            n=min(self.detay_ornek_sayisi, len(islenmis)),
            random_state=self.random_state,
        ).copy()
        long_frame = sample[secili_ozellikler + ["binary_label"]].melt(
            id_vars="binary_label", var_name="ozellik", value_name="deger"
        )
        long_frame["sinif"] = long_frame["binary_label"].apply(self._etiket_yazi)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=long_frame, x="ozellik", y="deger", hue="sinif")
        plt.title("Secili Ozelliklerde Sinif Bazli Kutu Grafigi", fontsize=13)
        plt.xlabel("Ozellik")
        plt.ylabel("Normalize Deger")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_oncesi", "06_secili_ozellik_kutu_grafigi.png"), dpi=150)
        plt.close()

    @staticmethod
    def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values)
        y = np.arange(1, len(values) + 1) / len(values)
        return x, y

    def model_skor_grafikleri(self, test_set: pd.DataFrame, threshold: float) -> None:
        model = tf.keras.models.load_model(self.model_path, compile=False)
        y_true = test_set["binary_label"].to_numpy(dtype=np.int32)
        x_test = test_set.drop(columns=["binary_label"]).to_numpy(dtype=np.float32)
        reconstructed = model.predict(x_test, verbose=0)
        mse_values = np.mean(np.square(x_test - reconstructed), axis=1)
        y_pred = (mse_values > threshold).astype(int)

        # 07 MSE dagilimi
        plt.figure(figsize=(11, 6))
        plt.hist(mse_values[y_true == 0], bins=80, alpha=0.6, label="normal", color="#2E8B57")
        plt.hist(mse_values[y_true == 1], bins=80, alpha=0.6, label="saldiri", color="#CD5C5C")
        plt.axvline(threshold, color="#1E1E1E", linestyle="--", linewidth=2, label=f"threshold={threshold:.6f}")
        plt.title("Test MSE Dagilimi ve Karar Esigi", fontsize=13)
        plt.xlabel("MSE")
        plt.ylabel("Frekans")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "07_test_mse_dagilimi_ve_esik.png"), dpi=150)
        plt.close()

        # 08 ROC
        fpr, tpr, _ = roc_curve(y_true, mse_values)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
        plt.title("ROC Egrisi", fontsize=13)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "08_roc_egrisi.png"), dpi=150)
        plt.close()

        # 09 PR
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, mse_values)
        pr_auc = auc(recall_curve, precision_curve)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color="#6A5ACD", linewidth=2, label=f"AUC={pr_auc:.4f}")
        plt.title("Precision-Recall Egrisi", fontsize=13)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "09_precision_recall_egrisi.png"), dpi=150)
        plt.close()

        # 10 confusion matrix (adet)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix (Adet)", fontsize=13)
        plt.xlabel("Tahmin")
        plt.ylabel("Gercek")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "10_confusion_matrix_adet.png"), dpi=150)
        plt.close()

        # 11 confusion matrix (oran)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples", cbar=False)
        plt.title("Confusion Matrix (Sinif Ici Oran)", fontsize=13)
        plt.xlabel("Tahmin")
        plt.ylabel("Gercek")
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "11_confusion_matrix_oran.png"), dpi=150)
        plt.close()

        # 12 threshold tarama
        scales = np.round(np.arange(0.80, 1.21, 0.05), 2)
        records = []
        for scale in scales:
            current_threshold = threshold * float(scale)
            current_pred = (mse_values > current_threshold).astype(int)
            records.append(
                {
                    "threshold_carpani": float(scale),
                    "precision": precision_score(y_true, current_pred, zero_division=0),
                    "recall": recall_score(y_true, current_pred, zero_division=0),
                    "f1": f1_score(y_true, current_pred, zero_division=0),
                }
            )
        metric_frame = pd.DataFrame(records)
        plt.figure(figsize=(10, 6))
        plt.plot(metric_frame["threshold_carpani"], metric_frame["precision"], marker="o", label="precision")
        plt.plot(metric_frame["threshold_carpani"], metric_frame["recall"], marker="o", label="recall")
        plt.plot(metric_frame["threshold_carpani"], metric_frame["f1"], marker="o", label="f1-score")
        plt.title("Threshold Carpani Tarama Sonuclari", fontsize=13)
        plt.xlabel("Threshold Carpani")
        plt.ylabel("Skor")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sirasi", "12_threshold_tarama_sonuclari.png"), dpi=150)
        plt.close()

        # 13 MSE kutu grafigi
        mse_frame = pd.DataFrame({"mse": mse_values, "sinif": [self._etiket_yazi(v) for v in y_true]})
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=mse_frame, x="sinif", y="mse", palette={"normal": "#2E8B57", "saldiri": "#CD5C5C"})
        plt.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label="threshold")
        plt.title("Sinif Bazli MSE Kutu Grafigi", fontsize=13)
        plt.xlabel("Sinif")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "13_sinif_bazli_mse_kutu_grafigi.png"), dpi=150)
        plt.close()

        # 14 MSE ECDF
        x_norm, y_norm = self._ecdf(mse_values[y_true == 0])
        x_att, y_att = self._ecdf(mse_values[y_true == 1])
        plt.figure(figsize=(9, 6))
        plt.plot(x_norm, y_norm, label="normal", color="#2E8B57")
        plt.plot(x_att, y_att, label="saldiri", color="#CD5C5C")
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="threshold")
        plt.title("MSE Kumulatif Dagilim (ECDF)", fontsize=13)
        plt.xlabel("MSE")
        plt.ylabel("Kumulatif Olasilik")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "14_mse_ecdf_sinif_karsilastirmasi.png"), dpi=150)
        plt.close()

        # 15 MSE zaman serisi
        plt.figure(figsize=(11, 5))
        plot_count = min(1200, len(mse_values))
        plt.plot(np.arange(plot_count), mse_values[:plot_count], color="#1f77b4", linewidth=1)
        plt.axhline(threshold, color="#8B0000", linestyle="--", label="threshold")
        plt.title("MSE Zaman Serisi (Ilk 1200 Paket)", fontsize=13)
        plt.xlabel("Paket Sirasi")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "15_mse_zaman_serisi_ilk_1200.png"), dpi=150)
        plt.close()

        # 16 Kumulatif recall
        cumulative_true_attack = np.cumsum(y_true == 1)
        cumulative_tp = np.cumsum((y_true == 1) & (y_pred == 1))
        cumulative_recall = np.divide(
            cumulative_tp,
            cumulative_true_attack,
            out=np.zeros_like(cumulative_tp, dtype=float),
            where=cumulative_true_attack != 0,
        )
        plt.figure(figsize=(11, 5))
        plt.plot(cumulative_recall, color="#6A5ACD")
        plt.title("Kumulatif Recall (Paket Akisi Boyunca)", fontsize=13)
        plt.xlabel("Paket Sirasi")
        plt.ylabel("Recall")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "16_kumulatif_recall_paket_akisi.png"), dpi=150)
        plt.close()

        # 17 QQ Plot (MSE)
        plt.figure(figsize=(8, 6))
        stats.probplot(mse_values, dist="norm", plot=plt)
        plt.title("MSE Icin QQ Plot", fontsize=13)
        plt.tight_layout()
        plt.savefig(self._kaydet_yolu("model_sonrasi", "17_mse_qq_plot.png"), dpi=150)
        plt.close()

        # 18/19 Hata tablolari
        result_frame = pd.DataFrame(
            {
                "mse": mse_values,
                "gercek": y_true,
                "tahmin": y_pred,
                "esik_farki": mse_values - threshold,
            }
        )
        en_kacirilan = result_frame[(result_frame["gercek"] == 1) & (result_frame["tahmin"] == 0)].copy()
        en_kacirilan = en_kacirilan.sort_values(by="esik_farki", ascending=True).head(30)
        en_kacirilan.to_csv(self._kaydet_yolu("model_sonrasi", "18_en_cok_kacirilan_saldirilar.csv"), index=False)

        en_yanlis_alarm = result_frame[(result_frame["gercek"] == 0) & (result_frame["tahmin"] == 1)].copy()
        en_yanlis_alarm = en_yanlis_alarm.sort_values(by="esik_farki", ascending=False).head(30)
        en_yanlis_alarm.to_csv(self._kaydet_yolu("model_sonrasi", "19_en_cok_yanlis_alarmlar.csv"), index=False)

    def calistir(self) -> None:
        print("[1/5] Ham ve islenmis veri yukleniyor...")
        ham = self._ham_veri()
        islenmis = self._islenmis_veri()
        secili_ozellikler = self._secili_ozellikler(islenmis)

        print("[2/5] Veri oncesi ve ozellik dagilimi grafik seti olusturuluyor...")
        self.sinif_dagilimi_grafigi(ham, islenmis)
        self.eksik_bozuk_deger_grafigi(ham)
        self.pca_dagilimi_grafigi(islenmis)
        self.korelasyon_matrisi_grafigi(islenmis, secili_ozellikler)
        self.secili_ozellik_violin_grafigi(islenmis, secili_ozellikler)
        self.secili_ozellik_kutu_grafigi(islenmis, secili_ozellikler)

        print("[3/5] Model skor ve performans grafik seti olusturuluyor...")
        test_set = self._test_verisi()
        threshold = float(self._threshold_bilgisi()["threshold"])
        self.model_skor_grafikleri(test_set=test_set, threshold=threshold)

        print("[4/5] Tablosal hata analizleri kaydedildi.")
        print("[5/5] Gorsellestirme tamamlandi.")
        print(f"Ciktilar: {self.output_dir}")


if __name__ == "__main__":
    GorsellestirmeAraci().calistir()

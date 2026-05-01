"""Siber guvenlik ajani icin Streamlit tabanli izleme paneli."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import platform
import subprocess
from time import sleep
from typing import List, Tuple

import json
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class FirewallBlocker:
    """IP bloklama aksiyonunu dry-run veya gercek modda uygular."""

    dry_run: bool = True

    def _build_command(self, ip_address: str) -> List[str]:
        system_name = platform.system().lower()
        if "windows" in system_name:
            return [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name=cyber-agent-block-{ip_address}",
                "dir=in",
                "action=block",
                f"remoteip={ip_address}",
            ]
        return ["iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"]

    def block_ip(self, ip_address: str) -> str:
        command = self._build_command(ip_address)
        command_str = " ".join(command)

        if self.dry_run:
            return f"[dry-run] {ip_address} icin komut hazirlandi: {command_str}"

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return f"{ip_address} icin firewall kurali uygulandi."
        except Exception as exc:
            return f"{ip_address} icin firewall komutu basarisiz: {exc}"


@dataclass
class CyberSecurityAgent:
    """Model ciktisina gore anomali tespit eder ve dummy bloklama uygular."""

    threshold: float
    decision_window_size: int = 5
    min_attack_votes: int = 3
    blocked_ips: List[str] = field(default_factory=list)
    recent_raw_predictions: deque = field(default_factory=deque)
    blocker: FirewallBlocker = field(default_factory=FirewallBlocker)

    def configure_window(self, window_size: int, min_attack_votes: int) -> None:
        self.decision_window_size = max(1, window_size)
        self.min_attack_votes = max(1, min(min_attack_votes, self.decision_window_size))
        self.recent_raw_predictions = deque(maxlen=self.decision_window_size)

    def decide(self, mse_value: float) -> Tuple[int, int]:
        raw_prediction = int(mse_value > self.threshold)
        self.recent_raw_predictions.append(raw_prediction)
        attack_votes = sum(self.recent_raw_predictions)
        smoothed_prediction = int(attack_votes >= self.min_attack_votes)
        return raw_prediction, smoothed_prediction

    def block_ip(self, ip_address: str) -> str:
        if ip_address not in self.blocked_ips:
            self.blocked_ips.append(ip_address)
        return self.blocker.block_ip(ip_address)


class AgentDashboard:
    """Modeli test setinde paket paket calistirir ve canli panelde gosterir."""

    def __init__(self, artifacts_dir: Path = Path("artifacts"), model_path: Path = Path("model.h5")) -> None:
        self.artifacts_dir = artifacts_dir
        self.model_path = model_path
        # Dashboard tarafinda sadece inferans yapildigi icin compile metadata'sini yuklemiyoruz.
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.threshold, self.threshold_metadata = self._load_threshold()
        self.agent = CyberSecurityAgent(threshold=self.threshold)
        self.test_x, self.test_y = self._load_test_data()
        self.feature_names = self._load_feature_names(self.test_x.shape[1])
        self.feature_defaults = np.median(self.test_x, axis=0)
        feature_std = np.std(self.test_x, axis=0)
        self.top_feature_indices = np.argsort(feature_std)[::-1][:12]

    def _load_threshold(self) -> Tuple[float, dict]:
        threshold_path = self.artifacts_dir / "threshold.json"
        with threshold_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return float(metadata["threshold"]), metadata

    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_path = self.artifacts_dir / "test_set.csv"
        frame = pd.read_csv(test_path)
        y = frame["binary_label"].to_numpy(dtype=np.int32)
        x = frame.drop(columns=["binary_label"]).to_numpy(dtype=np.float32)
        return x, y

    def _load_feature_names(self, expected_count: int) -> List[str]:
        columns_path = self.artifacts_dir / "feature_columns.json"
        if columns_path.exists():
            try:
                with columns_path.open("r", encoding="utf-8") as file:
                    names = json.load(file)
                if isinstance(names, list) and len(names) == expected_count:
                    return [str(name) for name in names]
            except Exception:
                pass
        return [f"ozellik_{idx + 1}" for idx in range(expected_count)]

    def _mse(self, sample: np.ndarray) -> float:
        sample = sample.reshape(1, -1)
        reconstructed = self.model.predict(sample, verbose=0)
        return float(np.mean(np.square(sample - reconstructed)))

    def run_simulation(self, max_packets: int, delay_seconds: float) -> None:
        st.set_page_config(page_title="Cyber Security Agent", layout="wide")
        st.title("Cyber Security Agent Dashboard")
        st.caption("Anomali tabanli otonom saldiri tespit ve dummy IP bloklama simulasyonu")

        left, right = st.columns([2, 1])
        with left:
            log_area = st.empty()
            chart_area = st.empty()
        with right:
            status_area = st.empty()
            counter_area = st.empty()

        mse_history: List[float] = []
        prediction_history: List[int] = []
        true_history: List[int] = []
        logs: List[str] = []

        packet_count = min(max_packets, len(self.test_x))
        for idx in range(packet_count):
            mse_value = self._mse(self.test_x[idx])
            raw_prediction, prediction = self.agent.decide(mse_value)
            true_label = int(self.test_y[idx])

            true_history.append(true_label)
            prediction_history.append(prediction)
            mse_history.append(mse_value)

            packet_ip = f"192.168.1.{(idx % 250) + 1}"
            if prediction == 1:
                block_message = self.agent.block_ip(packet_ip)
                logs.append(
                    f"[{idx + 1}] SALDIRI TESPIT EDILDI | mse={mse_value:.6f} | ham={raw_prediction} | {block_message}"
                )
                status_area.error("SALDIRI TESPIT EDILDI - IP BLOKLANIYOR")
            else:
                logs.append(f"[{idx + 1}] Guvenli trafik | mse={mse_value:.6f} | ham={raw_prediction}")
                status_area.success("Guvenli")

            chart_data = pd.DataFrame({"mse": mse_history})
            chart_area.line_chart(chart_data, y="mse")
            log_area.code("\n".join(logs[-20:]), language="text")
            counter_area.info(
                f"Islenen paket: {idx + 1}/{packet_count}\n"
                f"Threshold: {self.agent.threshold:.6f}\n"
                f"Pencere kurali: son {self.agent.decision_window_size} pakette en az {self.agent.min_attack_votes} saldiri\n"
                f"Bloklanan IP: {len(self.agent.blocked_ips)}"
            )
            sleep(delay_seconds)

        self.render_final_report(np.array(true_history), np.array(prediction_history))

    def render_final_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        st.subheader("Simulasyon Sonucu")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Precision", f"{precision:.4f}")
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.write(f"Toplam bloklanan IP sayisi: **{len(self.agent.blocked_ips)}**")

    def render_manual_test_panel(self) -> None:
        st.subheader("Manuel Paket Testi")
        st.caption(
            "Kritik ozellikleri manuel girerek tek paket icin anlik MSE ve saldiri kararini test edebilirsiniz. "
            "Girmediginiz ozellikler test seti medyan degeri ile doldurulur."
        )

        feature_count = st.slider(
            "Manuel girilecek kritik ozellik sayisi",
            min_value=4,
            max_value=min(12, len(self.top_feature_indices)),
            value=8,
            step=1,
            key="manuel_ozellik_sayisi",
        )

        manual_sample = self.feature_defaults.copy()
        selected_indices = self.top_feature_indices[:feature_count]
        input_keys = [f"manuel_{self.feature_names[int(idx)]}_{int(idx)}" for idx in selected_indices]

        if "manuel_panel_init" not in st.session_state:
            st.session_state["manuel_panel_init"] = True
            for idx, key in zip(selected_indices, input_keys):
                st.session_state[key] = float(np.clip(self.feature_defaults[int(idx)], 0.0, 1.0))

        action_cols = st.columns(3)
        if action_cols[0].button("Normal ornekten doldur", key="btn_normal_doldur"):
            normal_indices = np.where(self.test_y == 0)[0]
            if len(normal_indices) > 0:
                picked = int(normal_indices[0])
                for idx, key in zip(selected_indices, input_keys):
                    st.session_state[key] = float(np.clip(self.test_x[picked, int(idx)], 0.0, 1.0))

        if action_cols[1].button("Saldiri orneginden doldur", key="btn_saldiri_doldur"):
            attack_indices = np.where(self.test_y == 1)[0]
            if len(attack_indices) > 0:
                picked = int(attack_indices[0])
                for idx, key in zip(selected_indices, input_keys):
                    st.session_state[key] = float(np.clip(self.test_x[picked, int(idx)], 0.0, 1.0))

        if action_cols[2].button("Alanlari medyana sifirla", key="btn_medyan_sifirla"):
            for idx, key in zip(selected_indices, input_keys):
                st.session_state[key] = float(np.clip(self.feature_defaults[int(idx)], 0.0, 1.0))

        columns = st.columns(2)

        for order, (feature_idx, input_key) in enumerate(zip(selected_indices, input_keys)):
            feature_name = self.feature_names[int(feature_idx)]
            default_value = float(self.feature_defaults[int(feature_idx)])

            with columns[order % 2]:
                manual_value = st.number_input(
                    f"{feature_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(np.clip(default_value, 0.0, 1.0)),
                    step=0.01,
                    key=input_key,
                )
            manual_sample[int(feature_idx)] = manual_value

        apply_block_action = st.checkbox("Saldiri cikarsa bloklama aksiyonunu uygula", value=False, key="manuel_bloklama")
        manual_ip = st.text_input("Test IP adresi", value="192.168.100.10", key="manuel_ip")

        if st.button("Manuel paketi test et", key="manuel_test_buton"):
            mse_value = self._mse(manual_sample.astype(np.float32))
            raw_prediction = int(mse_value > self.agent.threshold)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("MSE", f"{mse_value:.6f}")
            with col2:
                st.metric("Aktif Threshold", f"{self.agent.threshold:.6f}")

            if raw_prediction == 1:
                st.error("Sonuc: SALDIRI ADAYI")
                if apply_block_action:
                    block_message = self.agent.block_ip(manual_ip)
                    st.warning(f"Bloklama sonucu: {block_message}")
            else:
                st.success("Sonuc: Guvenli Trafik")


def main() -> None:
    st.sidebar.header("Simulasyon Ayarlari")
    max_packets = st.sidebar.slider("Maksimum paket", min_value=50, max_value=5000, value=500, step=50)
    delay_seconds = st.sidebar.slider("Paketler arasi gecikme (sn)", min_value=0.0, max_value=0.3, value=0.03, step=0.01)

    dashboard = AgentDashboard()
    threshold_scale = st.sidebar.slider(
        "Threshold carpan (dusuk deger = daha yuksek recall)",
        min_value=0.50,
        max_value=1.50,
        value=1.00,
        step=0.05,
    )
    dashboard.agent.threshold = dashboard.threshold * threshold_scale
    window_size = st.sidebar.slider("Karar penceresi boyutu", min_value=1, max_value=10, value=5, step=1)
    min_attack_votes = st.sidebar.slider(
        "Pencerede minimum saldiri oyu",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    dashboard.agent.configure_window(window_size, min_attack_votes)
    dry_run_mode = st.sidebar.checkbox("Firewall dry-run modu", value=True)
    dashboard.agent.blocker = FirewallBlocker(dry_run=dry_run_mode)
    strategy = dashboard.threshold_metadata.get("threshold_strategy", "legacy")
    st.sidebar.caption(
        f"Temel threshold stratejisi: {strategy} | aktif threshold: {dashboard.agent.threshold:.6f}"
    )
    st.sidebar.caption(f"Firewall modu: {'dry-run' if dry_run_mode else 'gercek komut'}")

    if st.sidebar.button("Ajan simulasyonunu baslat"):
        dashboard.run_simulation(max_packets=max_packets, delay_seconds=delay_seconds)
    else:
        st.info("Simulasyonu baslatmak icin soldaki butonu kullanin.")

    st.markdown("---")
    dashboard.render_manual_test_panel()


if __name__ == "__main__":
    main()

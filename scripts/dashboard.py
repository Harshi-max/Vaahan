"""Streamlit leaderboard dashboard for ASR Shootout results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "data" / "outputs" / "metrics" / "full_results.csv"
CHARTS_DIR = ROOT / "data" / "outputs" / "charts"


@st.cache_data
def load_results() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


def main() -> None:
    st.set_page_config(page_title="ASR Shootout Leaderboard", layout="wide")
    st.title("ASR Shootout — Leaderboard")
    st.caption("Indian conversational speech · Hinglish · Locality entities")

    df = load_results()
    if df.empty:
        st.warning("No results yet. Run: `bash run.sh`")
        st.stop()

    summary = (
        df.groupby("model_name")
        .agg(
            WER=("wer", "mean"),
            CER=("cer", "mean"),
            Entity_Acc=("locality_correct", "mean"),
            Latency_s=("latency_seconds", "mean"),
            Token_F1=("token_f1", "mean"),
        )
        .round(4)
    )
    summary["Entity_Acc"] = (summary["Entity_Acc"] * 100).round(1)
    summary = summary.sort_values("Entity_Acc", ascending=False)

    st.subheader("Model Rankings")
    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Filter by Condition")
        conditions = ["All"] + sorted(df["condition"].unique().tolist())
        cond = st.selectbox("Condition", conditions)
        filtered = df if cond == "All" else df[df["condition"] == cond]
        st.bar_chart(
            filtered.groupby("model_name")["locality_correct"].mean() * 100
        )

    with col2:
        st.subheader("Hardest Localities")
        loc = (
            filtered.groupby("locality_name")["locality_correct"]
            .mean()
            .sort_values()
            .head(8)
        )
        st.bar_chart(loc * 100)

    st.subheader("Charts")
    if CHARTS_DIR.exists():
        for img in sorted(CHARTS_DIR.glob("*.png")):
            st.image(str(img), caption=img.name, use_container_width=True)

    st.subheader("Sample Failures")
    failures = df[~df["locality_correct"]].nlargest(10, "wer")
    st.dataframe(
        failures[
            ["filename", "model_name", "locality_name", "wer", "hypothesis"]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()

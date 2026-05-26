import io
from pathlib import Path
from typing import List
import sys

# Ensure project root is on Python path so `src` imports work when Streamlit runs
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local imports from project
from src.models.registry import get_active_models
from src.utils.audio import TARGET_SAMPLE_RATE
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.entity_accuracy import compute_entity_metrics

DATA_DIR = ROOT / "data"
OUTPUTS = DATA_DIR / "outputs"
METRICS_CSV = OUTPUTS / "metrics" / "full_results.csv"
TRANSCRIPTS_DIR = OUTPUTS / "transcripts"

st.set_page_config(page_title="ASR Shootout Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark mode friendly styles + enhanced UI
st.markdown(
    """
    <style>
    /* Base dark background and readable body text */
    .css-1d391kg, .reportview-container .main, .stApp {
        background-color: #0e1117 !important;
        color: #e6eef8 !important;
    }

    /* Ensure headings are highly visible on dark background */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 0 rgba(0,0,0,0.6) !important;
    }

    /* Markdown and subheader text */
    .stMarkdown, .stText, .css-1v0mbdj, .css-ffhzg2 {
        color: #e6eef8 !important;
    }

    /* Sidebar text contrast */
    .css-18e3th9, .css-1oe6wy5 {
        color: #e6eef8 !important;
    }

    /* Enhanced audio player visibility */
    audio {
        width: 100% !important;
        outline: 2px solid #00FFAA !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    /* Button visibility improvements */
    .stButton > button {
        background-color: #00FFAA !important;
        color: #0e1117 !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        background-color: #00dd88 !important;
    }

    /* Info box styling */
    .stInfo, .stSuccess, .stWarning {
        border-left: 5px solid #00FFAA !important;
        padding: 12px !important;
        border-radius: 6px !important;
    }

    /* File uploader visibility */
    .uploadedFile {
        border: 2px solid #00FFAA !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh option (sidebar)
auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=False)
refresh_seconds = st.sidebar.number_input("Refresh interval (sec)", min_value=5, max_value=600, value=10, step=5)
if auto_refresh:
    # Inject a meta refresh tag to reload the page periodically
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

@st.cache_data
def load_metrics() -> pd.DataFrame:
    if METRICS_CSV.exists():
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame()

@st.cache_data
def list_transcripts() -> List[str]:
    if TRANSCRIPTS_DIR.exists():
        return [p.name for p in TRANSCRIPTS_DIR.glob("*.csv")]
    return []

@st.cache_resource
def get_models():
    # instantiate available models (skip unavailable)
    models = get_active_models(skip_unavailable=True)
    return {m.name: m for m in models}

metrics_df = load_metrics()
transcript_files = list_transcripts()
models = get_models()

# Sidebar
st.sidebar.title("🎛️ Controls")
with st.sidebar.expander("📊 Model Selection", expanded=True):
    selected_models = st.multiselect("Select models to show", options=list(models.keys()), default=list(models.keys()), help="Choose models to compare")

with st.sidebar.expander("🔍 Filter by Condition", expanded=True):
    conditions = sorted(metrics_df["condition"].unique().tolist()) if not metrics_df.empty else []
    selected_conditions = st.multiselect("Filter by condition", options=conditions, default=conditions, help="Filter results by test condition")

# Main layout: tabs
tab1, tab2, tab3, tab4 = st.tabs(["Benchmark Results", "Audio Explorer", "Error Analysis", "Live Testing"])

with tab1:
    st.header("📊 Benchmark Results")
    if metrics_df.empty:
        st.info("ℹ️ No benchmark metrics found. Run the pipeline or use demo data.")
    else:
        df = metrics_df[metrics_df["model_name"].isin(selected_models) & metrics_df["condition"].isin(selected_conditions)]
        
        if df.empty:
            st.warning("⚠️ No data matches your filters. Try adjusting your selections.")
        else:
            # Summary metrics
            st.markdown("### 📈 Summary Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_wer = df["wer"].mean()
                st.metric("Avg WER", f"{avg_wer:.4f}")
            with col2:
                avg_entity = df["locality_correct"].mean() * 100
                st.metric("Avg Entity Accuracy", f"{avg_entity:.2f}%")
            with col3:
                avg_latency = df["latency_seconds"].mean()
                st.metric("Avg Latency", f"{avg_latency:.3f}s")
            
            st.divider()
            
            # Charts
            st.markdown("### 📉 Performance Charts")
            chart_col1, chart_col2 = st.columns(2)
            
            # WER by model
            with chart_col1:
                wer = df.groupby("model_name")["wer"].mean().reset_index()
                fig = px.bar(wer, x="model_name", y="wer", title="Mean WER by Model", color="model_name")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, key="wer_chart")
            
            # Entity accuracy
            with chart_col2:
                ent = df.groupby("model_name")["locality_correct"].mean().reset_index()
                ent["accuracy_pct"] = ent["locality_correct"] * 100
                fig2 = px.bar(ent, x="model_name", y="accuracy_pct", title="Locality Entity Accuracy (%)", color="model_name")
                fig2.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig2, key="entity_chart")
            
            # Latency boxplot
            st.markdown("### ⏱️ Latency Analysis")
            fig3 = px.box(df, x="model_name", y="latency_seconds", title="Inference Latency Distribution")
            fig3.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig3, key="latency_chart")
            
            # Leaderboard
            st.markdown("### 🏆 Leaderboard")
            leaderboard = df.groupby("model_name").agg(mean_wer=("wer","mean"), entity_acc=("locality_correct","mean"), mean_latency=("latency_seconds","mean")).reset_index()
            leaderboard = leaderboard.sort_values(["entity_acc","mean_wer"], ascending=[False,True])
            leaderboard["Rank"] = range(1, len(leaderboard) + 1)
            leaderboard = leaderboard[["Rank", "model_name", "mean_wer", "entity_acc", "mean_latency"]]
            leaderboard.columns = ["Rank", "Model", "Avg WER", "Entity Accuracy", "Avg Latency (s)"]
            st.dataframe(leaderboard.round(4), hide_index=True)

with tab2:
    st.header("🔊 Audio Explorer")
    # sample selector
    gt_path = DATA_DIR / "metadata" / "ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        sample = st.selectbox("Select sample to explore", options=gt["filename"].tolist(), help="Choose an audio sample to analyze")
        row = gt[gt["filename"] == sample].iloc[0]
        audio_path = DATA_DIR / "raw" / row["condition"] / sample
        
        # Info card
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            **📍 Locality:** {row['locality_name']}  
            **🎯 Reference Transcript:**  
            > {row['reference_transcript']}
            """)
        with col2:
            st.info(f"Condition: {row['condition']}")

        # Audio player
        st.markdown("### 🎧 Audio Playback")
        st.audio(str(audio_path), format="audio/wav")

        # Waveform
        st.markdown("### 📈 Waveform Analysis")
        y, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE)
        t = np.linspace(0, len(y)/sr, num=len(y))
        wave_fig = go.Figure(data=go.Scatter(x=t, y=y, mode='lines', line=dict(color='#00FFAA', width=1), fill='tozeroy', fillcolor='rgba(0, 255, 170, 0.1)'))
        wave_fig.update_layout(title='Waveform', xaxis_title='Time (s)', yaxis_title='Amplitude', template='plotly_dark', height=350)
        st.plotly_chart(wave_fig, key="audio_explorer_waveform")

        # Transcripts side-by-side
        st.markdown("### 📝 Model Transcriptions")
        if selected_models:
            cols = st.columns(len(selected_models))
            for i, m in enumerate(selected_models):
                try:
                    tdf = pd.read_csv(TRANSCRIPTS_DIR / f"{m}_transcripts.csv")
                    hyp = tdf[tdf["filename"] == sample]["hypothesis"].values
                    hyp_text = hyp[0] if len(hyp) else ""
                except Exception:
                    hyp_text = "(no transcript)"
                with cols[i]:
                    st.text_area(f"**{m}**", value=hyp_text, height=100, disabled=True, label_visibility="collapsed")
    else:
        st.warning("⚠️ No ground truth data found.")

with tab3:
    st.header("🔍 Error Analysis")
    if metrics_df.empty:
        st.info("ℹ️ No metrics to analyze.")
    else:
        # Filtering options
        col1, col2 = st.columns([2, 1])
        with col1:
            num_worst = st.slider("Show worst N samples", min_value=5, max_value=50, value=20, key="worst_slider")
        with col2:
            st.write("")  # Spacing
        
        # Show worst samples
        st.markdown("### ❌ Worst Performing Samples")
        worst = metrics_df.sort_values("wer", ascending=False).head(num_worst)
        worst_display = worst[["filename","model_name","wer","locality_name","condition"]].copy()
        worst_display.columns = ["Filename", "Model", "WER", "Locality", "Condition"]
        st.dataframe(worst_display, hide_index=True)
        
        # Show failure analysis summary
        st.markdown("### 📊 Failure Mode Summary")
        st.info("💡 Detailed failure analysis (code-switch rate, hallucination rates, etc.) is available in the full report generated by the pipeline.")

with tab4:
    st.header("Live Testing")
    st.markdown("**Upload an audio file to run live transcription against selected models.**")
    
    # Two-column layout for upload and info
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded = st.file_uploader("🎤 Upload audio", type=["wav","mp3","m4a"], accept_multiple_files=False, help="Supported formats: WAV, MP3, M4A")
    
    # Process uploaded file
    if uploaded is not None:
        data = uploaded.read()
        tmp = ROOT / "data" / "live_uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / uploaded.name
        target.write_bytes(data)
        
        # Load and analyze audio
        try:
            y, sr = librosa.load(str(target), sr=TARGET_SAMPLE_RATE)
            duration = librosa.get_duration(y=y, sr=sr)
            file_size_mb = len(data) / (1024 * 1024)
            
            # Display audio info
            with col_info:
                st.info(f"""
                📊 **Audio Info**
                - Duration: {duration:.2f}s
                - Sample Rate: {sr} Hz
                - Size: {file_size_mb:.2f} MB
                """)
            
            # Audio player
            st.markdown("### 🎧 Audio Playback")
            st.audio(str(target), format="audio/wav")
            
            # Waveform visualization
            st.markdown("### 📈 Waveform")
            t = np.linspace(0, len(y)/sr, num=len(y))
            wave_fig = go.Figure(data=go.Scatter(
                x=t, y=y, mode='lines', 
                line=dict(color='#00FFAA', width=1),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 170, 0.1)',
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
            ))
            wave_fig.update_layout(
                title='Audio Waveform',
                xaxis_title='Time (s)',
                yaxis_title='Amplitude',
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified'
            )
            st.plotly_chart(wave_fig, key="live_waveform")
            
            # Model selection and transcription
            st.markdown("### 🤖 Transcription")
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    live_models = st.multiselect(
                        "Select models to run", 
                        options=list(models.keys()), 
                        default=list(models.keys()),
                        help="Choose which ASR models to use for transcription"
                    )
                with col2:
                    st.write("")  # Spacing
                    st.write("")
                    transcribe_btn = st.button("▶️ Transcribe", use_container_width=False, key="transcribe_btn")
            
            if transcribe_btn and live_models:
                progress = st.progress(0, text="Starting transcription...")
                results = []
                
                for idx, name in enumerate(live_models):
                    model = models.get(name)
                    progress_text = f"Processing {name}... ({idx+1}/{len(live_models)})"
                    progress.progress(int((idx+1)/len(live_models)*100), text=progress_text)
                    
                    if not model:
                        results.append((name, "❌ Model not available"))
                        continue
                    try:
                        r = model.transcribe(target)
                        results.append((name, r.transcript))
                    except Exception as exc:
                        results.append((name, f"❌ ERROR: {exc}"))
                
                # Display results
                st.markdown("### 📝 Results")
                for name, txt in results:
                    status_icon = "✅" if not txt.startswith("❌") else "⚠️"
                    with st.expander(f"{status_icon} {name}", expanded=True):
                        st.text_area("Transcription", value=txt, height=100, disabled=True, label_visibility="collapsed", key=f"result_{name}")
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    🎤 <b>ASR Shootout</b> — Interactive dashboard for benchmarking speech recognition models  
    <br>
    <small>Built with ❤️ using Streamlit • Last updated: May 2024</small>
    </div>
    """,
    unsafe_allow_html=True
)

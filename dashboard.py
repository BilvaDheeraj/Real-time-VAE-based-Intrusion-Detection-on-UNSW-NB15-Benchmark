import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from datetime import datetime
from collections import Counter
from data_loader import DataLoader, sliding_window_stream
from vae_model import VAE, train_model
from inference import InferenceEngine
import torch
import os

st.set_page_config(
    page_title="CyberShield IDS | UNSW-NB15",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
    .stApp { background-color: #060b14; }
    .main .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 100%; }
    h1 { font-family: "Share Tech Mono", monospace !important; letter-spacing: 4px; }
    h2, h3, h4 { font-family: "Rajdhani", sans-serif !important; letter-spacing: 1px; color: #7eb8d4 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0b1a2e 0%, #122035 100%);
        border: 1px solid #1a3a5c; border-radius: 10px;
        padding: 14px 18px !important;
        box-shadow: 0 0 18px rgba(0,210,255,0.07);
    }
    [data-testid="stMetricLabel"] p { color: #5a8fa8 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 2px; }
    [data-testid="stMetricValue"] { color: #00f5c0 !important; font-size: 1.55rem !important; font-weight: 700 !important; font-family: "Share Tech Mono", monospace !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #04090f 0%, #081525 100%) !important; border-right: 1px solid #1a3a5c; }
    .stButton > button { background: linear-gradient(90deg,#0080ff,#00c8ff); color:white; border:none; border-radius:8px; padding:10px 28px; font-weight:700; letter-spacing:2px; font-size:0.95rem; transition:all 0.3s; width:100%; text-transform:uppercase; }
    .stButton > button:hover { transform:translateY(-2px); box-shadow:0 6px 22px rgba(0,180,255,0.45); }
    .stTabs [data-baseweb="tab-list"] { background:#0b1a2e; border-radius:8px; border:1px solid #1a3a5c; }
    .stTabs [aria-selected="true"] { color:#00f5c0 !important; background:#122035; border-radius:6px; }
    hr { border-color: #1a3a5c !important; }
    .cyber-alert-critical { background:linear-gradient(90deg,rgba(255,30,30,0.12),rgba(255,30,30,0.03)); border-left:3px solid #ff3333; border-radius:4px; padding:8px 14px; margin:3px 0; font-family:"Share Tech Mono",monospace; font-size:0.78rem; color:#ff8888; animation:glowPulse 2.5s ease-in-out infinite; }
    .cyber-alert-high { background:linear-gradient(90deg,rgba(255,140,0,0.10),rgba(255,140,0,0.02)); border-left:3px solid #ff8c00; border-radius:4px; padding:8px 14px; margin:3px 0; font-family:"Share Tech Mono",monospace; font-size:0.78rem; color:#ffb347; }
    .cyber-alert-normal { background:rgba(0,200,100,0.05); border-left:3px solid #00cc66; border-radius:4px; padding:5px 14px; margin:2px 0; font-family:"Share Tech Mono",monospace; font-size:0.72rem; color:#55dd88; }
    @keyframes glowPulse { 0%,100%{box-shadow:0 0 0 rgba(255,51,51,0);} 50%{box-shadow:0 0 10px rgba(255,51,51,0.35);} }
    .scanline { height:1px; overflow:hidden; position:relative; margin:6px 0; }
    .scanline::after { content:""; display:block; height:2px; background:linear-gradient(90deg,transparent,#00f5c0,transparent); animation:sweep 2.2s linear infinite; }
    @keyframes sweep { from{transform:translateX(-100%);} to{transform:translateX(100%);} }
    .badge-critical { display:inline-block; background:#ff3333; color:#fff; padding:2px 9px; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:1px; }
    .badge-high { display:inline-block; background:#ff8c00; color:#fff; padding:2px 9px; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:1px; }
    .badge-medium { display:inline-block; background:#ffd700; color:#000; padding:2px 9px; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:1px; }
    .badge-safe { display:inline-block; background:#00cc66; color:#000; padding:2px 9px; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:1px; }
    .stProgress > div > div > div > div { background:linear-gradient(90deg,#00f5c0,#0080ff) !important; }
</style>
""", unsafe_allow_html=True)

ATTACK_META = {
    "Normal":         {"icon": "G", "color": "#00cc66", "severity": "SAFE",     "badge": "badge-safe",     "desc": "Benign traffic - no threat"},
    "Generic":        {"icon": "O", "color": "#ff8c00", "severity": "HIGH",     "badge": "badge-high",     "desc": "Cipher/brute-force attack"},
    "Exploits":       {"icon": "R", "color": "#ff2244", "severity": "CRITICAL", "badge": "badge-critical", "desc": "Known CVE exploitation"},
    "Fuzzers":        {"icon": "Y", "color": "#ffd700", "severity": "MEDIUM",   "badge": "badge-medium",   "desc": "Malformed packet fuzzing"},
    "DoS":            {"icon": "R", "color": "#ff0000", "severity": "CRITICAL", "badge": "badge-critical", "desc": "Denial of Service attack"},
    "Reconnaissance": {"icon": "Y", "color": "#ffaa00", "severity": "MEDIUM",   "badge": "badge-medium",   "desc": "Port scan / host discovery"},
    "Analysis":       {"icon": "O", "color": "#ff6600", "severity": "HIGH",     "badge": "badge-high",     "desc": "Vulnerability probing"},
    "Backdoor":       {"icon": "R", "color": "#cc0000", "severity": "CRITICAL", "badge": "badge-critical", "desc": "Covert remote access"},
    "Shellcode":      {"icon": "R", "color": "#dd0022", "severity": "CRITICAL", "badge": "badge-critical", "desc": "Executable code injection"},
    "Worms":          {"icon": "R", "color": "#ff1133", "severity": "CRITICAL", "badge": "badge-critical", "desc": "Self-propagating malware"},
}

def get_meta(cat):
    return ATTACK_META.get(str(cat).strip(),
        {"icon": "?", "color": "#888", "severity": "UNKNOWN", "badge": "badge-safe", "desc": "Unknown"})

now_str = datetime.now().strftime("%H:%M:%S  .  %d %b %Y")
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;padding:10px 0 4px 0;">
  <span style="font-size:3rem;line-height:1;">&#128737;</span>
  <div style="flex:1;">
    <h1 style="margin:0;padding:0;font-size:2rem;background:linear-gradient(90deg,#00f5c0,#00aaff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      CYBERSHIELD &middot; IDS
    </h1>
    <p style="margin:0;color:#375f7a;font-family:'Share Tech Mono',monospace;font-size:0.72rem;letter-spacing:3px;">
      REAL-TIME VAE INTRUSION DETECTION &nbsp;&middot;&nbsp; UNSW-NB15 BENCHMARK
    </p>
  </div>
  <div style="text-align:right;">
    <span style="color:#2d5a7a;font-family:'Share Tech Mono';font-size:0.65rem;letter-spacing:2px;">SYSTEM CLOCK</span><br>
    <span style="color:#00f5c0;font-family:'Share Tech Mono';font-size:0.95rem;">{now_str}</span>
  </div>
</div>
<div class="scanline"></div>
<hr style="margin:4px 0 10px 0;">
""", unsafe_allow_html=True)

_defaults = {
    "model": None, "threshold": 0.05,
    "X_test": None, "y_test": None,
    "attack_cat_test": None, "input_dim": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")
    st.markdown("**Dataset Path**")
    data_path  = st.text_input("Dataset Directory", "UNSW-NB15", label_visibility="collapsed")
    train_file = os.path.join(data_path, "UNSW_NB15_training-set.parquet")
    test_file  = os.path.join(data_path, "UNSW_NB15_testing-set.parquet")
    st.markdown("**Stream Settings**")
    window_size = st.slider("Window Size (packets)", 10, 200, 50)
    stride      = st.slider("Stride", 1, 50, 10)
    speed_ms    = st.slider("Simulation Speed (ms/step)", 50, 2000, 200)
    st.markdown("**Model Architecture**")
    latent_dim = st.slider("Latent Dimension", 2, 32, 10)
    epochs     = st.number_input("Training Epochs", min_value=1, max_value=100, value=5)
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=64)
    st.markdown("---")
    train_btn = st.button("Train New Model")
    st.markdown("---")
    if st.session_state.model is not None:
        st.success("Model Ready")
        st.metric("Anomaly Threshold", f"{st.session_state.threshold:.6f}")
        if st.session_state.input_dim:
            st.metric("Input Dimensions", st.session_state.input_dim)
    else:
        st.warning("No model trained yet")
    st.markdown("---")
    st.markdown("""
    <div style="color:#2d5a7a;font-size:0.7rem;font-family:'Share Tech Mono';line-height:1.8;">
    UNSW-NB15 Attack Classes:<br>
    [R] Exploits, DoS, Backdoor<br>
    [R] Shellcode, Worms<br>
    [O] Generic, Analysis<br>
    [Y] Fuzzers, Reconnaissance<br>
    [G] Normal
    </div>
    """, unsafe_allow_html=True)

if train_btn:
    progress = st.progress(0, text="Initializing pipeline...")
    try:
        progress.progress(10, text="Loading dataset files...")
        test_df_raw = pd.read_parquet(test_file)
        attack_cats = test_df_raw["attack_cat"].values
        st.session_state.attack_cat_test = attack_cats

        progress.progress(25, text="Preprocessing features...")
        dl = DataLoader(data_path)
        X_train, X_test, y_test = dl.load_and_preprocess(train_file, test_file)
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        input_dim = X_train.shape[1]
        st.session_state.input_dim = input_dim
        progress.progress(40, text=f"Building VAE (input_dim={input_dim}, latent={latent_dim})...")
        model = VAE(input_dim, latent_dim)

        progress.progress(50, text=f"Training VAE for {epochs} epochs...")
        model = train_model(model, X_train, epochs=int(epochs), batch_size=int(batch_size))
        st.session_state.model = model

        progress.progress(88, text="Computing anomaly threshold (95th pct)...")
        model.eval()
        with torch.no_grad():
            X_arr = X_train if isinstance(X_train, np.ndarray) else X_train.toarray()
            train_tensor = torch.FloatTensor(X_arr)
            recon, _, _ = model(train_tensor)
            errors = torch.mean((train_tensor - recon) ** 2, dim=1).numpy()
            st.session_state.threshold = float(np.percentile(errors, 95))

        progress.progress(100, text="Done!")
        time.sleep(0.4)
        progress.empty()

        c1, c2, c3 = st.columns(3)
        c1.success(f"Trained on {X_train.shape[0]:,} normal samples")
        c2.info(f"Threshold: {st.session_state.threshold:.6f}")
        c3.info(f"Test set: {X_test.shape[0]:,} samples")
    except Exception as e:
        progress.empty()
        st.error(f"Training failed: {e}")

if st.session_state.model is not None and st.session_state.X_test is not None:
    st.markdown("### Live Monitoring Console")
    col_toggle, col_status = st.columns([2, 3])
    with col_toggle:
        start_sim = st.checkbox("Start Live Simulation", value=False, key="start_sim")
    with col_status:
        st.markdown(
            f"<span style='color:#2d5a7a;font-family:Share Tech Mono;font-size:0.75rem;'>"
            f"Threshold: <b style='color:#00f5c0'>{st.session_state.threshold:.6f}</b> &nbsp;|&nbsp; "
            f"Test samples: <b style='color:#00f5c0'>{st.session_state.X_test.shape[0]:,}</b> &nbsp;|&nbsp; "
            f"Window: <b style='color:#00f5c0'>{window_size}</b> &nbsp;|&nbsp; "
            f"Stride: <b style='color:#00f5c0'>{stride}</b></span>",
            unsafe_allow_html=True
        )

    if start_sim:
        engine = InferenceEngine(st.session_state.model, st.session_state.threshold)
        stream = sliding_window_stream(st.session_state.X_test, window_size=window_size, stride=stride)
        a_cats = st.session_state.attack_cat_test

        kpi_cols = st.columns(5)
        st.markdown("---")

        col_chart, col_dist = st.columns([3, 2])
        with col_chart:
            st.markdown("#### Reconstruction Error Timeline")
            chart_ph = st.empty()
        with col_dist:
            st.markdown("#### Attack Type Distribution")
            dist_ph = st.empty()

        st.markdown("---")

        col_log, col_heat = st.columns([3, 2])
        with col_log:
            st.markdown("#### Live Event Log  (latest 25 windows)")
            log_ph = st.empty()
        with col_heat:
            st.markdown("#### Window Severity Map")
            heat_ph = st.empty()

        st.markdown("---")
        st.markdown("#### Threat Alert Feed")
        alert_ph = st.empty()

        loss_hist     = []
        ts_hist       = []
        anomaly_hist  = []
        severity_hist = []
        cat_hist      = []
        event_rows    = []
        alerts        = []
        cat_counter   = Counter()
        sim_start     = time.time()

        for batch_idx, batch_data in enumerate(stream):
            if not st.session_state.get("start_sim", False):
                break

            t0 = time.time()
            per_sample_errors = engine.predict(batch_data)
            inf_ms = (time.time() - t0) * 1000

            avg_err = float(np.mean(per_sample_errors))
            max_err = float(np.max(per_sample_errors))
            is_anom = avg_err > st.session_state.threshold
            elapsed = time.time() - sim_start

            if a_cats is not None:
                s_idx = batch_idx * stride
                e_idx = min(s_idx + window_size, len(a_cats))
                win_cats = a_cats[s_idx:e_idx]
                cat_counts = Counter(win_cats)
                primary_cat = cat_counts.most_common(1)[0][0]
                cat_counter.update(cat_counts)
                n_normal = cat_counts.get("Normal", 0)
                pct_attack = round((1 - n_normal / max(len(win_cats), 1)) * 100, 1)
            else:
                primary_cat = "Unknown"
                pct_attack  = 100.0 if is_anom else 0.0
                cat_counter.update({"Unknown": window_size})

            meta    = get_meta(primary_cat)
            sev     = meta["severity"]
            now_ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            loss_hist.append(avg_err)
            ts_hist.append(batch_idx)
            anomaly_hist.append(1 if is_anom else 0)
            severity_hist.append(sev)
            cat_hist.append(primary_cat)

            status_label = "ANOMALY" if is_anom else "NORMAL"
            event_rows.append({
                "Timestamp":       now_ts,
                "Window":          batch_idx + 1,
                "Attack Category": primary_cat,
                "Severity":        sev,
                "Avg Error":       round(avg_err, 7),
                "Max Error":       round(max_err, 7),
                "Threshold":       round(st.session_state.threshold, 7),
                "Pct Attack":      f"{pct_attack}%",
                "Status":          status_label,
                "Inf ms":          round(inf_ms, 2),
            })

            if is_anom and sev in ("CRITICAL", "HIGH"):
                alerts.insert(0, {
                    "ts": now_ts, "cat": primary_cat, "icon": meta["icon"],
                    "sev": sev, "badge": meta["badge"], "desc": meta["desc"],
                    "err": avg_err, "pct": pct_attack,
                })
                alerts = alerts[:12]

            total_anom  = sum(anomaly_hist)
            anom_rate   = total_anom / max(len(anomaly_hist), 1) * 100
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

            kpi_vals = [
                ("Windows Processed", f"{batch_idx+1:,}", None),
                ("Total Anomalies",   f"{total_anom:,}",  f"+{anomaly_hist[-1]}"),
                ("Anomaly Rate",      f"{anom_rate:.1f}%", None),
                ("Current Error",     f"{avg_err:.5f}",    f"thr={st.session_state.threshold:.4f}"),
                ("Current Status",    "CRITICAL" if is_anom else "NORMAL", sev),
            ]
            for col, (label, val, delta) in zip(kpi_cols, kpi_vals):
                with col:
                    st.metric(label, val, delta)

            chart_df = pd.DataFrame({
                "Window":    ts_hist,
                "Error":     loss_hist,
                "Threshold": [st.session_state.threshold] * len(ts_hist),
                "IsAnomaly": ["Anomaly" if a else "Normal" for a in anomaly_hist],
            })

            line_chart = alt.Chart(chart_df).mark_line(strokeWidth=2).encode(
                x=alt.X("Window:Q", title="Window Index",
                         axis=alt.Axis(labelColor="#5a8fa8", titleColor="#5a8fa8")),
                y=alt.Y("Error:Q", title="Reconstruction Error",
                         axis=alt.Axis(labelColor="#5a8fa8", titleColor="#5a8fa8")),
                color=alt.value("#00f5c0"),
                tooltip=[alt.Tooltip("Window:Q"), alt.Tooltip("Error:Q", format=".6f")]
            )
            thr_line = alt.Chart(chart_df).mark_rule(strokeDash=[6,4], strokeWidth=1.5).encode(
                y="Threshold:Q", color=alt.value("#ff3333"),
                tooltip=[alt.Tooltip("Threshold:Q", title="Threshold", format=".6f")]
            )
            anom_df = chart_df[chart_df["IsAnomaly"] == "Anomaly"]
            anom_pts = alt.Chart(anom_df).mark_point(filled=True, size=70, opacity=0.85).encode(
                x="Window:Q", y="Error:Q", color=alt.value("#ff3333"),
                tooltip=[alt.Tooltip("Window:Q"), alt.Tooltip("Error:Q", format=".6f")]
            )
            chart_ph.altair_chart(
                (line_chart + thr_line + anom_pts)
                .properties(height=270)
                .configure(background="#0b1a2e")
                .configure_view(strokeWidth=0, fill="#0b1a2e")
                .configure_axis(gridColor="#162840", domainColor="#1a3a5c"),
                width='stretch'
            )

            if cat_counter:
                dist_df = pd.DataFrame([
                    {"Category": k, "Count": v, "Color": get_meta(k)["color"]}
                    for k, v in cat_counter.items()
                ]).sort_values("Count", ascending=False)
                bar = alt.Chart(dist_df).mark_bar(
                    cornerRadiusTopRight=4, cornerRadiusBottomRight=4
                ).encode(
                    x=alt.X("Count:Q", title="Packet Count",
                             axis=alt.Axis(labelColor="#5a8fa8", titleColor="#5a8fa8")),
                    y=alt.Y("Category:N", sort="-x", title="",
                             axis=alt.Axis(labelColor="#a0d4f5")),
                    color=alt.Color("Category:N", legend=None,
                        scale=alt.Scale(
                            domain=list(ATTACK_META.keys()),
                            range=[ATTACK_META[k]["color"] for k in ATTACK_META]
                        )
                    ),
                    tooltip=["Category", "Count"]
                ).properties(height=270, background="#0b1a2e")\
                 .configure_view(strokeWidth=0, fill="#0b1a2e")\
                 .configure_axis(gridColor="#162840", domainColor="#1a3a5c")
                dist_ph.altair_chart(bar, width='stretch')

            log_df = pd.DataFrame(event_rows[-25:][::-1])
            log_ph.dataframe(log_df, width='stretch', hide_index=True, height=320)

            SEV_SCORE = {"SAFE": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": 0}
            heat_rows = severity_hist[-100:]
            heat_df = pd.DataFrame({
                "Window":   list(range(len(heat_rows))),
                "Severity": heat_rows,
                "Category": cat_hist[-len(heat_rows):],
            })
            heat_chart = alt.Chart(heat_df).mark_rect(height=30).encode(
                x=alt.X("Window:O", axis=None),
                color=alt.Color("Severity:N",
                    legend=alt.Legend(orient="bottom", labelColor="#5a8fa8", titleColor="#5a8fa8"),
                    scale=alt.Scale(
                        domain=["SAFE","MEDIUM","HIGH","CRITICAL"],
                        range=["#00cc66","#ffd700","#ff8c00","#ff3333"]
                    )
                ),
                tooltip=["Window:O", "Category", "Severity"]
            ).properties(height=60)

            spark_df = pd.DataFrame({
                "Window": list(range(len(loss_hist[-100:]))),
                "Error":  loss_hist[-100:],
                "Thr":    [st.session_state.threshold] * len(loss_hist[-100:]),
            })
            spark = alt.Chart(spark_df).mark_area(opacity=0.25, color="#00f5c0").encode(
                x=alt.X("Window:Q", axis=None),
                y=alt.Y("Error:Q", axis=alt.Axis(labelColor="#5a8fa8", titleColor="#5a8fa8", title="Error")),
            ).properties(height=130)
            spark_thr = alt.Chart(spark_df).mark_line(
                color="#ff3333", strokeDash=[4,3], strokeWidth=1
            ).encode(x="Window:Q", y="Thr:Q")

            heat_ph.altair_chart(
                alt.vconcat(spark + spark_thr, heat_chart, spacing=2)
                .configure(background="#0b1a2e")
                .configure_view(strokeWidth=0, fill="#0b1a2e")
                .configure_axis(gridColor="#162840"),
                width='stretch'
            )

            if alerts:
                html_alerts = ""
                for a in alerts[:8]:
                    cls = "cyber-alert-critical" if a["sev"] == "CRITICAL" else "cyber-alert-high"
                    html_alerts += (
                        f'<div class="{cls}">'
                        f'[{a["ts"]}] &nbsp; [{a["icon"]}] &nbsp;'
                        f'<span class="{a["badge"]}">{a["sev"]}</span>'
                        f' &nbsp; <b>{a["cat"]}</b> &mdash; {a["desc"]}'
                        f' &nbsp;|&nbsp; Avg Error: <b>{a["err"]:.6f}</b>'
                        f' &nbsp;|&nbsp; Attack pkt: <b>{a["pct"]}%</b>'
                        f'</div>'
                    )
                alert_ph.markdown(html_alerts, unsafe_allow_html=True)
            else:
                alert_ph.markdown(
                    '<div class="cyber-alert-normal">[ NO THREATS ] - Monitoring... all traffic within normal baseline.</div>',
                    unsafe_allow_html=True
                )

            time.sleep(speed_ms / 1000)

        st.success(
            f"Simulation complete - {len(loss_hist):,} windows | "
            f"{sum(anomaly_hist):,} anomalies | "
            f"Rate: {sum(anomaly_hist)/max(len(anomaly_hist),1)*100:.1f}%"
        )

        if cat_counter:
            st.markdown("### Post-Simulation Attack Summary")
            total_pkts = sum(cat_counter.values())
            summary = pd.DataFrame([{
                "Attack Category": k,
                "Severity":        get_meta(k)["severity"],
                "Packet Count":    v,
                "Share (%)":       f"{v/total_pkts*100:.2f}%",
                "Description":     get_meta(k)["desc"],
            } for k, v in cat_counter.most_common()])
            st.dataframe(summary, width='stretch', hide_index=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:70px 40px;background:linear-gradient(135deg,#0b1a2e,#122035);border-radius:14px;border:1px dashed #1a3a5c;margin:30px 0;">
      <div style="font-size:4.5rem;margin-bottom:18px;">&#128737;</div>
      <h2 style="color:#375f7a;font-family:'Share Tech Mono',monospace;letter-spacing:4px;">SYSTEM STANDBY</h2>
      <p style="color:#1e4060;font-family:'Share Tech Mono';font-size:0.8rem;letter-spacing:2px;margin-top:8px;">
        CONFIGURE PARAMETERS IN THE SIDEBAR &nbsp;&middot;&nbsp;
        CLICK &nbsp;<span style="color:#00f5c0;">TRAIN NEW MODEL</span>&nbsp; TO INITIALIZE
      </p>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FALL vs CONTROLE — Análise Comparativa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (light, clean) ─────────────────────────────────────────────────
st.markdown("""
<style>
/* Cabeçalho */
.main-header {
    background: linear-gradient(90deg, #1a73e8 0%, #0d47a1 100%);
    color: white;
    padding: 18px 28px;
    border-radius: 10px;
    margin-bottom: 24px;
}
.main-header h1 { font-size: 1.5rem; margin: 0; }
.main-header p  { margin: 4px 0 0; font-size: 0.88rem; opacity: 0.85; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-fall { background: #fce4e4; color: #c62828; border: 1px solid #ef9a9a; }
.badge-ctrl { background: #e0f7fa; color: #00695c; border: 1px solid #80cbc4; }
/* Métricas */
[data-testid="metric-container"] {
    background: #f4f6fa;
    border: 1px solid #e0e4ef;
    border-radius: 8px;
    padding: 12px !important;
}
/* Tabela */
.stat-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.stat-table th {
    background: #e8edf5;
    padding: 9px 12px;
    text-align: left;
    color: #374151;
    border-bottom: 2px solid #c7d2e7;
    white-space: nowrap;
}
.stat-table td { padding: 8px 12px; border-bottom: 1px solid #eef1f7; }
.stat-table tr:hover td { background: #f0f4fb; }
.sig-row td { background: #fff8e1 !important; }
.sig-row:hover td { background: #fff3cd !important; }
.fall-col { color: #c62828; font-weight: 600; }
.ctrl-col { color: #006064; font-weight: 600; }
.info-box {
    background: #e8f0fe;
    border-left: 4px solid #1a73e8;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #374151;
    margin-bottom: 16px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Colors ───────────────────────────────────────────────────────────────────
C_FALL = "#e53935"
C_CTRL = "#00897b"
C_DIFF = "#f59e0b"
C_FALL_BG = "rgba(229,57,53,0.12)"
C_CTRL_BG = "rgba(0,137,123,0.12)"

# ── Data loading ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

SKIP_LABELS = {"Mediana", "Q1 (25%)", "Q3 (75%)", "DP", "Média"}

@st.cache_data
def load_data():
    def load_individuals(path):
        df = pd.read_excel(path, sheet_name=0)
        df = df[~df.iloc[:, 0].isin(SKIP_LABELS)]
        df = df[df.iloc[:, 0].notna()]
        df = df[df.iloc[:, 0].apply(lambda x: isinstance(x, str))]
        return df.reset_index(drop=True)

    def load_curve(path):
        return pd.read_excel(path, sheet_name=0)

    def load_curve_metrics(path):
        return pd.read_excel(path, sheet_name=1)

    fall_ind  = load_individuals(DATA_DIR / "metricas_individuaisFALL.xlsx")
    ctrl_ind  = load_individuals(DATA_DIR / "metricas_individuaisCONTROLE.xlsx")
    fall_curv = load_curve(DATA_DIR / "resultante_grupoFALL.xlsx")
    ctrl_curv = load_curve(DATA_DIR / "resultante_grupoCONTROLE.xlsx")
    fall_mr   = load_curve_metrics(DATA_DIR / "resultante_grupoFALL.xlsx")
    ctrl_mr   = load_curve_metrics(DATA_DIR / "resultante_grupoCONTROLE.xlsx")
    return fall_ind, ctrl_ind, fall_curv, ctrl_curv, fall_mr, ctrl_mr

fall_ind, ctrl_ind, fall_curv, ctrl_curv, fall_mr, ctrl_mr = load_data()

# Phase boundaries (avg between groups)
p2_fall = float(fall_mr.iloc[0, 1])
p3_fall = float(fall_mr.iloc[0, 2])
p2_ctrl = float(ctrl_mr.iloc[0, 1])
p3_ctrl = float(ctrl_mr.iloc[0, 2])
P2 = (p2_fall + p2_ctrl) / 2
P3 = (p3_fall + p3_ctrl) / 2
N_FALL = len(fall_ind)
N_CTRL = len(ctrl_ind)

# Numeric metric columns (skip Pessoa)
NUM_COLS = [c for c in fall_ind.columns[1:] if fall_ind[c].dtype in [np.float64, np.int64, float, int]
            or pd.to_numeric(fall_ind[c], errors='coerce').notna().sum() > 3]

# ── Stats helpers ─────────────────────────────────────────────────────────────

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def effect_label(d):
    a = abs(d)
    if a < 0.2:  return "trivial"
    if a < 0.5:  return "pequeno"
    if a < 0.8:  return "médio"
    return "grande"

def bh_correction(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        k = order[i]
        adj[k] = min(prev, pvals[k] * n / (i + 1))
        prev = adj[k]
    return np.clip(adj, 0, 1)

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def trapz(x, y):
    return float(np.trapz(y, x))

def phase_mask(fase, s, e):
    return (fase >= s) & (fase < e)

# ── Shared plotly layout ──────────────────────────────────────────────────────
def base_layout(**kw):
    return dict(
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        font=dict(family="Arial, sans-serif", color="#374151", size=12),
        legend=dict(bgcolor="white", bordercolor="#e0e4ef", borderwidth=1),
        margin=dict(l=60, r=40, t=50, b=50),
        **kw,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>📊 Análise Comparativa — Resultante de Aceleração</h1>
  <p>Ciclos de movimento por fases normalizadas
    <span class="badge badge-fall">FALL n={N_FALL}</span>
    <span class="badge badge-ctrl">CONTROLE n={N_CTRL}</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Curvas Resultantes",
    "📊 Estatísticas por Métrica",
    "🔬 Análise Temporal (SPM)",
    "🧪 Análises Adicionais",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    fase    = fall_curv["Fase_norm"].values
    fm      = fall_curv["Média (m/s²)"].values
    fdp     = fall_curv["DP (m/s²)"].values
    fhi     = fall_curv["Média+DP (m/s²)"].values
    flo     = fall_curv["Média-DP (m/s²)"].values
    cm      = ctrl_curv["Média (m/s²)"].values
    cdp     = ctrl_curv["DP (m/s²)"].values
    chi     = ctrl_curv["Média+DP (m/s²)"].values
    clo     = ctrl_curv["Média-DP (m/s²)"].values
    diff    = cm - fm

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # SD bands
    fig.add_trace(go.Scatter(x=np.concatenate([fase, fase[::-1]]),
                             y=np.concatenate([fhi, flo[::-1]]),
                             fill="toself", fillcolor=C_FALL_BG,
                             line=dict(color="rgba(0,0,0,0)"),
                             name="FALL ±1DP", showlegend=True), secondary_y=False)
    fig.add_trace(go.Scatter(x=np.concatenate([fase, fase[::-1]]),
                             y=np.concatenate([chi, clo[::-1]]),
                             fill="toself", fillcolor=C_CTRL_BG,
                             line=dict(color="rgba(0,0,0,0)"),
                             name="CTRL ±1DP", showlegend=True), secondary_y=False)
    # Mean lines
    fig.add_trace(go.Scatter(x=fase, y=fm, name="FALL (média)",
                             line=dict(color=C_FALL, width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=fase, y=cm, name="CONTROLE (média)",
                             line=dict(color=C_CTRL, width=2.5)), secondary_y=False)
    # Difference
    fig.add_trace(go.Scatter(x=fase, y=diff, name="Diferença CTRL−FALL",
                             line=dict(color=C_DIFF, width=1.8, dash="dot")),
                  secondary_y=True)
    # Phase lines
    for xv, label in [(P2, "P1|P2"), (P3, "P2|P3")]:
        fig.add_vline(x=xv, line=dict(color="#9e9e9e", dash="dash", width=1.2))
        fig.add_annotation(x=xv, y=1, yref="paper", text=label,
                           showarrow=False, font=dict(size=10, color="#757575"),
                           yshift=10, xshift=4)

    fig.update_layout(
        **base_layout(height=430),
        xaxis_title="Fase normalizada",
        yaxis_title="Aceleração resultante (m/s²)",
        yaxis2_title="Diferença (m/s²)",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Diferença CTRL−FALL (m/s²)",
                     showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics cards ──
    r_val   = float(np.corrcoef(fm, cm)[0, 1])
    rmse_v  = float(np.sqrt(np.mean(diff**2)))
    area_v  = float(np.trapz(np.abs(diff), fase))
    max_d   = float(np.max(np.abs(diff)))
    max_d_f = float(fase[np.argmax(np.abs(diff))])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correlação (r) entre curvas", f"{r_val:.4f}")
    c2.metric("RMSE entre médias", f"{rmse_v:.4f} m/s²")
    c3.metric("Área entre curvas", f"{area_v:.4f} m/s²")
    c4.metric("Diferença máxima", f"{max_d:.4f} m/s²", help=f"Fase ≈ {max_d_f:.3f}")

    st.markdown("---")
    st.markdown("**Diferença por fase (P1 / P2 / P3)**")

    phases = [("P1", 0, P2), ("P2", P2, P3), ("P3", P3, 1.0)]
    rows = []
    for ph, s, e in phases:
        mask = phase_mask(fase, s, e)
        fm_p, cm_p = fm[mask], cm[mask]
        d_abs  = float(np.mean(cm_p) - np.mean(fm_p))
        d_rel  = d_abs / abs(float(np.mean(fm_p))) * 100 if np.mean(fm_p) != 0 else 0
        rms    = float(np.sqrt(np.mean((cm_p - fm_p)**2)))
        r_p    = float(np.corrcoef(fm_p, cm_p)[0, 1])
        rows.append(dict(Fase=ph,
                         **{"FALL média": f"{np.mean(fm_p):.4f}",
                            "CTRL média": f"{np.mean(cm_p):.4f}",
                            "Δ absoluta": f"{d_abs:+.4f}",
                            "Δ relativa %": f"{d_rel:+.1f}%",
                            "RMSE": f"{rms:.4f}",
                            "Correlação r": f"{r_p:.4f}"}))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="info-box">
    Comparação por métrica individual (n<sub>FALL</sub>={nf}, n<sub>CTRL</sub>={nc}).
    Teste de <strong>Mann-Whitney U</strong> (bicaudal). Tamanho de efeito: <strong>d de Cohen</strong>
    (trivial &lt;0.2 · pequeno 0.2–0.5 · médio 0.5–0.8 · grande &gt;0.8).
    Correção de <strong>Benjamini-Hochberg</strong> (FDR) aplicada para comparações múltiplas.
    </div>""".format(nf=N_FALL, nc=N_CTRL), unsafe_allow_html=True)

    @st.cache_data
    def compute_stats():
        results = []
        for col in NUM_COLS:
            fa = pd.to_numeric(fall_ind[col], errors="coerce").dropna().values
            ca = pd.to_numeric(ctrl_ind[col], errors="coerce").dropna().values
            if len(fa) < 3 or len(ca) < 3:
                continue
            U, p = stats.mannwhitneyu(fa, ca, alternative="two-sided")
            d = cohen_d(fa, ca)
            d_pct = (np.mean(ca) - np.mean(fa)) / abs(np.mean(fa)) * 100 if np.mean(fa) != 0 else 0
            results.append(dict(
                Métrica=col,
                _fall_mean=np.mean(fa), _fall_std=np.std(fa, ddof=1),
                _ctrl_mean=np.mean(ca), _ctrl_std=np.std(ca, ddof=1),
                _d_pct=d_pct, U=U, _p=p, _d=d,
                _fa=fa, _ca=ca,
            ))
        padj = bh_correction([r["_p"] for r in results])
        for r, pa in zip(results, padj):
            r["_padj"] = pa
        return results

    results = compute_stats()

    col_f, col_s = st.columns([3, 1])
    with col_f:
        st.markdown("##### Filtrar métricas")
    with col_s:
        filtro = st.selectbox("", ["Todas", "Significativas (p adj < 0.05)", "Efeito grande (|d| > 0.8)"],
                              label_visibility="collapsed")

    if filtro == "Significativas (p adj < 0.05)":
        shown = [r for r in results if r["_padj"] < 0.05]
    elif filtro == "Efeito grande (|d| > 0.8)":
        shown = [r for r in results if abs(r["_d"]) > 0.8]
    else:
        shown = results

    # Build HTML table
    eff_colors = {"trivial": "#9e9e9e", "pequeno": "#1976d2", "médio": "#f59e0b", "grande": "#e53935"}
    rows_html = ""
    for r in shown:
        sig = r["_padj"] < 0.05
        row_cls = 'class="sig-row"' if sig else ""
        d_color = "#e53935" if abs(r["_d"]) > 0.8 else "#f59e0b" if abs(r["_d"]) > 0.5 else "#1976d2" if abs(r["_d"]) > 0.2 else "#757575"
        d_sign  = "+" if r["_d_pct"] >= 0 else ""
        eff     = effect_label(r["_d"])
        ec      = eff_colors[eff]
        rows_html += f"""<tr {row_cls}>
          <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{r['Métrica']}">{r['Métrica']}</td>
          <td class="fall-col">{r['_fall_mean']:.3f} <small style="color:#999">±{r['_fall_std']:.3f}</small></td>
          <td class="ctrl-col">{r['_ctrl_mean']:.3f} <small style="color:#999">±{r['_ctrl_std']:.3f}</small></td>
          <td style="color:{'#00695c' if r['_d_pct']>=0 else '#c62828'}">{d_sign}{r['_d_pct']:.1f}%</td>
          <td>{r['U']:.0f}</td>
          <td>{r['_p']:.4f}</td>
          <td style="font-weight:600;color:{'#c62828' if r['_padj']<0.05 else '#757575'}">{sig_stars(r['_padj'])} {r['_padj']:.4f}</td>
          <td style="color:{d_color};font-weight:600">{r['_d']:.3f}</td>
          <td><span style="background:{ec}22;color:{ec};padding:2px 8px;border-radius:12px;font-size:0.78rem;font-weight:600">{eff}</span></td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto">
    <table class="stat-table">
      <thead><tr>
        <th>Métrica</th>
        <th>FALL (média±DP)</th>
        <th>CTRL (média±DP)</th>
        <th>Δ%</th>
        <th>U</th>
        <th>p bruto</th>
        <th>p adj (BH)</th>
        <th>d Cohen</th>
        <th>Efeito</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <p style="font-size:0.78rem;color:#9e9e9e;margin-top:8px">
    *** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05 &nbsp; Linhas amarelas = significativas após correção FDR
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SPM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div class="info-box">
    <strong>Análise Temporal Ponto-a-Ponto (SPM-like):</strong>
    Para cada ponto da fase normalizada, z-test comparando médias grupais usando os DPs como
    estimativa de variância: <em>z = (CTRL − FALL) / √(DP_CTRL²/n_CTRL + DP_FALL²/n_FALL)</em>.
    Threshold ±1.96 (α=0.05, bicaudal). Regiões em vermelho = diferença significativa.
    </div>""", unsafe_allow_html=True)

    se    = np.sqrt(cdp**2 / N_CTRL + fdp**2 / N_FALL)
    z_arr = np.where(se > 0, diff / se, 0.0)
    sig   = np.abs(z_arr) > 1.96
    d_pct = np.where(fm != 0, diff / np.abs(fm) * 100, 0.0)

    # Z-score chart
    bar_colors = [C_FALL if abs(z) > 1.96 else "#90caf9" for z in z_arr]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=fase, y=z_arr, marker_color=bar_colors,
                          name="z-score", showlegend=False))
    fig2.add_hline(y=1.96,  line=dict(color=C_DIFF, dash="dash", width=1.5),
                   annotation_text="z=+1.96", annotation_position="right")
    fig2.add_hline(y=-1.96, line=dict(color=C_DIFF, dash="dash", width=1.5),
                   annotation_text="z=−1.96", annotation_position="right")
    for xv, lbl in [(P2, "P1|P2"), (P3, "P2|P3")]:
        fig2.add_vline(x=xv, line=dict(color="#9e9e9e", dash="dash", width=1))
        fig2.add_annotation(x=xv, y=1.02, yref="paper", text=lbl, showarrow=False,
                            font=dict(size=9, color="#757575"), xshift=4)
    fig2.update_layout(**base_layout(height=360),
                       xaxis_title="Fase normalizada", yaxis_title="z-score")
    st.plotly_chart(fig2, use_container_width=True)

    # Significant regions
    regions, in_r, start_i, max_z = [], False, 0, 0.0
    for i, z in enumerate(z_arr):
        if abs(z) > 1.96:
            if not in_r:
                in_r, start_i, max_z = True, i, z
            elif abs(z) > abs(max_z):
                max_z = z
            if i == len(z_arr) - 1:
                regions.append((start_i, i, max_z))
        else:
            if in_r:
                regions.append((start_i, i - 1, max_z))
                in_r = False

    st.markdown("**Regiões com diferença significativa (|z| > 1.96)**")
    if not regions:
        st.info("Nenhuma região com diferença estatisticamente significativa detectada.")
    else:
        reg_rows = []
        for i, (si, ei, mz) in enumerate(regions):
            ph = "P1" if fase[si] < P2 else "P2" if fase[si] < P3 else "P3"
            reg_rows.append({
                "Região": f"#{i+1}",
                "Início (fase)": f"{fase[si]:.4f}",
                "Fim (fase)": f"{fase[ei]:.4f}",
                "Duração": f"{fase[ei]-fase[si]:.4f}",
                "z máx": f"{mz:.3f}",
                "Direção": "CTRL > FALL" if mz > 0 else "FALL > CTRL",
                "Fase": ph,
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Diferença absoluta e relativa ao longo da fase**")
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=fase, y=diff, name="Δ absoluta (m/s²)",
                              line=dict(color=C_DIFF, width=2),
                              fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"),
                   secondary_y=False)
    fig3.add_trace(go.Scatter(x=fase, y=d_pct, name="Δ relativa (%)",
                              line=dict(color="#7c3aed", width=1.5, dash="dot")),
                   secondary_y=True)
    for xv in [P2, P3]:
        fig3.add_vline(x=xv, line=dict(color="#9e9e9e", dash="dash", width=1))
    fig3.update_layout(**base_layout(height=300),
                       xaxis_title="Fase normalizada",
                       yaxis_title="Δ m/s²",
                       yaxis2_title="Δ%")
    fig3.update_yaxes(showgrid=False, secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXTRA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── Scatter/jitter by metric ──
    st.markdown("#### Dispersão Individual por Métrica")
    metric_sel = st.selectbox("Escolha a métrica", NUM_COLS, key="met_sel")

    fa_s = pd.to_numeric(fall_ind[metric_sel], errors="coerce").dropna().values
    ca_s = pd.to_numeric(ctrl_ind[metric_sel], errors="coerce").dropna().values

    rng = np.random.default_rng(42)
    fig4 = go.Figure()
    fig4.add_trace(go.Box(x=["FALL"] * len(fa_s), y=fa_s,
                          name="FALL", marker_color=C_FALL, boxmean=True,
                          boxpoints="all", jitter=0.4, pointpos=0,
                          marker=dict(size=7, opacity=0.7)))
    fig4.add_trace(go.Box(x=["CONTROLE"] * len(ca_s), y=ca_s,
                          name="CONTROLE", marker_color=C_CTRL, boxmean=True,
                          boxpoints="all", jitter=0.4, pointpos=0,
                          marker=dict(size=7, opacity=0.7)))
    fig4.update_layout(**base_layout(height=360),
                       yaxis_title=metric_sel, showlegend=False)
    col_chart, col_stats = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(fig4, use_container_width=True)
    with col_stats:
        U, p = stats.mannwhitneyu(fa_s, ca_s, alternative="two-sided")
        d = cohen_d(fa_s, ca_s)
        st.markdown("**Estatísticas**")
        st.dataframe(pd.DataFrame({
            "": ["n", "Média", "Mediana", "DP", "Min", "Max"],
            "FALL":     [len(fa_s), f"{np.mean(fa_s):.3f}", f"{np.median(fa_s):.3f}",
                         f"{np.std(fa_s,ddof=1):.3f}", f"{fa_s.min():.3f}", f"{fa_s.max():.3f}"],
            "CONTROLE": [len(ca_s), f"{np.mean(ca_s):.3f}", f"{np.median(ca_s):.3f}",
                         f"{np.std(ca_s,ddof=1):.3f}", f"{ca_s.min():.3f}", f"{ca_s.max():.3f}"],
        }), hide_index=True, use_container_width=True)
        st.markdown(f"""
        | Teste | Valor |
        |-------|-------|
        | Mann-Whitney U | {U:.0f} |
        | p-valor | {p:.4f} {sig_stars(p)} |
        | d Cohen | {d:.3f} ({effect_label(d)}) |
        """)

    st.markdown("---")

    # ── Radar ──
    st.markdown("#### Radar — Métricas da Resultante Grupal (normalizadas)")
    mr_h   = fall_mr.columns[1:]
    fv     = fall_mr.iloc[0, 1:].values.astype(float)
    cv     = ctrl_mr.iloc[0, 1:].values.astype(float)
    # Keep only positive/magnitude metrics for radar
    pos_idx = [i for i, h in enumerate(mr_h) if "AUC-" not in h]
    labels  = [str(mr_h[i]).replace(" (fase_norm)","").replace(" (m/s²)","").replace(" (m/s²·s)","")
               for i in pos_idx]
    fv_r   = np.abs(fv[pos_idx])
    cv_r   = np.abs(cv[pos_idx])
    maxv   = np.maximum(fv_r, cv_r)
    maxv   = np.where(maxv == 0, 1, maxv)
    fv_n   = fv_r / maxv
    cv_n   = cv_r / maxv

    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(r=np.append(fv_n, fv_n[0]),
                                   theta=labels + [labels[0]],
                                   fill="toself", name="FALL",
                                   line_color=C_FALL, fillcolor=C_FALL_BG))
    fig5.add_trace(go.Scatterpolar(r=np.append(cv_n, cv_n[0]),
                                   theta=labels + [labels[0]],
                                   fill="toself", name="CONTROLE",
                                   line_color=C_CTRL, fillcolor=C_CTRL_BG))
    fig5.update_layout(**base_layout(height=400),
                       polar=dict(bgcolor="#f9fafb",
                                  radialaxis=dict(visible=True, range=[0, 1],
                                                  gridcolor="#e0e4ef"),
                                  angularaxis=dict(gridcolor="#e0e4ef")))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── CV chart ──
    st.markdown("#### Variabilidade Intragrupal — Coeficiente de Variação (CV%)")
    cv_labels, cv_fall, cv_ctrl = [], [], []
    for col in NUM_COLS:
        fa_c = pd.to_numeric(fall_ind[col], errors="coerce").dropna().values
        ca_c = pd.to_numeric(ctrl_ind[col], errors="coerce").dropna().values
        if len(fa_c) < 3 or len(ca_c) < 3:
            continue
        mf, mc = abs(np.mean(fa_c)), abs(np.mean(ca_c))
        if mf < 0.01 or mc < 0.01:
            continue
        short = col.split(" ")[0] + " " + col.split("_")[-1].replace("(m/s²)","").replace("(m/s²·s)","").replace("(ms)","").replace("(fase_norm)","")
        cv_labels.append(short.strip())
        cv_fall.append(np.std(fa_c, ddof=1) / mf * 100)
        cv_ctrl.append(np.std(ca_c, ddof=1) / mc * 100)

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name="FALL", x=cv_labels, y=cv_fall,
                          marker_color=C_FALL, opacity=0.8))
    fig6.add_trace(go.Bar(name="CONTROLE", x=cv_labels, y=cv_ctrl,
                          marker_color=C_CTRL, opacity=0.8))
    fig6.update_layout(**base_layout(height=340),
                       barmode="group",
                       yaxis_title="CV (%)",
                       xaxis=dict(tickangle=-40, tickfont=dict(size=10)))
    st.plotly_chart(fig6, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='font-size:0.78rem;color:#9e9e9e;text-align:center'>"
    "FALL vs CONTROLE · Mann-Whitney U · Cohen's d · BH-FDR · SPM-like z-test"
    "</p>",
    unsafe_allow_html=True,
)

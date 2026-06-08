import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import find_peaks, correlate, correlation_lags
from pathlib import Path
import io

st.set_page_config(page_title="FALL vs CONTROLE", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1a73e8 0%, #0d47a1 100%);
    color: white; padding: 18px 28px; border-radius: 10px; margin-bottom: 24px;
}
.main-header h1 { font-size: 1.5rem; margin: 0; }
.main-header p  { margin: 4px 0 0; font-size: 0.88rem; opacity: 0.85; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-left: 8px;
}
.badge-fall { background: #fce4e4; color: #c62828; border: 1px solid #ef9a9a; }
.badge-ctrl { background: #e0f7fa; color: #00695c; border: 1px solid #80cbc4; }
.info-box {
    background: #e8f0fe; border-left: 4px solid #1a73e8; border-radius: 4px;
    padding: 12px 16px; font-size: 0.85rem; color: #374151;
    margin-bottom: 16px; line-height: 1.6;
}
.stat-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.stat-table th {
    background: #e8edf5; padding: 9px 12px; text-align: left;
    color: #374151; border-bottom: 2px solid #c7d2e7; white-space: nowrap;
}
.stat-table td { padding: 8px 12px; border-bottom: 1px solid #eef1f7; }
.stat-table tr:hover td { background: #f0f4fb; }
.sig-row td { background: #fff8e1 !important; }
.fall-col { color: #c62828; font-weight: 600; }
.ctrl-col { color: #006064; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
C_FALL    = "#e53935"; C_CTRL    = "#00897b"; C_DIFF    = "#f59e0b"
C_FALL_BG = "rgba(229,57,53,0.12)"; C_CTRL_BG = "rgba(0,137,123,0.12)"
DATA_DIR  = Path(__file__).parent / "data"
META_COLS = {"SUJEITOS ", "SUJEITOS", "Grupo", "Device"}
SQ = 520  # square plot size

# ── Base helpers ──────────────────────────────────────────────────────────────
def base_layout(h=SQ, w=None, **kw):
    d = dict(paper_bgcolor="white", plot_bgcolor="#f9fafb",
             font=dict(family="Arial, sans-serif", color="#374151", size=12),
             legend=dict(bgcolor="white", bordercolor="#e0e4ef", borderwidth=1),
             margin=dict(l=60, r=40, t=50, b=50), height=h, **kw)
    if w: d["width"] = w
    return d

def sq_layout(**kw):
    """Square layout for scatter plots."""
    return base_layout(h=SQ, w=SQ,
                       xaxis=dict(constrain="domain"),
                       yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
                       **kw)

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (np.mean(a)-np.mean(b))/pooled if pooled > 0 else 0.0

def cohen_d_ci(d, n1, n2):
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2-2)))
    return d - 1.96*se, d + 1.96*se

def effect_label(d):
    a = abs(d)
    return "trivial" if a<0.2 else "pequeno" if a<0.5 else "médio" if a<0.8 else "grande"

def bh_correction(pvals):
    n = len(pvals)
    if n == 0: return []
    order = np.argsort(pvals); adj = np.empty(n); prev = 1.0
    for i in range(n-1,-1,-1):
        k = order[i]; adj[k] = min(prev, pvals[k]*n/(i+1)); prev = adj[k]
    return np.clip(adj, 0, 1)

def sig_stars(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

def add_phase_lines(fig, P2, P3):
    for xv, lbl in [(P2,"P1|P2"),(P3,"P2|P3")]:
        fig.add_vline(x=xv, line=dict(color="#9e9e9e",dash="dash",width=1.2))
        fig.add_annotation(x=xv, y=1, yref="paper", text=lbl,
                           showarrow=False, font=dict(size=10,color="#757575"),
                           yshift=10, xshift=4)

def phase_mask(fase, s, e): return (fase>=s) & (fase<e)

# ── Advanced stats helpers ────────────────────────────────────────────────────
def compute_roc(scores_g2, scores_g1):
    """ROC: g2=positive (FALL), g1=negative (CTRL). Returns fpr, tpr, auc, opt_thresh, sens, spec."""
    all_sc = np.concatenate([scores_g2, scores_g1])
    labs   = np.concatenate([np.ones(len(scores_g2)), np.zeros(len(scores_g1))])
    thres  = np.sort(np.unique(all_sc))[::-1]
    n_pos, n_neg = len(scores_g2), len(scores_g1)
    fprs, tprs = [0.0], [0.0]
    for t in thres:
        pred = all_sc >= t
        tprs.append(np.sum(pred & (labs==1)) / n_pos)
        fprs.append(np.sum(pred & (labs==0)) / n_neg)
    fprs.append(1.0); tprs.append(1.0)
    fprs = np.array(fprs); tprs = np.array(tprs)
    idx  = np.argsort(fprs); fprs, tprs = fprs[idx], tprs[idx]
    auc  = float(np.trapezoid(tprs, fprs) if hasattr(np, 'trapezoid') else np.trapz(tprs, fprs))
    if auc < 0.5:
        auc = 1 - auc; fprs, tprs = 1-fprs[::-1], 1-tprs[::-1]
    j = tprs - fprs; oi = np.argmax(j)
    ti = max(0, oi-1)
    opt_thresh = float(thres[ti]) if ti < len(thres) else float(thres[-1])
    return fprs, tprs, auc, opt_thresh, float(tprs[oi]), float(1-fprs[oi])

def pca_2d(X):
    mu = X.mean(0); sd = X.std(0, ddof=1); sd[sd<1e-10] = 1
    Xs = (X - mu) / sd
    try:
        _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    except Exception:
        return None, None, None
    var_exp = S**2 / np.sum(S**2)
    return Xs @ Vt.T, var_exp, Vt

def lda_2g(X, y):
    """LDA 2 groups. Returns projection, weights, threshold, training accuracy."""
    try:
        m0 = X[y==0].mean(0); m1 = X[y==1].mean(0)
        Sw = sum((X[y==c]-m).T @ (X[y==c]-m) for c,m in [(0,m0),(1,m1)])
        Sw += np.eye(Sw.shape[0])*1e-4
        w = np.linalg.solve(Sw, m1-m0); w /= np.linalg.norm(w)
        proj = X @ w
        thr  = (proj[y==0].mean() + proj[y==1].mean()) / 2
        acc  = np.mean((proj>thr).astype(int) == y) * 100
        return proj, w, thr, acc
    except Exception:
        return None, None, None, None

def permutation_pval(a, b, n=1000):
    obs,_ = stats.mannwhitneyu(a, b, alternative='two-sided')
    comb  = np.concatenate([a,b]); na = len(a); cnt = 0
    for _ in range(n):
        p = np.random.permutation(comb)
        s,_ = stats.mannwhitneyu(p[:na], p[na:], alternative='two-sided')
        if s <= obs: cnt += 1          # smaller U → more extreme
    return cnt / n

def bootstrap_d_ci(a, b, n=1000):
    ds = [cohen_d(np.random.choice(a,len(a),replace=True),
                  np.random.choice(b,len(b),replace=True)) for _ in range(n)]
    return float(np.percentile(ds,2.5)), float(np.percentile(ds,97.5))

# ── Data loaders ──────────────────────────────────────────────────────────────
SKIP_LABELS = {"Mediana", "Q1 (25%)", "Q3 (75%)", "DP", "Média"}

@st.cache_data
def load_embedded():
    fc = pd.read_excel(DATA_DIR/"resultante_grupoFALL.xlsx",    sheet_name=0)
    cc = pd.read_excel(DATA_DIR/"resultante_grupoCONTROLE.xlsx", sheet_name=0)
    fm = pd.read_excel(DATA_DIR/"resultante_grupoFALL.xlsx",    sheet_name=1)
    cm = pd.read_excel(DATA_DIR/"resultante_grupoCONTROLE.xlsx", sheet_name=1)
    def load_ind(path):
        df = pd.read_excel(path, sheet_name=0)
        df = df[~df.iloc[:,0].isin(SKIP_LABELS)]
        df = df[df.iloc[:,0].notna()]
        df = df[df.iloc[:,0].apply(lambda x: isinstance(x, str))]
        return df.reset_index(drop=True)
    fall_ind = load_ind(DATA_DIR/"metricas_individuaisFALL.xlsx")
    ctrl_ind = load_ind(DATA_DIR/"metricas_individuaisCONTROLE.xlsx")
    return fc, cc, fm, cm, fall_ind, ctrl_ind

def load_upload(file_bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return {s: xl.parse(s).rename(columns=str.strip) for s in xl.sheet_names}

def get_metric_cols(df):
    return [c for c in df.columns if c not in META_COLS]

def detect_groups(df):
    gs = [g.strip() for g in df["Grupo"].dropna().unique()]
    ctrl_kw = ["saud","control","ctrl"]
    g1 = next((g for g in gs if any(k in g.lower() for k in ctrl_kw)), gs[0])
    g2 = next((g for g in gs if g!=g1), gs[1] if len(gs)>1 else gs[0])
    return g1, g2

def filt_dev(df, key):
    return df[df["Device"].str.contains(key, case=False, na=False)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD RENDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def prep_data(df_sub, g1_name, g2_name):
    mc    = get_metric_cols(df_sub)
    g1_df = df_sub[df_sub["Grupo"].str.strip()==g1_name]
    g2_df = df_sub[df_sub["Grupo"].str.strip()==g2_name]
    return mc, g1_df, g2_df

def compute_results(mc, g1_df, g2_df):
    results = []
    for col in mc:
        a = pd.to_numeric(g1_df[col], errors="coerce").dropna().values
        b = pd.to_numeric(g2_df[col], errors="coerce").dropna().values
        if len(a)<3 or len(b)<3: continue
        U,p = stats.mannwhitneyu(a,b,alternative="two-sided")
        d = cohen_d(a,b); d_lo,d_hi = cohen_d_ci(d,len(a),len(b))
        d_pct = (np.mean(b)-np.mean(a))/abs(np.mean(a))*100 if np.mean(a)!=0 else 0
        results.append(dict(col=col,a=a,b=b,
                            am=np.mean(a),astd=np.std(a,ddof=1),
                            bm=np.mean(b),bstd=np.std(b,ddof=1),
                            d_pct=d_pct,U=U,p=p,d=d,d_lo=d_lo,d_hi=d_hi))
    if results:
        padj = bh_correction([r["p"] for r in results])
        for r,pa in zip(results,padj): r["padj"]=pa
    return results

# ── Section 1: Stats table ────────────────────────────────────────────────────
def render_stats_table(results, g1_s, g2_s, ks):
    eff_c = {"trivial":"#9e9e9e","pequeno":"#1976d2","médio":"#f59e0b","grande":"#e53935"}
    _,cs = st.columns([3,1])
    with cs:
        filt = st.selectbox("Filtrar",["Todas","Significativas (p adj<0.05)","Efeito grande (|d|>0.8)"],
                            label_visibility="collapsed", key=f"filt_{ks}")
    shown = ([r for r in results if r["padj"]<0.05] if filt=="Significativas (p adj<0.05)"
             else [r for r in results if abs(r["d"])>0.8] if filt=="Efeito grande (|d|>0.8)"
             else results)
    rows=""
    for r in shown:
        sig  = r["padj"]<0.05; rc = 'class="sig-row"' if sig else ""
        dc   = "#e53935" if abs(r["d"])>0.8 else "#f59e0b" if abs(r["d"])>0.5 else "#1976d2" if abs(r["d"])>0.2 else "#757575"
        ds   = "+" if r["d_pct"]>=0 else ""; eff=effect_label(r["d"]); ec=eff_c[eff]
        rows += f"""<tr {rc}>
          <td style="white-space:nowrap">{r['col']}</td>
          <td class="ctrl-col">{r['am']:.3f}<small style="color:#999"> ±{r['astd']:.3f}</small></td>
          <td class="fall-col">{r['bm']:.3f}<small style="color:#999"> ±{r['bstd']:.3f}</small></td>
          <td style="color:{'#00695c' if r['d_pct']>=0 else '#c62828'}">{ds}{r['d_pct']:.1f}%</td>
          <td>{r['U']:.0f}</td><td>{r['p']:.4f}</td>
          <td style="font-weight:600;color:{'#c62828' if r['padj']<0.05 else '#757575'}">{sig_stars(r['padj'])} {r['padj']:.4f}</td>
          <td style="color:{dc};font-weight:600">{r['d']:.3f}</td>
          <td><span style="background:{ec}22;color:{ec};padding:2px 7px;border-radius:12px;font-size:0.78rem;font-weight:600">{eff}</span></td>
        </tr>"""
    st.markdown(f"""<div style="overflow-x:auto"><table class="stat-table">
      <thead><tr><th>Métrica</th><th>{g1_s} (média±DP)</th><th>{g2_s} (média±DP)</th>
      <th>Δ%</th><th>U</th><th>p bruto</th><th>p adj (BH)</th><th>d Cohen</th><th>Efeito</th>
      </tr></thead><tbody>{rows}</tbody></table></div>
      <p style="font-size:0.78rem;color:#9e9e9e;margin-top:6px">*** p&lt;0.001 ** p&lt;0.01 * p&lt;0.05</p>
    """, unsafe_allow_html=True)

# ── Section 2: Boxplot ────────────────────────────────────────────────────────
def render_boxplot(results, g1_df, g2_df, g1_s, g2_s, mc, ks):
    met = st.selectbox("Métrica", mc, key=f"met_{ks}")
    r   = next((x for x in results if x["col"]==met), None)
    if r is None: return
    a, b = r["a"], r["b"]
    fig  = go.Figure()
    for vals, name, color in [(a,g1_s,C_CTRL),(b,g2_s,C_FALL)]:
        fig.add_trace(go.Box(x=[name]*len(vals), y=vals, name=name,
                             marker_color=color, boxmean=True, boxpoints="all",
                             jitter=0.4, pointpos=0, marker=dict(size=7,opacity=0.7)))
    fig.update_layout(**base_layout(h=SQ), yaxis_title=met, showlegend=False)
    c1,c2 = st.columns([3,2])
    with c1: st.plotly_chart(fig, use_container_width=False)
    with c2:
        st.markdown("**Estatísticas**")
        st.dataframe(pd.DataFrame({"":["n","Média","Mediana","DP","Min","Max"],
            g1_s:[str(len(a)),f"{np.mean(a):.3f}",f"{np.median(a):.3f}",f"{np.std(a,ddof=1):.3f}",f"{a.min():.3f}",f"{a.max():.3f}"],
            g2_s:[str(len(b)),f"{np.mean(b):.3f}",f"{np.median(b):.3f}",f"{np.std(b,ddof=1):.3f}",f"{b.min():.3f}",f"{b.max():.3f}"],
        }), hide_index=True, use_container_width=True)
        st.markdown(f"| | |\n|---|---|\n| Mann-Whitney U | {r['U']:.0f} |\n| p-valor | {r['p']:.4f} {sig_stars(r['p'])} |\n| d Cohen | {r['d']:.3f} ({effect_label(r['d'])}) |")

# ── Section 3: CV% ────────────────────────────────────────────────────────────
def render_cv_bar(results, g1_s, g2_s):
    labs,ca,cb=[],[],[]
    for r in results:
        if abs(r["am"])<0.001 or abs(r["bm"])<0.001: continue
        labs.append(r["col"]); ca.append(r["astd"]/abs(r["am"])*100); cb.append(r["bstd"]/abs(r["bm"])*100)
    fig = go.Figure()
    fig.add_trace(go.Bar(name=g1_s, x=labs, y=ca, marker_color=C_CTRL, opacity=0.85))
    fig.add_trace(go.Bar(name=g2_s, x=labs, y=cb, marker_color=C_FALL, opacity=0.85))
    fig.update_layout(**base_layout(h=420), barmode="group", yaxis_title="CV (%)",
                      xaxis=dict(tickangle=-40,tickfont=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

# ── 🔴 ROC + AUC ──────────────────────────────────────────────────────────────
def render_roc(results, g1_s, g2_s, ks):
    st.markdown("""<div class="info-box">
    Cada métrica é usada como classificador binário (FALL vs CTRL).
    <strong>AUC &gt; 0.7</strong> = clinicamente relevante.
    Ponto ótimo via índice de <strong>Youden (J = sensibilidade + especificidade − 1)</strong>.
    </div>""", unsafe_allow_html=True)

    # ROC plot — all metrics
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                             line=dict(dash="dash",color="#9e9e9e"),name="Acaso",showlegend=False))
    roc_rows=[]
    for r in results:
        fprs,tprs,auc,ot,sens,spec = compute_roc(r["b"],r["a"])
        roc_rows.append({"Métrica":r["col"],"AUC":auc,"Threshold ótimo":ot,
                         "Sensibilidade":sens,"Especificidade":spec,
                         "Relevante":"✓" if auc>0.7 else ""})
        color = f"rgba(229,57,53,{min(1,auc)})" if auc>0.7 else "#b0bec5"
        fig.add_trace(go.Scatter(x=fprs,y=tprs,mode="lines",name=f"{r['col']} (AUC={auc:.2f})",
                                 line=dict(color=color,width=1.8),
                                 hovertemplate=f"<b>{r['col']}</b><br>AUC={auc:.3f}<extra></extra>"))
    fig.update_layout(**sq_layout(title="Curvas ROC"),
                      xaxis_title="1 − Especificidade (FPR)",
                      yaxis_title="Sensibilidade (TPR)",
                      showlegend=False)
    c1,c2=st.columns([1,1])
    with c1: st.plotly_chart(fig, use_container_width=False)
    with c2:
        df_roc = pd.DataFrame(roc_rows).sort_values("AUC",ascending=False)
        df_roc["AUC"] = df_roc["AUC"].map("{:.3f}".format)
        df_roc["Sensibilidade"] = df_roc["Sensibilidade"].map("{:.2f}".format)
        df_roc["Especificidade"] = df_roc["Especificidade"].map("{:.2f}".format)
        df_roc["Threshold ótimo"] = df_roc["Threshold ótimo"].map("{:.3f}".format)
        st.dataframe(df_roc, use_container_width=True, hide_index=True)

# ── 🔴 Forest Plot ────────────────────────────────────────────────────────────
def render_forest(results, g1_s, g2_s):
    st.markdown("""<div class="info-box">
    <strong>d de Cohen</strong> com IC 95% (Hedges & Olkin).
    Linha vertical em 0 = sem efeito. Valores positivos = {g2_s} maior que {g1_s}.
    </div>""".replace("{g2_s}",g2_s).replace("{g1_s}",g1_s), unsafe_allow_html=True)
    sorted_r = sorted(results, key=lambda r: r["d"])
    labels   = [r["col"] for r in sorted_r]
    ds       = [r["d"] for r in sorted_r]
    d_lo     = [r["d_lo"] for r in sorted_r]
    d_hi     = [r["d_hi"] for r in sorted_r]
    colors   = [C_FALL if abs(d)>0.8 else C_DIFF if abs(d)>0.5 else C_CTRL if abs(d)>0.2 else "#9e9e9e" for d in ds]
    fig = go.Figure()
    for i,(lbl,d,lo,hi,col) in enumerate(zip(labels,ds,d_lo,d_hi,colors)):
        fig.add_trace(go.Scatter(x=[lo,hi],y=[i,i],mode="lines",
                                 line=dict(color=col,width=2),showlegend=False))
        fig.add_trace(go.Scatter(x=[d],y=[i],mode="markers",
                                 marker=dict(color=col,size=9,symbol="diamond"),
                                 name=lbl,showlegend=False,
                                 hovertemplate=f"<b>{lbl}</b><br>d={d:.3f} [{lo:.3f},{hi:.3f}]<extra></extra>"))
    fig.add_vline(x=0,line=dict(color="#374151",width=1.5))
    fig.add_vline(x=0.8,line=dict(color=C_FALL,dash="dot",width=1))
    fig.add_vline(x=-0.8,line=dict(color=C_FALL,dash="dot",width=1))
    fig.update_layout(**base_layout(h=max(350,len(results)*28)),
                      xaxis_title="d de Cohen",
                      yaxis=dict(tickvals=list(range(len(labels))),ticktext=labels,
                                 tickfont=dict(size=10)),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── 🔴 PCA ────────────────────────────────────────────────────────────────────
def render_pca(results, g1_df, g2_df, g1_s, g2_s, g1_name, g2_name, ks):
    st.markdown("""<div class="info-box">
    As métricas são reduzidas a 2 componentes principais. Cada ponto = 1 sujeito.
    Elipses = ±1DP. Barras = contribuição de cada métrica (loadings).
    </div>""", unsafe_allow_html=True)
    cols_use = [r["col"] for r in results if r["col"] in g1_df.columns and r["col"] in g2_df.columns]
    g1_X = g1_df[cols_use].apply(pd.to_numeric, errors="coerce").dropna()
    g2_X = g2_df[cols_use].apply(pd.to_numeric, errors="coerce").dropna()
    X    = pd.concat([g1_X, g2_X]).values
    y    = np.array([0]*len(g1_X)+[1]*len(g2_X))
    if X.shape[0] < 4 or X.shape[1] < 2:
        st.warning("Dados insuficientes para PCA."); return
    scores, var_exp, Vt = pca_2d(X)
    if scores is None: st.warning("Erro no cálculo da PCA."); return
    fig = go.Figure()
    for grp, col, name, label in [(0,C_CTRL,g1_s,g1_name),(1,C_FALL,g2_s,g2_name)]:
        idx  = y==grp; sc = scores[idx]
        # Scatter
        subj_col = next((c for c in ["SUJEITOS ","SUJEITOS"] if c in g1_df.columns), None)
        try:
            df_tmp = (g1_df if grp==0 else g2_df).reset_index(drop=True)
            hover = df_tmp[subj_col].values[:len(sc)] if subj_col and subj_col in df_tmp.columns else [f"S{i}" for i in range(len(sc))]
        except: hover=[f"S{i}" for i in range(len(sc))]
        fig.add_trace(go.Scatter(x=sc[:,0],y=sc[:,1],mode="markers",
                                 name=name,marker=dict(color=col,size=9,opacity=0.8),
                                 text=hover, hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f} PC2=%{y:.2f}<extra></extra>"))
        # 1SD ellipse
        if len(sc) > 2:
            theta = np.linspace(0,2*np.pi,60)
            mu_e  = sc.mean(0); cov_e = np.cov(sc.T)
            vals,vecs = np.linalg.eigh(cov_e)
            order = vals.argsort()[::-1]; vals,vecs = vals[order],vecs[:,order]
            a_e,b_e = np.sqrt(vals[0]),np.sqrt(vals[1])
            angle_e = np.arctan2(vecs[1,0],vecs[0,0])
            ex = mu_e[0]+a_e*np.cos(theta)*np.cos(angle_e)-b_e*np.sin(theta)*np.sin(angle_e)
            ey = mu_e[1]+a_e*np.cos(theta)*np.sin(angle_e)+b_e*np.sin(theta)*np.cos(angle_e)
            fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",showlegend=False,
                                     line=dict(color=col,width=1.5,dash="dot")))
    fig.update_layout(**sq_layout(title=f"PCA — PC1 ({var_exp[0]*100:.1f}%) vs PC2 ({var_exp[1]*100:.1f}%)"),
                      xaxis_title=f"PC1 ({var_exp[0]*100:.1f}%)",
                      yaxis_title=f"PC2 ({var_exp[1]*100:.1f}%)")
    c1,c2=st.columns([1,1])
    with c1: st.plotly_chart(fig, use_container_width=False)
    with c2:
        # Loadings bar for PC1
        load1 = Vt[0,:]
        srt   = np.argsort(np.abs(load1))[::-1][:10]
        fig2  = go.Figure(go.Bar(x=load1[srt], y=[cols_use[i] for i in srt],
                                 orientation="h",
                                 marker_color=[C_FALL if v>0 else C_CTRL for v in load1[srt]]))
        fig2.update_layout(**base_layout(h=SQ,w=SQ), title="Loadings PC1 (top 10)",
                           xaxis_title="Loading", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=False)

# ── 🟡 Heatmap de Correlação ──────────────────────────────────────────────────
def render_heatmap(results, g1_df, g2_df, g1_s, g2_s):
    st.markdown("""<div class="info-box">
    Correlação de Pearson entre métricas, calculada separadamente por grupo.
    Células quentes = métricas redundantes (correlacionadas). Azul = correlação inversa.
    </div>""", unsafe_allow_html=True)
    cols_use = [r["col"] for r in results]
    for df, name, color_scale in [(g1_df,g1_s,"Teal"),(g2_df,g2_s,"Reds")]:
        sub  = df[cols_use].apply(pd.to_numeric,errors="coerce").dropna(axis=1,how="all")
        corr = sub.corr().values
        lbls = list(sub.columns)
        fig  = go.Figure(go.Heatmap(z=corr, x=lbls, y=lbls, colorscale="RdBu_r",
                                    zmid=0, zmin=-1, zmax=1,
                                    text=np.round(corr,2),texttemplate="%{text}",
                                    textfont=dict(size=8), hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r=%{z:.3f}<extra></extra>"))
        n = len(lbls); sz = max(SQ, n*35)
        fig.update_layout(**base_layout(h=sz,w=sz), title=f"Correlação — {name}",
                          xaxis=dict(tickfont=dict(size=9),tickangle=-40),
                          yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=False)

# ── 🟡 Cluster Hierárquico ────────────────────────────────────────────────────
def render_cluster(results, g1_df, g2_df, g1_name, g2_name, g1_s, g2_s, ks):
    st.markdown("""<div class="info-box">
    Agrupamento hierárquico (Ward) dos sujeitos <strong>sem usar o label do grupo</strong>.
    Verifica se os dados se organizam naturalmente em FALL e CTRL.
    </div>""", unsafe_allow_html=True)
    cols_use = [r["col"] for r in results]
    sub_col  = next((c for c in ["SUJEITOS ","SUJEITOS"] if c in g1_df.columns), None)
    extra = [sub_col] if sub_col else []
    g1_X = g1_df[cols_use+extra].apply(lambda c: pd.to_numeric(c,errors="coerce") if c.name!=sub_col else c).dropna(subset=cols_use)
    g2_X = g2_df[cols_use+extra].apply(lambda c: pd.to_numeric(c,errors="coerce") if c.name!=sub_col else c).dropna(subset=cols_use)
    all_X  = pd.concat([g1_X,g2_X])
    X_mat  = all_X[cols_use].values.astype(float)
    true_y = np.array([0]*len(g1_X)+[1]*len(g2_X))
    labels_subj = all_X[sub_col].astype(str).values if sub_col and sub_col in all_X.columns else [str(i) for i in range(len(all_X))]
    # Standardize
    mu=X_mat.mean(0); sd=X_mat.std(0); sd[sd<1e-10]=1
    Xs=(X_mat-mu)/sd
    # Linkage
    try:
        Z = linkage(Xs, method="ward")
    except Exception:
        st.warning("Erro no cálculo do cluster."); return
    # Dendrogram via plotly ff
    lab_list = [f"{s} ({'FALL' if y==1 else 'CTRL'})" for s,y in zip(labels_subj,true_y)]
    try:
        dend = ff.create_dendrogram(Xs, orientation="left", labels=lab_list,
                                    colorscale=["#b0bec5"]*7)
        dend.update_layout(**base_layout(h=max(400,len(all_X)*22),w=700),
                           title="Dendrograma — Cluster Hierárquico (Ward)")
        st.plotly_chart(dend, use_container_width=False)
    except Exception:
        st.info("Dendrograma não pôde ser gerado. Exibindo assignação de clusters.")
    # Cluster assignment accuracy
    n_clust = st.slider("Número de clusters", 2, min(6,len(all_X)//2), 2, key=f"nc_{ks}")
    pred_y  = fcluster(Z, t=n_clust, criterion="maxclust") - 1
    # Match cluster labels to true labels (greedy)
    best_acc = 0
    for flip in [False,True]:
        p = 1-pred_y if flip else pred_y
        acc = np.mean(p==true_y)*100
        if acc>best_acc: best_acc=acc
    st.metric("Acurácia de recuperação dos grupos (sem supervisão)", f"{best_acc:.1f}%",
              help="Proporção de sujeitos corretamente agrupados comparado com os grupos reais")

# ── 🟡 Score Composto ─────────────────────────────────────────────────────────
def render_risk_score(results, g1_df, g2_df, g1_s, g2_s, g1_name, g2_name, ks):
    st.markdown("""<div class="info-box">
    Score composto ponderado por |d de Cohen|, construído com métricas significativas (p adj &lt; 0.05).
    Score mais alto = maior similaridade com o perfil FALL.
    </div>""", unsafe_allow_html=True)
    sig_r = [r for r in results if r.get("padj",1)<0.05]
    if not sig_r:
        st.info("Nenhuma métrica significativa (p adj < 0.05) para compor o score."); return
    thresh = st.slider("Mínimo |d| para incluir no score", 0.0, 2.0, 0.2, 0.1, key=f"sc_{ks}")
    sig_r  = [r for r in sig_r if abs(r["d"])>=thresh]
    if not sig_r:
        st.info("Nenhuma métrica atende ao critério de efeito mínimo."); return
    cols = [r["col"] for r in sig_r]
    weights = np.array([abs(r["d"]) for r in sig_r])
    weights /= weights.sum()
    def get_score(df):
        sub = df[cols].apply(pd.to_numeric,errors="coerce")
        # Normalize each col to [0,1] using pooled min/max
        return sub, df
    g1_sub = g1_df[cols].apply(pd.to_numeric,errors="coerce").dropna()
    g2_sub = g2_df[cols].apply(pd.to_numeric,errors="coerce").dropna()
    all_sub = pd.concat([g1_sub, g2_sub])
    mn = all_sub.min(); mx = all_sub.max(); rng = (mx-mn).replace(0,1)
    g1_n = (g1_sub-mn)/rng; g2_n = (g2_sub-mn)/rng
    # Direction: if d>0 (g2>g1), higher raw = more FALL-like
    for r in sig_r:
        if r["d"] < 0:  # g1>g2, so invert
            col = r["col"]
            g1_n[col] = 1 - g1_n[col]; g2_n[col] = 1 - g2_n[col]
    s1 = (g1_n * weights).sum(axis=1)
    s2 = (g2_n * weights).sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Box(x=[g1_s]*len(s1),y=s1,name=g1_s,marker_color=C_CTRL,
                         boxmean=True,boxpoints="all",jitter=0.4,pointpos=0))
    fig.add_trace(go.Box(x=[g2_s]*len(s2),y=s2,name=g2_s,marker_color=C_FALL,
                         boxmean=True,boxpoints="all",jitter=0.4,pointpos=0))
    fig.update_layout(**base_layout(h=SQ,w=SQ), yaxis_title="Fall Risk Score",showlegend=False)
    c1,c2=st.columns([1,1])
    with c1:
        st.plotly_chart(fig, use_container_width=False)
    with c2:
        st.markdown(f"**{len(sig_r)} métricas incluídas:**")
        wdf = pd.DataFrame({"Métrica":cols,"Peso (|d|)":weights,"d Cohen":[r['d'] for r in sig_r]})
        wdf["Peso (|d|)"] = wdf["Peso (|d|)"].map("{:.3f}".format)
        wdf["d Cohen"]    = wdf["d Cohen"].map("{:.3f}".format)
        st.dataframe(wdf, hide_index=True, use_container_width=True)
        if len(s1)>2 and len(s2)>2:
            U,p = stats.mannwhitneyu(s1,s2,alternative="two-sided")
            d   = cohen_d(s1.values,s2.values)
            st.markdown(f"| | |\n|---|---|\n| U | {U:.0f} |\n| p | {p:.4f} {sig_stars(p)} |\n| d | {d:.3f} ({effect_label(d)}) |")

# ── 🟢 LDA ────────────────────────────────────────────────────────────────────
def render_lda(results, g1_df, g2_df, g1_s, g2_s, ks):
    st.markdown("""<div class="info-box">
    Análise Discriminante Linear — maximiza a separação entre grupos.
    Calcula a combinação linear de métricas que melhor diferencia FALL de CTRL.
    </div>""", unsafe_allow_html=True)
    cols_use = [r["col"] for r in results]
    g1_X = g1_df[cols_use].apply(pd.to_numeric,errors="coerce").dropna()
    g2_X = g2_df[cols_use].apply(pd.to_numeric,errors="coerce").dropna()
    X    = np.vstack([g1_X.values, g2_X.values])
    y    = np.array([0]*len(g1_X)+[1]*len(g2_X))
    if X.shape[0]<6: st.warning("Poucos sujeitos para LDA."); return
    # Standardize
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-10]=1
    Xs=(X-mu)/sd
    proj, w, thr, acc = lda_2g(Xs, y)
    if proj is None: st.warning("Erro no cálculo da LDA."); return
    # Histogram of projections
    fig = go.Figure()
    for grp,col,name in [(0,C_CTRL,g1_s),(1,C_FALL,g2_s)]:
        fig.add_trace(go.Histogram(x=proj[y==grp],name=name,
                                   marker_color=col,opacity=0.7,nbinsx=15))
    fig.add_vline(x=float(thr),line=dict(color=C_DIFF,dash="dash",width=2),
                  annotation_text=f"Threshold={thr:.2f}",annotation_position="top")
    fig.update_layout(**base_layout(h=SQ,w=SQ), barmode="overlay",
                      xaxis_title="Score Discriminante",yaxis_title="Frequência",
                      title=f"LDA — Acurácia de treino: {acc:.1f}%")
    c1,c2=st.columns([1,1])
    with c1: st.plotly_chart(fig, use_container_width=False)
    with c2:
        # Top discriminant features
        abs_w = np.abs(w)
        srt   = np.argsort(abs_w)[::-1][:10]
        fig2  = go.Figure(go.Bar(x=w[srt],y=[cols_use[i] for i in srt],
                                 orientation="h",
                                 marker_color=[C_FALL if wv>0 else C_CTRL for wv in w[srt]]))
        fig2.update_layout(**base_layout(h=SQ,w=SQ), title="Vetor Discriminante (top 10)",
                           xaxis_title="Peso",yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=False)
    st.metric("Acurácia de treino (LDA)", f"{acc:.1f}%",
              help="Classificação no conjunto de treino — não é validação cruzada")

# ── 🟢 Bootstrap ──────────────────────────────────────────────────────────────
def render_bootstrap(results, ks):
    st.markdown("""<div class="info-box">
    <strong>Permutation test</strong> (1000×): embaralha os labels aleatoriamente e recalcula Mann-Whitney.
    p-valor = proporção de estatísticas permutadas mais extremas que a observada.<br>
    <strong>Bootstrap IC 95%</strong> para d de Cohen (1000× reamostragem com reposição).
    </div>""", unsafe_allow_html=True)
    n_boot = st.slider("Número de reamostras", 200, 2000, 1000, 200, key=f"nb_{ks}")
    if st.button("▶ Rodar Bootstrap/Permutação", key=f"btn_{ks}"):
        boot_rows=[]
        prog = st.progress(0, text="Calculando...")
        for i, r in enumerate(results):
            prog.progress((i+1)/len(results), text=f"Métrica {i+1}/{len(results)}: {r['col']}")
            try:
                p_perm = permutation_pval(r["a"],r["b"],n=n_boot)
                d_lo_b,d_hi_b = bootstrap_d_ci(r["a"],r["b"],n=n_boot)
            except Exception:
                p_perm,d_lo_b,d_hi_b = np.nan,np.nan,np.nan
            boot_rows.append({
                "Métrica":r["col"],
                "p (Mann-Whitney)":f"{r['p']:.4f}",
                "p (Permutação)":f"{p_perm:.4f}" if not np.isnan(p_perm) else "–",
                "d Cohen":f"{r['d']:.3f}",
                "IC 95% bootstrap":(f"[{d_lo_b:.3f}, {d_hi_b:.3f}]" if not np.isnan(d_lo_b) else "–"),
                "IC 95% analítico":f"[{r['d_lo']:.3f}, {r['d_hi']:.3f}]",
                "Sig. permut.":"✓" if (not np.isnan(p_perm) and p_perm<0.05) else "",
            })
        prog.empty()
        st.dataframe(pd.DataFrame(boot_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Clique em '▶ Rodar' para iniciar o cálculo (pode demorar alguns segundos).")

# ── MAIN render function ──────────────────────────────────────────────────────
def render_metrics_analysis(df_sub, g1_name, g2_name, key_suffix=""):
    mc, g1_df, g2_df = prep_data(df_sub, g1_name, g2_name)
    g1_s = g1_name.replace("GRUPO","").strip().title()
    g2_s = g2_name.replace("GRUPO","").strip().title()
    n1, n2 = len(g1_df), len(g2_df)
    st.markdown(f"""<div style="margin-bottom:12px">
      <span class="badge badge-ctrl">{g1_s} n={n1}</span>
      <span class="badge badge-fall">{g2_s} n={n2}</span>
    </div>""", unsafe_allow_html=True)
    results = compute_results(mc, g1_df, g2_df)
    if not results: st.warning("Dados insuficientes para análise."); return

    ks = key_suffix

    # ── Stats table (always visible)
    st.markdown("#### 📊 Comparação estatística por métrica")
    st.markdown("""<div class="info-box"><strong>Mann-Whitney U</strong> · <strong>d de Cohen</strong> ·
    Correção <strong>BH-FDR</strong>. Linhas amarelas = significativas.</div>""", unsafe_allow_html=True)
    render_stats_table(results, g1_s, g2_s, ks)

    st.markdown("---")
    st.markdown("#### 📦 Dispersão individual")
    render_boxplot(results, g1_df, g2_df, g1_s, g2_s, mc, ks)

    st.markdown("---")
    st.markdown("#### 📉 CV% intragrupal por métrica")
    render_cv_bar(results, g1_s, g2_s)
    st.markdown("---")
    st.markdown("#### 📝 Texto para Artigo — Comparação Estatística")
    txt_boxes(apa_m_stats(n1,n2,g1_s,g2_s,len(results)),
              apa_r_stats(results,g1_s,g2_s), ks+"_stat_txt")
    st.markdown("---")
    st.markdown("#### 📝 Texto para Artigo — Dispersão Individual")
    # boxplot text uses last selected metric from render_boxplot — use general description
    txt_boxes(apa_m_boxplot(n1,n2,g1_s,g2_s),
              "See the statistics reported above for the selected metric. "
              "Individual data points are displayed in the box plot above. "
              "Report values as: median [IQR] or mean ± SD per group.", ks+"_box_txt")

    # Análises avançadas ficam na aba dedicada (render_advanced_tab)


# ══════════════════════════════════════════════════════════════════════════════
# APA TEXT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
PHYS = {
    "tempo P1":    "the initiation phase duration, reflecting time to generate forward momentum and anterior trunk displacement prior to seat-off; shorter durations may indicate adequate anticipatory postural adjustments",
    "tempo P2":    "the transitional (seat-off) phase duration, corresponding to the unweighting period requiring adequate quadriceps force production; prolonged durations may reflect reduced lower-limb extension strength",
    "tempo P3":    "the stabilization phase duration following seat-off, reflecting the time to achieve upright balance; longer durations may indicate deficits in postural righting reactions",
    "tempo total": "the total sit-to-stand duration, a global index of movement speed and efficiency; slower performance is associated with reduced strength, balance, and fall risk",
    "range Z":     "the vertical acceleration range, reflecting the amplitude of vertical momentum generated during the phase; reduced range suggests diminished force production capacity",
    "acc max Z":   "the peak upward vertical acceleration, reflecting the maximal lower-limb extension impulse; lower values in fallers indicate reduced power during the rising phase",
    "acc min Z":   "the minimum vertical acceleration (braking), reflecting deceleration control; altered braking may compromise balance during the terminal phase",
    "jerk score Z":"the vertical jerk (rate of change of acceleration), inversely related to movement smoothness; higher jerk scores indicate less coordinated, more fragmented motor patterns associated with fall risk",
    "frequencia Z":"the dominant frequency of vertical acceleration, reflecting rhythmicity of vertical momentum transfer; alterations may reflect neuromuscular fatigue or compensatory strategies",
    "range ML":    "the mediolateral acceleration range, an indicator of lateral weight-shift amplitude; excessive or reduced lateral displacement may compromise lateral stability",
    "acc max ML":  "the peak mediolateral acceleration, reflecting the magnitude of lateral forces during balance transitions",
    "acc min ML":  "the minimum mediolateral acceleration, reflecting braking of lateral body sway; impaired braking may contribute to lateral instability",
    "jerk score ML":"the mediolateral jerk, inversely related to lateral movement smoothness; elevated values suggest less coordinated lateral weight transfer, potentially associated with postural instability",
    "frequencia ML":"the dominant frequency of mediolateral oscillations; higher frequencies may reflect compensatory tremor or lateral instability patterns",
}

def get_phys(metric):
    for key, val in PHYS.items():
        if key.lower() in metric.lower():
            return val
    return "a kinematic metric reflecting sit-to-stand movement quality and neuromuscular control"

def txt_boxes(methods, results_txt, ks):
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📝 Methods — APA Style**")
        st.text_area("", value=methods, height=280, key=f"m_{ks}", label_visibility="collapsed")
    with c2:
        st.markdown("**📊 Results + Physiological Interpretation — APA Style**")
        st.text_area("", value=results_txt, height=280, key=f"r_{ks}", label_visibility="collapsed")

# ── APA generators ────────────────────────────────────────────────────────────
def apa_m_stats(n1, n2, g1_s, g2_s, nm):
    return (f"Group differences in kinematic metrics between {g2_s} (n = {n2}) and {g1_s} "
            f"(n = {n1}) were assessed using the Mann-Whitney U test (Mann & Whitney, 1947), "
            f"a non-parametric alternative appropriate for non-normal distributions and small samples. "
            f"Effect sizes were estimated using Cohen's d (Cohen, 1988), classified as trivial (<0.2), "
            f"small (0.2–0.5), moderate (0.5–0.8), or large (>0.8). To control the familywise "
            f"error rate across {nm} simultaneous comparisons, p-values were adjusted using the "
            f"Benjamini-Hochberg false discovery rate (FDR) procedure (Benjamini & Hochberg, 1995). "
            f"Statistical significance was set at α = 0.05.")

def apa_r_stats(results, g1_s, g2_s):
    n_total = len(results)
    sig = sorted([r for r in results if r.get("padj",1)<0.05], key=lambda r: abs(r["d"]), reverse=True)
    if not sig:
        return (f"No statistically significant differences were observed between {g2_s} and "
                f"{g1_s} for any of the {n_total} kinematic metrics assessed following "
                f"Benjamini-Hochberg FDR correction (all p_adj > 0.05).")
    large = [r for r in sig if abs(r["d"])>=0.8]
    mod   = [r for r in sig if 0.5<=abs(r["d"])<0.8]
    small = [r for r in sig if abs(r["d"])<0.5]
    lines = [f"Statistically significant group differences were observed for {len(sig)} of "
             f"{n_total} kinematic metrics following BH-FDR correction."]
    for grp, label in [(large,"Large effects (|d| ≥ 0.8)"),(mod,"Moderate effects (0.5 ≤ |d| < 0.8)"),(small,"Small effects (|d| < 0.5)")]:
        if grp:
            lines.append(f"\n{label}:")
            for r in grp:
                direction = "higher" if r["bm"]>r["am"] else "lower"
                lines.append(f"  • {r['col']}: {g2_s} showed {direction} values "
                             f"({r['bm']:.3f} ± {r['bstd']:.3f} vs. {r['am']:.3f} ± {r['astd']:.3f}), "
                             f"U = {r['U']:.0f}, p_adj = {r['padj']:.3f}, "
                             f"d = {r['d']:.3f} [95% CI: {r['d_lo']:.3f}, {r['d_hi']:.3f}]. "
                             f"This reflects differences in {get_phys(r['col'])}.")
    lines.append(f"\nCollectively, these findings suggest that {g2_s} exhibit altered kinematic "
                 f"profiles during the sit-to-stand task, particularly in metrics related to "
                 f"momentum generation, movement smoothness, and postural stabilization, which "
                 f"may reflect underlying neuromuscular deficits associated with fall risk.")
    return "\n".join(lines)

def apa_m_roc(n1, n2, g1_s, g2_s):
    return (f"The discriminative capacity of each kinematic metric to classify participants as "
            f"fallers ({g2_s}, n = {n2}) or non-fallers ({g1_s}, n = {n1}) was evaluated using "
            f"Receiver Operating Characteristic (ROC) curve analysis (Hanley & McNeil, 1982). "
            f"The area under the ROC curve (AUC) was computed using the trapezoidal rule. "
            f"Optimal classification thresholds were determined by maximizing Youden's J statistic "
            f"(J = sensitivity + specificity − 1; Youden, 1950). AUC values ≥ 0.70 were considered "
            f"clinically acceptable and AUC ≥ 0.80 excellent (Hosmer & Lemeshow, 2000).")

def apa_r_roc(roc_data, g1_s, g2_s):
    if not roc_data: return "ROC analysis could not be completed due to insufficient data."
    srt = sorted(roc_data, key=lambda x: x["auc"], reverse=True)
    exc = [r for r in srt if r["auc"]>=0.80]; good = [r for r in srt if 0.70<=r["auc"]<0.80]
    n_rel = len(exc)+len(good)
    lines = [f"ROC analysis identified {n_rel} metric(s) with clinically relevant discriminative "
             f"capacity (AUC ≥ 0.70) for classifying {g2_s} from {g1_s}."]
    for grp, label in [(exc,"Excellent classifiers (AUC ≥ 0.80)"),(good,"Acceptable classifiers (0.70 ≤ AUC < 0.80)")]:
        if grp:
            lines.append(f"\n{label}:")
            for r in grp:
                lines.append(f"  • {r['col']}: AUC = {r['auc']:.3f}, sensitivity = {r['sens']:.2f}, "
                             f"specificity = {r['spec']:.2f}, optimal threshold = {r['threshold']:.3f}. "
                             f"Physiologically, this metric reflects {get_phys(r['col'])}, "
                             f"suggesting its utility as a fall-risk screening marker.")
    if srt:
        b = srt[0]
        lines.append(f"\nOverall, {b['col']} demonstrated the highest discriminative capacity "
                     f"(AUC = {b['auc']:.3f}), identifying it as the most informative single "
                     f"kinematic variable for fall-risk classification in this cohort.")
    return "\n".join(lines)

def apa_m_forest(n1, n2, g1_s, g2_s, nm):
    return (f"Effect sizes (Cohen's d) and 95% confidence intervals (CIs) were computed for all "
            f"{nm} kinematic metrics and presented in a forest plot (Borenstein et al., 2009). "
            f"CIs were estimated using the Hedges & Olkin (1985) approximation: "
            f"SE(d) = √[(n₁+n₂)/(n₁·n₂) + d²/2(n₁+n₂−2)], "
            f"where n₁ = {n1} ({g1_s}) and n₂ = {n2} ({g2_s}). "
            f"Metrics with 95% CI entirely excluding zero were considered to have statistically "
            f"robust effect estimates, independent of arbitrary p-value thresholds.")

def apa_r_forest(results, g1_s, g2_s):
    robust = [r for r in results if (r["d_lo"]>0 and r["d_hi"]>0) or (r["d_lo"]<0 and r["d_hi"]<0)]
    top3 = sorted(results, key=lambda r: abs(r["d"]), reverse=True)[:3]
    lines = [f"Forest plot analysis across {len(results)} kinematic metrics revealed {len(robust)} "
             f"metric(s) with 95% CIs entirely excluding zero, indicating robust effect estimates."]
    lines.append(f"\nThe three largest effect sizes observed were:")
    for r in top3:
        direction = "higher" if r["bm"]>r["am"] else "lower"
        lines.append(f"  • {r['col']}: d = {r['d']:.3f} [95% CI: {r['d_lo']:.3f}, {r['d_hi']:.3f}], "
                     f"{effect_label(r['d'])} effect. {g2_s} showed {direction} values, "
                     f"reflecting differences in {get_phys(r['col'])}.")
    lines.append(f"\nThe forest plot provides a comprehensive overview of effect magnitude and "
                 f"precision, facilitating identification of the most clinically meaningful "
                 f"kinematic differences between fallers and non-fallers during sit-to-stand.")
    return "\n".join(lines)

def apa_m_pca(n1, n2, g1_s, g2_s, nm):
    return (f"Principal Component Analysis (PCA) was performed on {nm} standardized kinematic "
            f"metrics from {n1+n2} participants to reduce dimensionality and examine multivariate "
            f"group structure (Jolliffe, 2002). Variables were standardized (mean = 0, SD = 1) "
            f"prior to analysis. Scores on PC1 and PC2 were plotted per participant, color-coded "
            f"by group ({g1_s}, n = {n1}; {g2_s}, n = {n2}). Ellipses represent ±1 SD of the "
            f"group distribution. Component loadings were examined to identify the kinematic "
            f"variables most strongly contributing to each component.")

def apa_r_pca(pca_var, top_loadings, g1_s, g2_s):
    if pca_var is None: return "PCA could not be computed due to insufficient data."
    lines = [f"PCA revealed that PC1 and PC2 collectively explained {(pca_var[0]+pca_var[1])*100:.1f}% "
             f"of total variance (PC1: {pca_var[0]*100:.1f}%; PC2: {pca_var[1]*100:.1f}%)."]
    if top_loadings:
        lines.append(f"\nThe three metrics with highest absolute loadings on PC1 were:")
        for metric, loading in top_loadings:
            lines.append(f"  • {metric} (loading = {loading:.3f}): reflects {get_phys(metric)}.")
    lines.append(f"\nVisual inspection of the PC score plot revealed partial separation between "
                 f"{g2_s} and {g1_s} clusters, indicating that sit-to-stand kinematics carry "
                 f"multivariate discriminant information beyond individual metric comparisons. "
                 f"The dominant principal components likely represent latent motor patterns related "
                 f"to force production magnitude (PC1) and movement timing/smoothness (PC2).")
    return "\n".join(lines)

def apa_m_cluster(n1, n2, g1_s, g2_s, nm):
    return (f"Unsupervised hierarchical cluster analysis was performed using Ward's minimum "
            f"variance linkage (Ward, 1963) on {nm} standardized kinematic metrics from all "
            f"{n1+n2} participants, without access to group labels. Euclidean distance was used "
            f"as the dissimilarity measure. To quantify the ecological validity of the solution, "
            f"the proportion of participants correctly assigned to their known groups "
            f"({g1_s}, n = {n1}; {g2_s}, n = {n2}) was computed as unsupervised classification accuracy.")

def apa_r_cluster(acc, n1, n2, g1_s, g2_s):
    q = "excellent" if acc>=80 else "good" if acc>=70 else "moderate" if acc>=60 else "limited"
    lines = [f"Ward's hierarchical clustering achieved {q} unsupervised recovery of the known "
             f"group structure, correctly assigning {acc:.1f}% of participants to their respective "
             f"groups ({g1_s}, n = {n1}; {g2_s}, n = {n2}) without label information."]
    interp = ("This suggests that the multivariate sit-to-stand kinematic profile naturally "
              "segregates into distinct movement patterns broadly corresponding to faller and "
              "non-faller status, providing unsupervised evidence for the discriminative capacity "
              "of these metrics." if acc>=70 else
              "The limited recovery suggests considerable kinematic overlap between groups, "
              "possibly reflecting heterogeneity in fall mechanisms or compensatory strategies "
              "within the faller cohort.")
    lines.append(f"\n{interp}")
    return "\n".join(lines)

def apa_m_score(n1, n2, g1_s, g2_s):
    return (f"A composite Fall Risk Score was derived from statistically significant kinematic "
            f"metrics (p_adj < 0.05). Each metric was normalized to [0, 1] using pooled "
            f"minimum and maximum values. Metric direction was adjusted so higher scores "
            f"reflect greater similarity to the {g2_s} (faller) profile. The composite score "
            f"was a weighted linear combination, with weights proportional to |Cohen's d|, "
            f"normalized to sum to 1. Group differences in the composite score were assessed "
            f"with the Mann-Whitney U test (α = 0.05).")

def apa_r_score(score_res, g1_s, g2_s, n_met):
    if score_res is None:
        return "The composite risk score could not be computed (no statistically significant metrics identified)."
    U, p, d = score_res
    lines = [f"The Fall Risk Score, derived from {n_met} significant kinematic metric(s), "
             f"demonstrated {'statistically significant' if p<0.05 else 'non-significant'} "
             f"group separation (Mann-Whitney U = {U:.0f}, p = {p:.3f}, d = {d:.3f} "
             f"[{effect_label(d)} effect])."]
    if p<0.05:
        lines.append(f"\nThe composite score effectively distinguished {g2_s} from {g1_s}, "
                     f"suggesting that a weighted combination of sit-to-stand kinematic metrics "
                     f"provides a clinically meaningful index of fall risk. This multi-metric "
                     f"approach leverages complementary information across movement phases and "
                     f"axes, potentially offering superior discriminative power over any single metric.")
    return "\n".join(lines)

def apa_m_lda(n1, n2, g1_s, g2_s, nm):
    return (f"Linear Discriminant Analysis (LDA) was performed to identify the optimal linear "
            f"combination of {nm} kinematic metrics maximizing separation between {g2_s} (n = {n2}) "
            f"and {g1_s} (n = {n1}) (Fisher, 1936; McLachlan, 1992). Variables were standardized "
            f"prior to analysis. Discriminant weights were obtained as w = S_W⁻¹(μ₁ − μ₂), "
            f"where S_W is the pooled within-class scatter matrix. Training accuracy was computed "
            f"as the proportion of correctly classified participants using the linear boundary "
            f"(threshold at the midpoint of projected class means). Note: training accuracy may "
            f"overestimate generalizability; cross-validation in independent samples is recommended.")

def apa_r_lda(acc, top_w, g1_s, g2_s):
    if acc is None: return "LDA could not be computed due to insufficient data or matrix singularity."
    q = "excellent" if acc>=85 else "good" if acc>=75 else "acceptable" if acc>=65 else "limited"
    lines = [f"LDA achieved a training classification accuracy of {acc:.1f}%, representing {q} "
             f"discrimination between {g2_s} and {g1_s} based on the combined kinematic profile."]
    if top_w:
        lines.append(f"\nThe three most influential discriminant features were:")
        for metric, weight in top_w:
            lines.append(f"  • {metric} (weight = {weight:.3f}): reflects {get_phys(metric)}.")
    lines.append(f"\nThese results suggest that sit-to-stand kinematics contain sufficient "
                 f"multivariate discriminant information to classify fallers with clinically "
                 f"meaningful accuracy. The identified discriminant features represent the "
                 f"kinematic dimensions most relevant for distinguishing pathological from "
                 f"typical movement strategies during sit-to-stand.")
    return "\n".join(lines)

def apa_m_bootstrap(n1, n2, g1_s, g2_s, n_perm):
    return (f"To assess robustness of group comparisons under sampling variability, a permutation "
            f"test was performed for each metric (Edgington & Onghena, 2007). Group labels were "
            f"randomly permuted {n_perm} times, and the Mann-Whitney U statistic recomputed at "
            f"each iteration; the permutation p-value was the proportion of permuted statistics "
            f"as extreme as the observed. Additionally, 95% bootstrap confidence intervals for "
            f"Cohen's d were estimated from {n_perm} resamples with replacement from {g1_s} "
            f"(n = {n1}) and {g2_s} (n = {n2}). This approach is particularly appropriate "
            f"for small samples where asymptotic assumptions may not hold.")

def apa_r_bootstrap():
    return ("Run the bootstrap analysis above to generate specific results. Once completed, "
            "report for each metric: the observed p-value (Mann-Whitney), the permutation "
            "p-value, the bootstrap 95% CI for Cohen's d, and whether the CI excludes zero.\n\n"
            "Suggested reporting format:\n"
            "  'Permutation testing confirmed the robustness of [metric] group differences "
            "(p_perm = X.XXX). Bootstrap 95% CI for Cohen's d [X.XX, X.XX] excluded zero, "
            "supporting the stability of the effect estimate under resampling.'")

# ── APA: Curvas Resultantes ───────────────────────────────────────────────────
def apa_m_curves(n_fall, n_ctrl):
    return (f"Mean ± standard deviation (SD) acceleration resultant curves were computed "
            f"for the faller group (FALL, n = {n_fall}) and the non-faller control group "
            f"(CTRL, n = {n_ctrl}) across normalized movement phases (0–1). "
            f"Curve similarity was quantified using (1) Pearson's correlation coefficient (r) "
            f"between group mean curves, (2) root mean square error (RMSE, m/s²), and "
            f"(3) the area between mean curves computed via trapezoidal integration. "
            f"Phase-specific analyses were conducted for three biomechanically defined phases: "
            f"P1 (initiation), P2 (transition/seat-off), and P3 (stabilization), "
            f"delimited by the mean phase boundaries across groups.")

def apa_r_curves(r_val, rmse_v, area_v, max_d, max_d_f, rows, P2, P3):
    lines = [
        f"The mean acceleration resultant curves of FALL and CTRL groups showed "
        f"{'high' if r_val>0.9 else 'moderate' if r_val>0.7 else 'low'} overall similarity "
        f"(r = {r_val:.4f}, RMSE = {rmse_v:.4f} m/s²). "
        f"The area between curves was {area_v:.4f} m/s², indicating "
        f"{'minimal' if area_v<0.5 else 'moderate' if area_v<2 else 'substantial'} "
        f"accumulated difference across the full movement cycle. "
        f"Maximum instantaneous difference was {max_d:.4f} m/s² at normalized phase ≈ {max_d_f:.3f}."
    ]
    if rows:
        lines.append("\nPhase-specific analysis revealed:")
        for r in rows:
            ph = r['Fase']
            delta = r.get('Δ','')
            pct   = r.get('Δ%','')
            rmse_ph = r.get('RMSE','')
            corr_ph = r.get('r','')
            ph_name = "initiation" if ph=="P1" else "transitional" if ph=="P2" else "stabilization"
            lines.append(
                f"  • {ph} ({ph_name}): FALL = {r.get('FALL','')}, CTRL = {r.get('CTRL','')} m/s², "
                f"Δ = {delta} m/s² ({pct}), RMSE = {rmse_ph} m/s², r = {corr_ph}. "
                f"{'Higher CTRL acceleration during this phase may reflect greater momentum generation capacity.' if delta.startswith('+') else 'Lower CTRL acceleration during this phase may reflect more controlled movement.' if delta.startswith('-') else ''}"
            )
    lines.append(
        "\nThese curve-level differences provide complementary information to scalar metric "
        "comparisons, capturing the temporal dynamics of between-group differences across "
        "the entire sit-to-stand movement cycle."
    )
    return "\n".join(lines)

# ── APA: SPM ─────────────────────────────────────────────────────────────────
def apa_m_spm(n_fall, n_ctrl):
    return (f"A Statistical Parametric Mapping-inspired point-by-point z-test was performed "
            f"to identify time intervals with statistically significant between-group differences "
            f"in acceleration resultant across the normalized movement cycle (Friston et al., 1994; "
            f"Pataky et al., 2013). At each normalized time point, a z-statistic was computed as: "
            f"z = (μ_CTRL − μ_FALL) / √(σ²_CTRL/n_CTRL + σ²_FALL/n_FALL), "
            f"where μ and σ² represent the group mean and variance at that time point, "
            f"n_CTRL = {n_ctrl} and n_FALL = {n_fall}. "
            f"Time intervals with |z| > 1.96 were considered statistically significant (α = 0.05, "
            f"two-tailed). Note: this analysis does not account for temporal autocorrelation; "
            f"for formal SPM inference, spm1d software (Pataky, 2012) is recommended.")

def apa_r_spm(regs, fase, z_arr, P2, P3):
    if not regs:
        return ("The point-by-point z-test revealed no time intervals with statistically "
                "significant between-group differences in acceleration resultant (|z| ≤ 1.96 "
                "throughout the normalized movement cycle), suggesting broadly similar "
                "temporal acceleration profiles between fallers and non-fallers.")
    lines = [f"Point-by-point z-test analysis identified {len(regs)} region(s) with "
             f"statistically significant between-group differences in acceleration resultant (|z| > 1.96):"]
    for i, (si, ei, mz) in enumerate(regs):
        ph = "P1 (initiation)" if fase[si]<P2 else "P2 (transition)" if fase[si]<P3 else "P3 (stabilization)"
        direction = "CTRL > FALL" if mz>0 else "FALL > CTRL"
        dur = fase[ei]-fase[si]
        lines.append(
            f"  • Region #{i+1}: normalized phase {fase[si]:.4f}–{fase[ei]:.4f} "
            f"(duration = {dur:.4f}; located in {ph}), "
            f"peak z = {mz:.3f} ({direction}). "
            f"{'Higher CTRL acceleration in this region may reflect greater momentum generation or more efficient weight transfer.' if mz>0 else 'Higher FALL acceleration in this region may reflect compensatory movement strategies or reduced movement smoothness.'}"
        )
    lines.append(
        "\nThese temporal windows of significant difference highlight the specific "
        "movement phases where faller and non-faller kinematics diverge most, "
        "providing mechanistic insight beyond global curve similarity metrics."
    )
    return "\n".join(lines)

# ── APA: Análise da Forma ─────────────────────────────────────────────────────
def apa_m_shape():
    return ("Movement quality and curve shape were characterized using four complementary analyses. "
            "(1) Coefficient of Variation (CV%): intragroup variability at each normalized time point "
            "was quantified as CV(t) = SD(t)/|mean(t)| × 100%, capturing where within the movement "
            "cycle each group is most variable. "
            "(2) Envelope overlap: the proportional overlap between ±1 SD bands of both groups at "
            "each time point was computed as overlap = max(0, min(hi_FALL,hi_CTRL)−max(lo_FALL,lo_CTRL)) "
            "/ (max(hi_FALL,hi_CTRL)−min(lo_FALL,lo_CTRL)) × 100%; 0% = complete separation, "
            "100% = complete overlap. "
            "(3) Cross-correlation: Pearson cross-correlation between standardized mean curves was "
            "computed across all lag values to assess temporal alignment and shape similarity. "
            "(4) Peak analysis: local maxima in each group's mean curve were identified using "
            "prominence-based peak detection; peak amplitude and timing were compared between groups.")

def apa_r_shape(cv_f, cv_c, ov_pct, zc, pl, peaks_f, peaks_c, fase, P2, P3, has_overlap):
    lines = []
    # CV
    cv_f_mean = float(np.nanmean(cv_f)); cv_c_mean = float(np.nanmean(cv_c))
    more_var = "FALL" if cv_f_mean>cv_c_mean else "CTRL"
    lines.append(f"(1) CV% analysis: mean intragroup variability across the full movement cycle "
                 f"was {cv_f_mean:.1f}% for FALL and {cv_c_mean:.1f}% for CTRL, "
                 f"indicating that the {more_var} group exhibited greater temporal variability. "
                 f"Elevated CV% may reflect reduced motor consistency and increased movement-to-movement "
                 f"variability, which are associated with fall risk.")
    # Overlap
    ov_mean = float(np.mean(ov_pct))
    pct_sep = float(np.mean(~has_overlap)*100)
    lines.append(f"\n(2) Envelope overlap: mean ±1 SD band overlap was {ov_mean:.1f}% across the "
                 f"full cycle; the groups were completely separated (0% overlap) for {pct_sep:.1f}% "
                 f"of the movement. "
                 f"{'Low overlap indicates that the groups occupy largely distinct acceleration amplitude ranges, supporting their kinematic distinctiveness.' if ov_mean<50 else 'Moderate-to-high overlap suggests that while mean curves differ, individual variability produces substantial group overlap in the acceleration amplitude space.'}")
    # Cross-correlation
    lines.append(f"\n(3) Cross-correlation: at lag = 0 (perfect temporal alignment), r = {zc:.4f}. "
                 f"{'This indicates high shape similarity between group mean curves, with differences primarily in amplitude rather than timing.' if abs(zc)>0.8 else 'This indicates moderate shape similarity, with both amplitude and temporal differences contributing to between-group divergence.'}"
                 f"{f' The peak correlation occurred at lag = {pl} samples, suggesting a temporal offset between groups.' if pl!=0 else ' No temporal offset was detected (peak at lag = 0).'}")
    # Peaks
    if len(peaks_f)>0 or len(peaks_c)>0:
        lines.append(f"\n(4) Peak analysis: {len(peaks_f)} peak(s) detected in FALL and "
                     f"{len(peaks_c)} peak(s) in CTRL mean curves. "
                     f"Differences in peak number, amplitude, and timing may reflect distinct "
                     f"momentum generation strategies between groups during sit-to-stand.")
    return "\n".join(lines)

# ── APA: Stats in upload (Estatísticas tab) ───────────────────────────────────
def apa_m_boxplot(n1, n2, g1_s, g2_s):
    return (f"Individual data distributions for each kinematic metric were visualized using "
            f"box-and-whisker plots with overlaid individual data points. Box bounds represent "
            f"the interquartile range (IQR; Q1–Q3), the horizontal line denotes the median, "
            f"the cross (×) indicates the mean, and whiskers extend to 1.5×IQR. "
            f"Descriptive statistics (n, mean, median, SD, min, max) are reported for "
            f"{g1_s} (n = {n1}) and {g2_s} (n = {n2}). "
            f"Statistical comparison used the Mann-Whitney U test (Mann & Whitney, 1947) "
            f"with effect size quantified by Cohen's d (Cohen, 1988).")

# ── Render advanced analyses tab ──────────────────────────────────────────────
def render_advanced_tab(df_sub, g1_name, g2_name, ks):
    mc, g1_df, g2_df = prep_data(df_sub, g1_name, g2_name)
    g1_s = g1_name.replace("GRUPO","").strip().title()
    g2_s = g2_name.replace("GRUPO","").strip().title()
    n1, n2 = len(g1_df), len(g2_df)
    results = compute_results(mc, g1_df, g2_df)
    if not results: st.warning("Dados insuficientes para análises avançadas."); return

    # ── Stats table overview + APA text
    st.markdown("### 📊 Comparação Estatística Geral")
    render_stats_table(results, g1_s, g2_s, ks+"_adv")
    txt_boxes(apa_m_stats(n1,n2,g1_s,g2_s,len(results)),
              apa_r_stats(results,g1_s,g2_s), ks+"_stat")

    st.markdown("---")

    # ── ROC
    st.markdown("### 🔴 Curvas ROC + AUC")
    roc_data = []
    for r in results:
        try:
            fprs,tprs,auc,ot,sens,spec = compute_roc(r["b"],r["a"])
            roc_data.append({"col":r["col"],"auc":auc,"threshold":ot,"sens":sens,"spec":spec})
        except: pass
    render_roc(results, g1_s, g2_s, ks+"_roc")
    txt_boxes(apa_m_roc(n1,n2,g1_s,g2_s), apa_r_roc(roc_data,g1_s,g2_s), ks+"_roc_t")

    st.markdown("---")

    # ── Forest Plot
    st.markdown("### 🔴 Forest Plot — d de Cohen com IC 95%")
    render_forest(results, g1_s, g2_s)
    txt_boxes(apa_m_forest(n1,n2,g1_s,g2_s,len(results)),
              apa_r_forest(results,g1_s,g2_s), ks+"_for")

    st.markdown("---")

    # ── PCA
    st.markdown("### 🔴 PCA — Análise de Componentes Principais")
    cols_use = [r["col"] for r in results if r["col"] in g1_df.columns]
    g1_X = g1_df[cols_use].apply(pd.to_numeric,errors="coerce").dropna()
    g2_X = g2_df[cols_use].apply(pd.to_numeric,errors="coerce").dropna()
    X = np.vstack([g1_X.values,g2_X.values])
    pca_scores, pca_var, pca_Vt = pca_2d(X)
    top_pc1 = sorted(zip(cols_use,pca_Vt[0]),key=lambda x:abs(x[1]),reverse=True)[:3] if pca_Vt is not None else []
    render_pca(results,g1_df,g2_df,g1_s,g2_s,g1_name,g2_name,ks+"_pca")
    txt_boxes(apa_m_pca(n1,n2,g1_s,g2_s,len(cols_use)),
              apa_r_pca(pca_var,top_pc1,g1_s,g2_s), ks+"_pca_t")

    st.markdown("---")

    # ── Heatmap
    st.markdown("### 🟡 Heatmap de Correlação entre Métricas")
    render_heatmap(results,g1_df,g2_df,g1_s,g2_s)

    st.markdown("---")

    # ── Cluster
    st.markdown("### 🟡 Análise de Cluster Hierárquica")
    render_cluster(results,g1_df,g2_df,g1_name,g2_name,g1_s,g2_s,ks+"_cl")
    # Get cluster accuracy for text
    try:
        mu=X.mean(0); sd=X.std(0); sd[sd<1e-10]=1; Xs=(X-mu)/sd
        y=np.array([0]*len(g1_X)+[1]*len(g2_X))
        Z=linkage(Xs,method="ward"); pred=fcluster(Z,t=2,criterion="maxclust")-1
        cl_acc=max(np.mean(pred==y),np.mean((1-pred)==y))*100
    except: cl_acc=50.0
    txt_boxes(apa_m_cluster(n1,n2,g1_s,g2_s,len(cols_use)),
              apa_r_cluster(cl_acc,n1,n2,g1_s,g2_s), ks+"_cl_t")

    st.markdown("---")

    # ── Risk Score
    st.markdown("### 🟡 Score Composto de Risco")
    sig_r = [r for r in results if r.get("padj",1)<0.05]
    render_risk_score(results,g1_df,g2_df,g1_s,g2_s,g1_name,g2_name,ks+"_rs")
    score_res = None
    if sig_r:
        try:
            cols_s=[r["col"] for r in sig_r]
            ws=np.array([abs(r["d"]) for r in sig_r]); ws/=ws.sum()
            a_s=g1_df[cols_s].apply(pd.to_numeric,errors="coerce").dropna()
            b_s=g2_df[cols_s].apply(pd.to_numeric,errors="coerce").dropna()
            all_s=pd.concat([a_s,b_s]); mn=all_s.min(); mx=all_s.max(); rng=(mx-mn).replace(0,1)
            an=(a_s-mn)/rng; bn=(b_s-mn)/rng
            for r in sig_r:
                if r["d"]<0: an[r["col"]]=1-an[r["col"]]; bn[r["col"]]=1-bn[r["col"]]
            s1=(an*ws).sum(axis=1); s2=(bn*ws).sum(axis=1)
            U,p=stats.mannwhitneyu(s1,s2,alternative="two-sided")
            d_sc=cohen_d(s1.values,s2.values)
            score_res=(U,p,d_sc)
        except: pass
    txt_boxes(apa_m_score(n1,n2,g1_s,g2_s),
              apa_r_score(score_res,g1_s,g2_s,len(sig_r)), ks+"_rs_t")

    st.markdown("---")

    # ── LDA
    st.markdown("### 🟢 LDA — Análise Discriminante Linear")
    render_lda(results,g1_df,g2_df,g1_s,g2_s,ks+"_lda")
    try:
        mu=X.mean(0); sd_=X.std(0); sd_[sd_<1e-10]=1; Xs=(X-mu)/sd_
        y=np.array([0]*len(g1_X)+[1]*len(g2_X))
        _,w,_,acc=lda_2g(Xs,y)
        top_w=sorted(zip(cols_use,w),key=lambda x:abs(x[1]),reverse=True)[:3] if w is not None else []
    except: acc=None; top_w=[]
    txt_boxes(apa_m_lda(n1,n2,g1_s,g2_s,len(cols_use)),
              apa_r_lda(acc,top_w,g1_s,g2_s), ks+"_lda_t")

    st.markdown("---")

    # ── Bootstrap
    st.markdown("### 🟢 Bootstrap dos p-values + IC de Cohen's d")
    render_bootstrap(results, ks+"_boot")
    txt_boxes(apa_m_bootstrap(n1,n2,g1_s,g2_s,1000),
              apa_r_bootstrap(), ks+"_boot_t")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + SOURCE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="main-header">
  <h1>📊 Análise Comparativa — Sit-to-Stand</h1>
  <p>FALL vs CONTROLE · Curvas médias e métricas por fases normalizadas</p>
</div>""", unsafe_allow_html=True)

data_source = st.radio("**Fonte dos dados**",
                       ["📁 Dados embutidos","📤 Upload de arquivo"], horizontal=True)
use_upload = data_source == "📤 Upload de arquivo"

# ══════════════════════════════════════════════════════════════════════════════
# MODO UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if use_upload:
    st.markdown("""<div class="info-box">
    Arquivo <strong>.xlsx</strong> com abas <em>Vertical</em> e <em>Mediolateral</em>.<br>
    Colunas obrigatórias: <code>SUJEITOS</code> · <code>Grupo</code> · <code>Device</code> + métricas.
    </div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Arquivo de dados (.xlsx)", type="xlsx", key="main_upload")
    if not uploaded:
        st.info("Faça upload do arquivo para ver as análises."); st.stop()
    sheets     = load_upload(uploaded.getvalue())
    sheet_names= list(sheets.keys())
    first_df   = sheets[sheet_names[0]]
    if "Grupo" not in first_df.columns or "Device" not in first_df.columns:
        st.error("Colunas 'Grupo' e/ou 'Device' não encontradas."); st.stop()
    g1_name, g2_name = detect_groups(first_df)
    g1_s = g1_name.replace("GRUPO","").strip().title()
    g2_s = g2_name.replace("GRUPO","").strip().title()
    all_devs = first_df["Device"].dropna().unique().tolist()
    has_k = any("kinem" in d.lower() for d in all_devs)
    has_m = any("mobil" in d.lower() for d in all_devs)
    st.markdown(f"""<div style="margin-bottom:16px">
      <span class="badge badge-ctrl">{g1_s}</span>
      <span class="badge badge-fall">{g2_s}</span>
      &nbsp;&nbsp;<span style="font-size:0.83rem;color:#6b7280">
      Devices: {', '.join(all_devs)} | Abas: {', '.join(sheet_names)}</span>
    </div>""", unsafe_allow_html=True)
    dev_labels = (["⚙️ Kinem"] if has_k else []) + (["📱 Mobile"] if has_m else []) or ["📊 Análise"]
    dev_tabs   = st.tabs(dev_labels)
    for dev_tab, dev_label in zip(dev_tabs, dev_labels):
        with dev_tab:
            dk = "kinem" if "Kinem" in dev_label else "mobile" if "Mobile" in dev_label else ""
            axis_tabs = st.tabs([f"↕️ {s}" for s in sheet_names])
            for axis_tab, sheet_name in zip(axis_tabs, sheet_names):
                with axis_tab:
                    df_dev = filt_dev(sheets[sheet_name], dk) if dk else sheets[sheet_name]
                    if df_dev.empty:
                        st.warning(f"Nenhum dado para '{dk}' em '{sheet_name}'."); continue
                    ks = f"{dk}_{sheet_name}"
                    stat_tab, adv_tab = st.tabs(["📊 Estatísticas", "🔬 Análises Avançadas + Texto APA"])
                    with stat_tab:
                        render_metrics_analysis(df_dev, g1_name, g2_name, key_suffix=ks)
                    with adv_tab:
                        render_advanced_tab(df_dev, g1_name, g2_name, ks=ks+"_adv")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODO EMBUTIDO — curvas
# ══════════════════════════════════════════════════════════════════════════════
fall_curv, ctrl_curv, fall_mr, ctrl_mr, fall_ind, ctrl_ind = load_embedded()
fase=fall_curv["Fase_norm"].values; fm=fall_curv["Média (m/s²)"].values
fdp=fall_curv["DP (m/s²)"].values;  fhi=fall_curv["Média+DP (m/s²)"].values
flo=fall_curv["Média-DP (m/s²)"].values
cm=ctrl_curv["Média (m/s²)"].values; cdp=ctrl_curv["DP (m/s²)"].values
chi=ctrl_curv["Média+DP (m/s²)"].values; clo=ctrl_curv["Média-DP (m/s²)"].values
diff=cm-fm
try:
    P2=(float(fall_mr.iloc[0,1])+float(ctrl_mr.iloc[0,1]))/2
    P3=(float(fall_mr.iloc[0,2])+float(ctrl_mr.iloc[0,2]))/2
except: P2,P3=0.33,0.66
try:
    N_FALL=int(fall_mr.iloc[0,0].split("n=")[1].replace(")",""))
    N_CTRL=int(ctrl_mr.iloc[0,0].split("n=")[1].replace(")",""))
except: N_FALL=N_CTRL=18

st.markdown(f"""<div style="margin-bottom:16px">
  <span class="badge badge-fall">FALL n={N_FALL}</span>
  <span class="badge badge-ctrl">CONTROLE n={N_CTRL}</span>
</div>""", unsafe_allow_html=True)

tab1,tab3,tab5,tab_adv=st.tabs(["📈 Curvas Resultantes","🔬 Análise Temporal (SPM)","🧬 Análise da Forma das Curvas","🔬 Análises Avançadas + Texto APA"])

with tab1:
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([fhi,flo[::-1]]),
                             fill="toself",fillcolor=C_FALL_BG,line=dict(color="rgba(0,0,0,0)"),name="FALL ±1DP"),secondary_y=False)
    fig.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([chi,clo[::-1]]),
                             fill="toself",fillcolor=C_CTRL_BG,line=dict(color="rgba(0,0,0,0)"),name="CTRL ±1DP"),secondary_y=False)
    fig.add_trace(go.Scatter(x=fase,y=fm,name="FALL (média)",line=dict(color=C_FALL,width=2.5)),secondary_y=False)
    fig.add_trace(go.Scatter(x=fase,y=cm,name="CONTROLE (média)",line=dict(color=C_CTRL,width=2.5)),secondary_y=False)
    fig.add_trace(go.Scatter(x=fase,y=diff,name="Diferença CTRL−FALL",line=dict(color=C_DIFF,width=1.8,dash="dot")),secondary_y=True)
    add_phase_lines(fig,P2,P3)
    fig.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Fase normalizada",
                      yaxis_title="Aceleração resultante (m/s²)",hovermode="x unified")
    fig.update_yaxes(title_text="Diferença CTRL−FALL (m/s²)",showgrid=False,secondary_y=True)
    _c1,_c2=st.columns([1,1])
    with _c1: st.plotly_chart(fig,use_container_width=False)
    r_val=float(np.corrcoef(fm,cm)[0,1]); rmse_v=float(np.sqrt(np.mean(diff**2)))
    area_v=float(np.trapezoid(np.abs(diff),fase)); max_d=float(np.max(np.abs(diff)))
    max_d_f=float(fase[np.argmax(np.abs(diff))])
    with _c2:
        st.markdown("**Métricas de comparação**")
        st.metric("Correlação (r)",f"{r_val:.4f}")
        st.metric("RMSE",f"{rmse_v:.4f} m/s²")
        st.metric("Área entre curvas",f"{area_v:.4f}")
        st.metric("Dif. máxima",f"{max_d:.4f}",help=f"Fase≈{max_d_f:.3f}")
    st.markdown("**Diferença por fase**")
    rows=[]
    for ph,s,e in [("P1",0,P2),("P2",P2,P3),("P3",P3,1.0)]:
        mk=phase_mask(fase,s,e); fp,cp=fm[mk],cm[mk]
        da=float(np.mean(cp)-np.mean(fp)); dr=da/abs(float(np.mean(fp)))*100 if np.mean(fp)!=0 else 0
        rows.append(dict(Fase=ph,**{"FALL":f"{np.mean(fp):.4f}","CTRL":f"{np.mean(cp):.4f}",
            "Δ":f"{da:+.4f}","Δ%":f"{dr:+.1f}%","RMSE":f"{float(np.sqrt(np.mean((cp-fp)**2))):.4f}",
            "r":f"{float(np.corrcoef(fp,cp)[0,1]):.4f}"}))
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    txt_boxes(apa_m_curves(N_FALL,N_CTRL),
              apa_r_curves(r_val,rmse_v,area_v,max_d,max_d_f,rows,P2,P3),
              "emb_curves")

with tab3:
    st.markdown("""<div class="info-box"><strong>SPM-like z-test</strong>:
    z = (CTRL−FALL)/√(DP²_CTRL/n_CTRL + DP²_FALL/n_FALL). Threshold ±1.96 (α=0.05).
    </div>""",unsafe_allow_html=True)
    se=np.sqrt(cdp**2/N_CTRL+fdp**2/N_FALL); z_arr=np.where(se>0,diff/se,0.0)
    d_pct=np.where(fm!=0,diff/np.abs(fm)*100,0.0)
    fig2=go.Figure()
    fig2.add_trace(go.Bar(x=fase,y=z_arr,marker_color=[C_FALL if abs(z)>1.96 else "#90caf9" for z in z_arr],showlegend=False))
    fig2.add_hline(y=1.96,line=dict(color=C_DIFF,dash="dash",width=1.5),annotation_text="z=+1.96",annotation_position="right")
    fig2.add_hline(y=-1.96,line=dict(color=C_DIFF,dash="dash",width=1.5),annotation_text="z=−1.96",annotation_position="right")
    add_phase_lines(fig2,P2,P3)
    fig2.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Fase normalizada",yaxis_title="z-score")
    fig3=make_subplots(specs=[[{"secondary_y":True}]])
    fig3.add_trace(go.Scatter(x=fase,y=diff,name="Δ abs (m/s²)",line=dict(color=C_DIFF,width=2),
                              fill="tozeroy",fillcolor="rgba(245,158,11,0.1)"),secondary_y=False)
    fig3.add_trace(go.Scatter(x=fase,y=d_pct,name="Δ rel (%)",line=dict(color="#7c3aed",width=1.5,dash="dot")),secondary_y=True)
    add_phase_lines(fig3,P2,P3)
    fig3.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Fase normalizada",yaxis_title="Δ m/s²")
    fig3.update_yaxes(title_text="Δ%",showgrid=False,secondary_y=True)
    _c1,_c2=st.columns([1,1])
    with _c1: st.plotly_chart(fig2,use_container_width=False)
    with _c2: st.plotly_chart(fig3,use_container_width=False)
    regs,in_r,si,mz=[],False,0,0.0
    for i,z in enumerate(z_arr):
        if abs(z)>1.96:
            if not in_r: in_r,si,mz=True,i,z
            elif abs(z)>abs(mz): mz=z
            if i==len(z_arr)-1: regs.append((si,i,mz))
        else:
            if in_r: regs.append((si,i-1,mz)); in_r=False
    st.markdown("**Regiões significativas**")
    if not regs: st.info("Nenhuma região significativa detectada.")
    else:
        st.dataframe(pd.DataFrame([{"Região":f"#{i+1}","Início":f"{fase[s]:.4f}","Fim":f"{fase[e]:.4f}",
            "Duração":f"{fase[e]-fase[s]:.4f}","z máx":f"{mz:.3f}",
            "Direção":"CTRL>FALL" if mz>0 else "FALL>CTRL",
            "Fase":"P1" if fase[s]<P2 else "P2" if fase[s]<P3 else "P3"}
            for i,(s,e,mz) in enumerate(regs)]),use_container_width=True,hide_index=True)
    txt_boxes(apa_m_spm(N_FALL,N_CTRL),
              apa_r_spm(regs,fase,z_arr,P2,P3),
              "emb_spm")

with tab5:
    st.markdown("### CV% ao longo da fase")
    cv_f=np.where(np.abs(fm)>0.01,fdp/np.abs(fm)*100,np.nan)
    cv_c=np.where(np.abs(cm)>0.01,cdp/np.abs(cm)*100,np.nan)
    fig_cv=go.Figure()
    fig_cv.add_trace(go.Scatter(x=fase,y=cv_f,name="FALL CV%",line=dict(color=C_FALL,width=2)))
    fig_cv.add_trace(go.Scatter(x=fase,y=cv_c,name="CONTROLE CV%",line=dict(color=C_CTRL,width=2)))
    add_phase_lines(fig_cv,P2,P3)
    fig_cv.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Fase normalizada",yaxis_title="CV (%)",hovermode="x unified")

    ov_abs=np.maximum(0,np.minimum(fhi,chi)-np.maximum(flo,clo))
    union=np.maximum(fhi,chi)-np.minimum(flo,clo)
    ov_pct=np.where(union>0,ov_abs/union*100,0.0)
    fig_ov=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.08)
    fig_ov.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([fhi,flo[::-1]]),fill="toself",fillcolor=C_FALL_BG,line=dict(color="rgba(0,0,0,0)"),name="FALL ±1DP"),row=1,col=1)
    fig_ov.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([chi,clo[::-1]]),fill="toself",fillcolor=C_CTRL_BG,line=dict(color="rgba(0,0,0,0)"),name="CTRL ±1DP"),row=1,col=1)
    fig_ov.add_trace(go.Scatter(x=fase,y=fm,name="FALL",line=dict(color=C_FALL,width=2)),row=1,col=1)
    fig_ov.add_trace(go.Scatter(x=fase,y=cm,name="CTRL",line=dict(color=C_CTRL,width=2)),row=1,col=1)
    fig_ov.add_trace(go.Bar(x=fase,y=ov_pct,marker_color=[f"rgba(229,57,53,{0.4+0.6*(1-v/100):.2f})" if v<50 else f"rgba(0,137,123,{0.3+0.7*(v/100):.2f})" for v in ov_pct],showlegend=False),row=2,col=1)
    for xv in [P2,P3]: fig_ov.add_vline(x=xv,line=dict(color="#9e9e9e",dash="dash",width=1))
    fig_ov.update_layout(**base_layout(h=SQ,w=SQ),hovermode="x unified")
    fig_ov.update_yaxes(title_text="Aceleração (m/s²)",row=1,col=1)
    fig_ov.update_yaxes(title_text="Sobreposição (%)",row=2,col=1,range=[0,105])
    fig_ov.update_xaxes(title_text="Fase normalizada",row=2,col=1)
    _c1,_c2=st.columns([1,1])
    with _c1:
        st.markdown("**CV% por grupo**"); st.plotly_chart(fig_cv,use_container_width=False)
    with _c2:
        st.markdown("**Sobreposição ±1DP**"); st.plotly_chart(fig_ov,use_container_width=False)

    st.markdown("---")
    fm_n=(fm-fm.mean())/(fm.std()+1e-10); cm_n=(cm-cm.mean())/(cm.std()+1e-10)
    xcorr=correlate(fm_n,cm_n,mode="full")/len(fm); lags=correlation_lags(len(fm),len(cm),mode="full")
    pi=int(np.argmax(xcorr)); pl=lags[pi]; pc=float(xcorr[pi]); zc=float(xcorr[len(fm)-1])
    fig_xc=go.Figure()
    fig_xc.add_trace(go.Scatter(x=lags,y=xcorr,line=dict(color="#1a73e8",width=2),name="Cross-correlação"))
    fig_xc.add_vline(x=0,line=dict(color="#9e9e9e",dash="dash",width=1.2),annotation_text="Lag=0",annotation_position="top right")
    fig_xc.add_vline(x=pl,line=dict(color=C_DIFF,dash="dot",width=1.5),annotation_text=f"Pico lag={pl}",annotation_position="top left")
    fig_xc.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Lag (amostras)",yaxis_title="Correlação normalizada")

    cp,cd=st.columns(2)
    with cp: prom=st.slider("Proeminência mínima (m/s²)",0.1,3.0,0.5,0.1)
    with cd: mdist=st.slider("Distância mínima (amostras)",5,50,15)
    pf,_=find_peaks(fm,prominence=prom,distance=mdist); pc_,_=find_peaks(cm,prominence=prom,distance=mdist)
    fig_pk=go.Figure()
    fig_pk.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([fhi,flo[::-1]]),fill="toself",fillcolor=C_FALL_BG,line=dict(color="rgba(0,0,0,0)"),name="FALL ±1DP"))
    fig_pk.add_trace(go.Scatter(x=np.concatenate([fase,fase[::-1]]),y=np.concatenate([chi,clo[::-1]]),fill="toself",fillcolor=C_CTRL_BG,line=dict(color="rgba(0,0,0,0)"),name="CTRL ±1DP"))
    fig_pk.add_trace(go.Scatter(x=fase,y=fm,name="FALL",line=dict(color=C_FALL,width=2.5)))
    fig_pk.add_trace(go.Scatter(x=fase,y=cm,name="CTRL",line=dict(color=C_CTRL,width=2.5)))
    if len(pf): fig_pk.add_trace(go.Scatter(x=fase[pf],y=fm[pf],mode="markers",marker=dict(color=C_FALL,size=10,symbol="triangle-up",line=dict(color="white",width=1.5)),name="Picos FALL"))
    if len(pc_): fig_pk.add_trace(go.Scatter(x=fase[pc_],y=cm[pc_],mode="markers",marker=dict(color=C_CTRL,size=10,symbol="triangle-up",line=dict(color="white",width=1.5)),name="Picos CTRL"))
    add_phase_lines(fig_pk,P2,P3)
    fig_pk.update_layout(**base_layout(h=SQ,w=SQ),xaxis_title="Fase normalizada",yaxis_title="Aceleração (m/s²)",hovermode="x unified")
    _c1,_c2=st.columns([1,1])
    with _c1:
        st.markdown("**Cross-Correlação**"); st.plotly_chart(fig_xc,use_container_width=False)
        ca,cb,cc=st.columns(3)
        ca.metric("r lag=0",f"{zc:.4f}"); cb.metric("r máx",f"{pc:.4f}"); cc.metric("Lag pico",f"{pl}")
    with _c2:
        st.markdown("**Detecção de Picos**"); st.plotly_chart(fig_pk,use_container_width=False)
    has_ov = ov_abs > 0
    txt_boxes(apa_m_shape(),
              apa_r_shape(cv_f,cv_c,ov_pct,zc,pl,pf,pc_,fase,P2,P3,has_ov),
              "emb_shape")

with tab_adv:
    _mc_emb = [c for c in fall_ind.columns[1:]
               if pd.to_numeric(fall_ind[c], errors='coerce').notna().sum() > 3]
    render_advanced_tab(
        pd.concat([ctrl_ind.assign(Grupo="CONTROLE"), fall_ind.assign(Grupo="FALL")]),
        "CONTROLE", "FALL", ks="emb"
    )

st.markdown("---")
st.markdown("<p style='font-size:0.78rem;color:#9e9e9e;text-align:center'>FALL vs CONTROLE · Mann-Whitney · Cohen's d · BH-FDR · SPM · ROC · PCA · LDA · Bootstrap</p>",unsafe_allow_html=True)

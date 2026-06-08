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
    auc  = float(np.trapz(tprs, fprs))
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

    # ── Advanced analyses (expanders)
    with st.expander("🔴 Curvas ROC + AUC por métrica", expanded=False):
        render_roc(results, g1_s, g2_s, ks)

    with st.expander("🔴 Forest Plot — d de Cohen com IC 95%", expanded=False):
        render_forest(results, g1_s, g2_s)

    with st.expander("🔴 PCA — Análise de Componentes Principais", expanded=False):
        render_pca(results, g1_df, g2_df, g1_s, g2_s, g1_name, g2_name, ks)

    with st.expander("🟡 Heatmap de Correlação entre Métricas", expanded=False):
        render_heatmap(results, g1_df, g2_df, g1_s, g2_s)

    with st.expander("🟡 Análise de Cluster Hierárquica", expanded=False):
        render_cluster(results, g1_df, g2_df, g1_name, g2_name, g1_s, g2_s, ks)

    with st.expander("🟡 Score Composto de Risco (Fall Risk Score)", expanded=False):
        render_risk_score(results, g1_df, g2_df, g1_s, g2_s, g1_name, g2_name, ks)

    with st.expander("🟢 LDA — Análise Discriminante Linear", expanded=False):
        render_lda(results, g1_df, g2_df, g1_s, g2_s, ks)

    with st.expander("🟢 Bootstrap dos p-values + IC de Cohen's d", expanded=False):
        render_bootstrap(results, ks)


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
                    render_metrics_analysis(df_dev, g1_name, g2_name,
                                            key_suffix=f"{dk}_{sheet_name}")
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

tab1,tab3,tab5,tab_adv=st.tabs(["📈 Curvas Resultantes","🔬 Análise Temporal (SPM)","🧬 Análise da Forma das Curvas","🔬 Análises Avançadas"])

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

# ══════════════════════════════════════════════════════════════════════════════
# TAB ANÁLISES AVANÇADAS — modo embutido (usa metricas_individuais)
# ══════════════════════════════════════════════════════════════════════════════
with tab_adv:
    st.markdown("""<div class="info-box">
    Análises baseadas nos dados individuais dos sujeitos (<em>metricas_individuais</em>).
    Clique em cada seção para expandir.
    </div>""", unsafe_allow_html=True)

    # Prepara dados
    _mc = [c for c in fall_ind.columns[1:]
           if pd.to_numeric(fall_ind[c], errors='coerce').notna().sum() > 3]
    _g1_s, _g2_s = "Controle", "Fall"
    _results = compute_results(_mc, ctrl_ind, fall_ind)

    if not _results:
        st.warning("Dados individuais insuficientes.")
    else:
        with st.expander("🔴 Curvas ROC + AUC por métrica", expanded=False):
            render_roc(_results, _g1_s, _g2_s, "emb_roc")

        with st.expander("🔴 Forest Plot — d de Cohen com IC 95%", expanded=False):
            render_forest(_results, _g1_s, _g2_s)

        with st.expander("🔴 PCA — Análise de Componentes Principais", expanded=False):
            render_pca(_results, ctrl_ind, fall_ind, _g1_s, _g2_s, "CONTROLE", "FALL", "emb_pca")

        with st.expander("🟡 Heatmap de Correlação entre Métricas", expanded=False):
            render_heatmap(_results, ctrl_ind, fall_ind, _g1_s, _g2_s)

        with st.expander("🟡 Análise de Cluster Hierárquica", expanded=False):
            render_cluster(_results, ctrl_ind, fall_ind, "CONTROLE", "FALL", _g1_s, _g2_s, "emb_cl")

        with st.expander("🟡 Score Composto de Risco (Fall Risk Score)", expanded=False):
            render_risk_score(_results, ctrl_ind, fall_ind, _g1_s, _g2_s, "CONTROLE", "FALL", "emb_rs")

        with st.expander("🟢 LDA — Análise Discriminante Linear", expanded=False):
            render_lda(_results, ctrl_ind, fall_ind, _g1_s, _g2_s, "emb_lda")

        with st.expander("🟢 Bootstrap dos p-values + IC de Cohen's d", expanded=False):
            render_bootstrap(_results, "emb_boot")

st.markdown("---")
st.markdown("<p style='font-size:0.78rem;color:#9e9e9e;text-align:center'>FALL vs CONTROLE · Mann-Whitney · Cohen's d · BH-FDR · SPM · ROC · PCA · LDA · Bootstrap</p>",unsafe_allow_html=True)

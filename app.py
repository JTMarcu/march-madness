"""March Madness 2026 — Model Performance Dashboard.

Multi-page Streamlit app entry point. Shows tournament progress,
model accuracy, and per-game prediction tracking.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bracket import BracketPredictor, REGION_NAMES
from src.results import (
    load_results,
    build_performance_table,
    compute_metrics,
    compute_round_metrics,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

st.set_page_config(
    page_title="March Madness 2026",
    page_icon="\U0001f3c0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom theme CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #f0f6fc !important; }
p, span, label, .stCaption, [data-testid="stCaptionContainer"] { color: #8b949e !important; }

/* ── Metric cards ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-weight: 600 !important; text-transform: uppercase; font-size: .75rem !important; letter-spacing: .05em; }
[data-testid="stMetricDelta"] { color: #8b949e !important; }

/* ── Tables ────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Hero banner ───────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #0f2744 50%, #1a1a2e 100%);
    border: 1px solid #30363d; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,140,0,.12) 0%, transparent 70%);
}
.hero-banner h1 { margin: 0 0 4px 0; font-size: 2.2rem; color: #f0f6fc !important; }
.hero-banner p  { margin: 0; color: #8b949e !important; font-size: 1rem; }

/* ── Stat card ─────────────────────────────────────────── */
.stat-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; height: 100%;
}
.stat-card h3 { margin: 0 0 4px 0; font-size: 1.1rem; color: #f0f6fc !important; }
.stat-card .subtitle { color: #8b949e; font-size: .82rem; margin-bottom: 12px; }
.stat-card .item { padding: 8px 0; border-bottom: 1px solid #21262d; font-size: .9rem; color: #c9d1d9; }
.stat-card .item:last-child { border-bottom: none; }
.stat-card .item strong { color: #f0f6fc; }
.stat-card .empty { color: #484f58; font-style: italic; padding: 12px 0; }

/* ── Nav cards ─────────────────────────────────────────── */
.nav-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 20px 24px; text-align: center; transition: border-color .2s;
}
.nav-card:hover { border-color: #58a6ff; }
.nav-card .icon { font-size: 2rem; margin-bottom: 4px; }
.nav-card .label { color: #f0f6fc; font-weight: 600; font-size: 1rem; }
.nav-card .desc { color: #8b949e; font-size: .82rem; margin-top: 2px; }

/* ── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor() -> BracketPredictor:
    return BracketPredictor(MODELS_DIR)


def _check_models() -> bool:
    required = [
        "men_model.pkl", "women_model.pkl", "men_scaler.pkl",
        "women_scaler.pkl", "team_features.pkl", "teams.pkl", "config.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    # ── Hero Banner ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-banner">'
        '<h1>\U0001f3c0 March Madness 2026</h1>'
        '<p>Live model performance tracker \u2014 see how our ML predictions '
        'hold up against the real tournament</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not _check_models():
        st.error("**Models not found!** Run `python -m src.export_models` first.")
        st.stop()

    predictor = load_predictor()

    # ── Sidebar: Settings ─────────────────────────────────────────────────
    st.sidebar.header("Settings")
    gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], key="dash_gender")
    gender_code = "M" if gender == "Men's" else "W"

    sub_files = sorted(OUTPUT_DIR.glob("submission*.csv"))
    sub_names = [f.name for f in sub_files]
    default_idx = (
        sub_names.index("submission_refined.csv")
        if "submission_refined.csv" in sub_names else 0
    )
    selected_sub = st.sidebar.selectbox(
        "Submission to evaluate",
        sub_names,
        index=default_idx,
        help="Which submission CSV to compare against actual results",
    )
    sub_path = OUTPUT_DIR / selected_sub

    cfg = predictor.config
    feat_str = ", ".join(f.replace("Diff_", "") for f in predictor.features)
    with st.sidebar.expander("\U0001f916 Model Info", expanded=False):
        st.write(f"**Type:** Split M/W {cfg.get('model_type', 'logreg').upper()}")
        st.write(f"**Features:** {feat_str}")
        st.write(f"**Training:** {cfg['n_men_games']}M + {cfg['n_women_games']}W games")
        st.write(f"**Clipping:** [{cfg['clip_range'][0]}, {cfg['clip_range'][1]}]")

    # ── Load actual results & build perf table ────────────────────────────
    results_data = load_results()
    gender_key = "men" if gender_code == "M" else "women"
    actual_count = len(results_data.get(gender_key, {}))
    last_updated = results_data.get("last_updated", "unknown")

    perf_df = build_performance_table(gender_code, sub_path, season=2026)
    metrics = compute_metrics(perf_df)

    # ══════════════════════════════════════════════════════════════════════
    # KPI metrics
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(f"### \U0001f4ca {gender} Tournament")
    st.caption(f"Last updated: {last_updated} \u00b7 Evaluating: {selected_sub}")

    if actual_count == 0:
        st.markdown(
            '<div class="stat-card" style="text-align:center;padding:40px 24px;">'
            '<div style="font-size:3rem;margin-bottom:8px;">\U0001f4ed</div>'
            '<h3>No Results Yet</h3>'
            '<p style="color:#8b949e;margin:0;">Head to <b>\U0001f4dd Enter Results</b> '
            'in the sidebar to add game outcomes as they\'re played.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Games Played", metrics.get("n_games", 0))
        with m2:
            acc = metrics.get("accuracy", 0)
            n_correct = metrics.get("n_correct", 0)
            n_preds = metrics.get("n_with_preds", 0)
            st.metric("Accuracy", f"{acc:.1%}", delta=f"{n_correct}/{n_preds}")
        with m3:
            st.metric(
                "Running MSE",
                f"{metrics.get('mse', 0):.4f}",
                help="Mean Squared Error \u2014 how far off our predicted "
                     "probabilities are from reality. Lower is better. "
                     "Under 0.20 is strong; over 0.25 needs work.",
            )
        with m4:
            st.metric(
                "Log Loss",
                f"{metrics.get('log_loss', 0):.4f}",
                help="Log Loss \u2014 penalizes confident wrong predictions "
                     "harshly. Lower is better.",
            )

        # ── Per-round breakdown ───────────────────────────────────────────
        round_df = compute_round_metrics(perf_df)
        if not round_df.empty:
            st.markdown("#### Per-Round Breakdown")
            c1, c2 = st.columns([2, 3])
            with c1:
                display_rd = round_df.copy()
                display_rd["Accuracy"] = display_rd["Accuracy"].apply(
                    lambda x: f"{x:.1%}")
                display_rd["MSE"] = display_rd["MSE"].apply(
                    lambda x: f"{x:.4f}")
                st.dataframe(display_rd, hide_index=True, width="stretch")
            with c2:
                chart_data = round_df[["Round", "Accuracy"]].set_index("Round")
                st.bar_chart(chart_data, color="#58a6ff")

        # ── Game-by-game results ──────────────────────────────────────────
        st.markdown("#### Game-by-Game Results")
        if not perf_df.empty:
            display = perf_df.copy()
            display["P(Winner)"] = display["P(Winner)"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "\u2014"
            )
            display["MSE"] = display["MSE"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "\u2014"
            )
            display["Result"] = display["Correct"].apply(
                lambda x: "\u2705" if x is True else (
                    "\u274c" if x is False else "\u2014")
            )
            show_cols = [
                "Round", "Winner", "Loser", "Score",
                "P(Winner)", "Result", "MSE",
            ]
            st.dataframe(
                display[show_cols],
                hide_index=True,
                width="stretch",
                column_config={
                    "P(Winner)": st.column_config.TextColumn(
                        "P(Winner)",
                        help="Model's predicted probability for the actual winner",
                    ),
                    "Result": st.column_config.TextColumn(
                        "Correct?", width="small"),
                    "MSE": st.column_config.TextColumn(
                        "MSE Contrib",
                        help="Contribution to overall MSE",
                    ),
                },
            )

    # ══════════════════════════════════════════════════════════════════════
    # Surprised / Nailed It cards
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("")
    c_left, c_right = st.columns(2, gap="medium")

    with c_left:
        items_html = ""
        if not perf_df.empty:
            upsets = perf_df[perf_df["Correct"] == False].sort_values(
                "P(Winner)", ascending=True)
            if upsets.empty:
                items_html = (
                    '<div class="empty">All predictions correct so far! '
                    '\U0001f3af</div>')
            else:
                for _, row in upsets.head(5).iterrows():
                    p = row["P(Winner)"]
                    pstr = f"{p:.1%}" if pd.notna(p) else "?"
                    items_html += (
                        f'<div class="item">\U0001f534 '
                        f'<strong>{_esc(str(row["Winner"]))}</strong> beat '
                        f'{_esc(str(row["Loser"]))} '
                        f'({_esc(str(row["Score"]))}) '
                        f'\u2014 we gave them {pstr}</div>'
                    )
        else:
            items_html = '<div class="empty">No results yet</div>'
        st.markdown(
            '<div class="stat-card">'
            '<h3>\U0001f534 Surprised Us</h3>'
            '<div class="subtitle">Our model picked the wrong winner</div>'
            f'{items_html}'
            '</div>',
            unsafe_allow_html=True,
        )

    with c_right:
        items_html = ""
        if not perf_df.empty:
            best = perf_df[perf_df["Correct"] == True].sort_values(
                "P(Winner)", ascending=False)
            if best.empty:
                items_html = (
                    '<div class="empty">No correct predictions yet '
                    '\u2014 early rounds are the toughest</div>')
            else:
                for _, row in best.head(5).iterrows():
                    p = row["P(Winner)"]
                    pstr = f"{p:.1%}" if pd.notna(p) else "?"
                    items_html += (
                        f'<div class="item">\U0001f7e2 '
                        f'<strong>{_esc(str(row["Winner"]))}</strong> beat '
                        f'{_esc(str(row["Loser"]))} '
                        f'({_esc(str(row["Score"]))}) '
                        f'\u2014 predicted at {pstr}</div>'
                    )
        else:
            items_html = '<div class="empty">No results yet</div>'
        st.markdown(
            '<div class="stat-card">'
            '<h3>\U0001f7e2 Nailed It</h3>'
            '<div class="subtitle">Our model correctly picked the winner</div>'
            f'{items_html}'
            '</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Navigation cards
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("")
    nav1, nav2 = st.columns(2, gap="medium")
    with nav1:
        st.markdown(
            '<div class="nav-card">'
            '<div class="icon">\U0001f3c6</div>'
            '<div class="label">Bracket Predictor</div>'
            '<div class="desc">Build your bracket with AI-powered '
            'win probabilities</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with nav2:
        st.markdown(
            '<div class="nav-card">'
            '<div class="icon">\U0001f4dd</div>'
            '<div class="label">Enter Results</div>'
            '<div class="desc">Log real game outcomes to track '
            'model accuracy</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.caption("Use the sidebar to navigate between pages.")


if __name__ == "__main__":
    main()

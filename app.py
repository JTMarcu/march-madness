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
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> BracketPredictor:
    return BracketPredictor(MODELS_DIR)


def _check_models() -> bool:
    required = [
        "men_model.pkl", "women_model.pkl", "men_scaler.pkl",
        "women_scaler.pkl", "team_features.pkl", "teams.pkl", "config.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


def main():
    st.title("🏀 March Madness 2026")
    st.markdown(
        "Track how our ML model performs against the real NCAA tournament. "
        "We predicted win probabilities for every possible matchup — "
        "now let's see how those predictions hold up."
    )

    if not _check_models():
        st.error("**Models not found!** Run `python -m src.export_models` first.")
        st.stop()

    predictor = load_predictor()

    # ── Sidebar: Settings ─────────────────────────────────────────────────
    st.sidebar.header("Settings")
    gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], key="dash_gender")
    gender_code = "M" if gender == "Men's" else "W"

    # Pick submission file
    sub_files = sorted(OUTPUT_DIR.glob("submission*.csv"))
    sub_names = [f.name for f in sub_files]
    default_idx = sub_names.index("submission_refined.csv") if "submission_refined.csv" in sub_names else 0
    selected_sub = st.sidebar.selectbox(
        "Submission to evaluate",
        sub_names,
        index=default_idx,
        help="Which submission CSV to compare against actual results",
    )
    sub_path = OUTPUT_DIR / selected_sub

    # ── Model info card ───────────────────────────────────────────────────
    cfg = predictor.config
    feat_str = ", ".join(f.replace("Diff_", "") for f in predictor.features)
    with st.sidebar.expander("🤖 Model Info", expanded=False):
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
    # Top-level metrics
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader(f"📊 {gender} Tournament — Model Performance")
    st.caption(f"Last updated: {last_updated} · Evaluating: {selected_sub}")

    if actual_count == 0:
        st.info(
            "No actual results recorded yet. Head to **📝 Enter Results** "
            "in the sidebar to add game outcomes as they're played.",
            icon="📭",
        )
    else:
        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Games Played", metrics.get("n_games", 0))
        with m2:
            acc = metrics.get("accuracy", 0)
            n_correct = metrics.get("n_correct", 0)
            n_preds = metrics.get("n_with_preds", 0)
            st.metric("Accuracy", f"{acc:.1%}", delta=f"{n_correct}/{n_preds}")
        with m3:
            mse = metrics.get("mse", 0)
            st.metric(
                "Running MSE",
                f"{mse:.4f}",
                help="Mean Squared Error — measures how far off our predicted probabilities are from reality. Lower is better. Under 0.20 is strong; over 0.25 needs work.",
            )
        with m4:
            ll = metrics.get("log_loss", 0)
            st.metric(
                "Log Loss",
                f"{ll:.4f}",
                help="Log Loss — penalizes confident wrong predictions harshly. Lower is better.",
            )

        # ── Per-round breakdown ───────────────────────────────────────────
        round_df = compute_round_metrics(perf_df)
        if not round_df.empty:
            st.markdown("#### Per-Round Breakdown")
            c1, c2 = st.columns([2, 3])
            with c1:
                display_rd = round_df.copy()
                display_rd["Accuracy"] = display_rd["Accuracy"].apply(lambda x: f"{x:.1%}")
                display_rd["MSE"] = display_rd["MSE"].apply(lambda x: f"{x:.4f}")
                st.dataframe(display_rd, hide_index=True, width='stretch')
            with c2:
                # Simple accuracy bar chart
                chart_data = round_df[["Round", "Accuracy"]].set_index("Round")
                st.bar_chart(chart_data, color="#28a745")

        # ── Game-by-game results ──────────────────────────────────────────
        st.markdown("#### Game-by-Game Results")

        # Style the table
        if not perf_df.empty:
            display = perf_df.copy()
            display["P(Winner)"] = display["P(Winner)"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
            display["MSE"] = display["MSE"].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "—"
            )
            display["Result"] = display["Correct"].apply(
                lambda x: "✅" if x is True else ("❌" if x is False else "—")
            )
            show_cols = ["Round", "Winner", "Loser", "Score", "P(Winner)", "Result", "MSE"]
            st.dataframe(
                display[show_cols],
                hide_index=True,
                width='stretch',
                column_config={
                    "P(Winner)": st.column_config.TextColumn("P(Winner)", help="Model's predicted probability for the actual winner"),
                    "Result": st.column_config.TextColumn("Correct?", width="small"),
                    "MSE": st.column_config.TextColumn("MSE Contrib", help="Contribution to overall MSE"),
                },
            )

    # ══════════════════════════════════════════════════════════════════════
    # Side-by-side: Upcoming games & bracket preview
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("� Surprised Us")
        st.caption("Games where our model picked the wrong winner")
        if not perf_df.empty:
            upsets = perf_df[perf_df["Correct"] == False].sort_values(
                "P(Winner)", ascending=True
            )
            if upsets.empty:
                st.success("All predictions correct so far! 🎯")
            else:
                for _, row in upsets.head(5).iterrows():
                    p = row["P(Winner)"]
                    pstr = f"{p:.1%}" if pd.notna(p) else "?"
                    st.write(
                        f"**{row['Winner']}** beat {row['Loser']} "
                        f"({row['Score']}) — our model gave them {pstr}"
                    )
        else:
            st.caption("No results yet.")

    with c_right:
        st.subheader("🟢 Nailed It")
        st.caption("Games where our model correctly picked the winner")
        if not perf_df.empty:
            best = perf_df[perf_df["Correct"] == True].sort_values(
                "P(Winner)", ascending=False
            )
            if best.empty:
                st.info("No correct predictions yet — early games are often the hardest to call.")
            else:
                for _, row in best.head(5).iterrows():
                    p = row["P(Winner)"]
                    pstr = f"{p:.1%}" if pd.notna(p) else "?"
                    st.write(
                        f"**{row['Winner']}** beat {row['Loser']} "
                        f"({row['Score']}) — predicted at {pstr}"
                    )
        else:
            st.caption("No results yet.")

    # ══════════════════════════════════════════════════════════════════════
    # Navigation hints
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        st.info("**🏆 Bracket** — Build your bracket with our model's predictions", icon="👈")
    with nav2:
        st.info("**📝 Enter Results** — Add real game outcomes as they happen", icon="👈")
    st.caption("Use the sidebar to navigate between pages.")


if __name__ == "__main__":
    main()

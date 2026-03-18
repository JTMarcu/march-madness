"""📝 Enter Tournament Results.

Add actual game outcomes as they happen to track model performance.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bracket import BracketPredictor, BracketSimulator, REGION_NAMES
from src.results import (
    load_results,
    add_result,
    remove_result,
    ROUND_ORDER,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Round display names for slots
_ROUND_LABELS = {
    "R1": "Round of 64",
    "R2": "Round of 32",
    "R3": "Sweet 16",
    "R4": "Elite 8",
    "R5": "Final Four",
    "R6": "Championship",
}

_REGION_NAMES = {
    "W": "East",
    "X": "South",
    "Y": "Midwest",
    "Z": "West",
}


def _readable_game_label(slot: str, s_name: str, w_name: str) -> str:
    """Build a human-readable label for a game slot.

    Examples:
        'R1W1' → 'Round of 64 — East: Duke vs Stetson'
        'Y16' → 'Play-In: Howard vs UMBC'
    """
    if not slot.startswith("R"):
        return f"Play-In: {s_name} vs {w_name}"
    rnd = slot[:2]
    rlabel = _ROUND_LABELS.get(rnd, rnd)
    if rnd in ("R5", "R6"):
        return f"{rlabel}: {s_name} vs {w_name}"
    region_code = slot[2] if len(slot) > 2 else ""
    region_name = _REGION_NAMES.get(region_code, region_code)
    return f"{rlabel} — {region_name}: {s_name} vs {w_name}"

st.set_page_config(
    page_title="Enter Results",
    page_icon="\U0001f4dd",
    layout="wide",
)

# ── Dark theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
    header[data-testid="stHeader"] { background: transparent !important; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #30363d; }
    section[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
    /* Typography */
    h1, h2, h3, h4 { color: #f0f6fc !important; }
    .stMarkdown p, .stMarkdown li { color: #c9d1d9; }
    /* Form / inputs */
    .stForm { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1.2rem; }
    /* Buttons */
    .stButton > button[kind="primary"] { background: #238636 !important; border-color: #2ea043 !important; }
    /* Metric colours */
    [data-testid="stMetricValue"] { color: #58a6ff !important; }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor() -> BracketPredictor:
    return BracketPredictor(MODELS_DIR)


def main():
    st.title("📝 Enter Tournament Results")
    st.markdown(
        "Record real game outcomes here as they happen. "
        "Results will appear on the **Dashboard** to track model performance "
        "and will be locked into the **Bracket** view."
    )

    predictor = load_predictor()

    # Sidebar
    st.sidebar.header("Settings")
    gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], key="res_gender")
    gender_code = "M" if gender == "Men's" else "W"

    # Load current results
    results_data = load_results()
    gender_key = "men" if gender_code == "M" else "women"
    current_results = results_data.get(gender_key, {})

    # ══════════════════════════════════════════════════════════════════════
    # Quick-Add Form
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("➕ Add Game Result")

    # Load bracket data to get team info
    sim = BracketSimulator(predictor, 2026, gender_code, DATA_DIR)
    slots_df = sim.slots.sort_values("Slot")

    # Show available games that don't have results yet
    unplayed = []
    for _, row in slots_df.iterrows():
        slot = row["Slot"]
        if slot not in current_results:
            s_id = sim.get_team_for_code(row["StrongSeed"])
            w_id = sim.get_team_for_code(row["WeakSeed"])
            if s_id is not None and w_id is not None:
                s_name = predictor.team_name(s_id)
                w_name = predictor.team_name(w_id)
                unplayed.append({
                    "slot": slot,
                    "strong_id": s_id,
                    "weak_id": w_id,
                    "strong_name": s_name,
                    "weak_name": w_name,
                    "label": _readable_game_label(slot, s_name, w_name),
                })

    if not unplayed:
        st.info("All available games have results or teams aren't resolved yet.")
    else:
        with st.form("add_result_form", clear_on_submit=True):
            # Game selector
            game_labels = {g["slot"]: g["label"] for g in unplayed}
            selected_slot = st.selectbox(
                "Select game",
                [g["slot"] for g in unplayed],
                format_func=lambda s: game_labels.get(s, s),
            )

            game_info = next(g for g in unplayed if g["slot"] == selected_slot)

            # Winner selector
            winner_options = {
                game_info["strong_id"]: game_info["strong_name"],
                game_info["weak_id"]: game_info["weak_name"],
            }
            winner_id = st.radio(
                "Winner",
                list(winner_options.keys()),
                format_func=lambda tid: winner_options[tid],
                horizontal=True,
            )

            # Scores
            sc1, sc2 = st.columns(2)
            with sc1:
                winner_score = st.number_input(
                    "Winner's score", min_value=0, max_value=200,
                    value=70, step=1,
                )
            with sc2:
                loser_score = st.number_input(
                    "Loser's score", min_value=0, max_value=200,
                    value=65, step=1,
                )

            submitted = st.form_submit_button("✅ Save Result", type="primary")

            if submitted:
                loser_id = (game_info["weak_id"]
                            if winner_id == game_info["strong_id"]
                            else game_info["strong_id"])
                winner_name = winner_options[winner_id]
                loser_name = winner_options[loser_id]

                add_result(
                    gender=gender_code,
                    slot=selected_slot,
                    winner_id=winner_id,
                    winner_name=winner_name,
                    loser_id=loser_id,
                    loser_name=loser_name,
                    winner_score=int(winner_score),
                    loser_score=int(loser_score),
                )
                st.success(
                    f"Saved: **{winner_name} {int(winner_score)}** — "
                    f"{loser_name} {int(loser_score)}"
                )
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # Current Results
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader(f"📋 Current {gender} Results ({len(current_results)} games)")

    if not current_results:
        st.caption("No results recorded yet. Use the form above to add games as they're played.")
    else:
        # Sort by round order
        sorted_results = sorted(
            current_results.items(),
            key=lambda x: ROUND_ORDER.get(x[1].get("round", ""), 99),
        )

        for slot, info in sorted_results:
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                round_name = info.get("round", slot)
                winner = info.get("winner_name", "?")
                loser = info.get("loser_name", "?")
                score = info.get("score", "")
                st.write(
                    f"**{round_name}** ({slot}): "
                    f"**{winner}** def. {loser} ({score})"
                )
            with c2:
                # Show our prediction
                sub_path = OUTPUT_DIR / "submission_refined.csv"
                if sub_path.exists():
                    from src.results import get_prediction_for_matchup
                    w_id = info.get("winner_id")
                    l_id = info.get("loser_id")
                    if w_id and l_id:
                        pred = get_prediction_for_matchup(sub_path, 2026, w_id, l_id)
                        if pred is not None:
                            lo = min(w_id, l_id)
                            p_winner = pred if lo == w_id else 1.0 - pred
                            color = "green" if p_winner > 0.5 else "red"
                            st.markdown(f":{color}[P={p_winner:.1%}]")
            with c3:
                if st.button("🗑️", key=f"del_{slot}", help="Remove this result"):
                    remove_result(gender_code, slot)
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # Bulk import section
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    with st.expander("⚙️ Advanced: Bulk Import (JSON)", expanded=False):
        st.caption(
            "For power users — paste a JSON object to import multiple results at once. "
            "Each key is a bracket slot code, and the value contains winner/loser info."
        )
        st.code(
            '{"R1W1": {"winner_id": 1181, "winner_name": "Duke", '
            '"loser_id": 1371, "loser_name": "Stetson", "score": "85-60"}}',
            language="json",
        )
        json_input = st.text_area("JSON data", height=200, key="bulk_json")
        if st.button("Import", key="bulk_import"):
            import json
            try:
                bulk = json.loads(json_input)
                data = load_results()
                for slot, info in bulk.items():
                    info["round"] = info.get("round", "")
                    data[gender_key][slot] = info
                from src.results import save_results
                save_results(data)
                st.success(f"Imported {len(bulk)} results.")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")


if __name__ == "__main__":
    main()

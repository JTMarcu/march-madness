"""March Madness Bracket Predictor — Streamlit App.

Interactive bracket-filling tool powered by trained ML models.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bracket import BracketPredictor, BracketSimulator, ROUND_NAMES, REGION_NAMES

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="March Madness Bracket Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor() -> BracketPredictor:
    """Load trained models (cached across reruns)."""
    return BracketPredictor(MODELS_DIR)


def check_models_exist() -> bool:
    """Check if exported model files exist."""
    required = ["men_model.pkl", "women_model.pkl", "men_scaler.pkl",
                "women_scaler.pkl", "team_features.pkl", "teams.pkl", "config.json"]
    return all((MODELS_DIR / f).exists() for f in required)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_seed_display(seed_str: str) -> str:
    """Format seed string for display. 'W01' -> '(1)'."""
    try:
        num = int(seed_str[1:3])
        return f"({num})"
    except (ValueError, IndexError):
        return ""


def prob_color(prob: float) -> str:
    """Return a color based on confidence level."""
    confidence = abs(prob - 0.5) * 2  # 0 = toss-up, 1 = certain
    if confidence > 0.6:
        return "#2ecc71"  # green — strong pick
    elif confidence > 0.3:
        return "#f39c12"  # yellow — moderate
    else:
        return "#e74c3c"  # red — close game


def format_prob(prob: float, team_id: int, strong_team: int) -> str:
    """Format probability from perspective of displayed team."""
    if team_id == strong_team:
        return f"{prob:.1%}"
    else:
        return f"{1 - prob:.1%}"


# ---------------------------------------------------------------------------
# Bracket display
# ---------------------------------------------------------------------------
def render_matchup_card(
    matchup: dict,
    slot: str,
    sim: BracketSimulator,
    key_prefix: str,
) -> int | None:
    """Render a single matchup card with team selection.

    Returns the selected winner TeamID.
    """
    t1_id = matchup["strong_team"]
    t2_id = matchup["weak_team"]
    t1_name = matchup["strong_name"]
    t2_name = matchup["weak_name"]
    prob = matchup["probability"]

    if t1_id is None or t2_id is None:
        st.markdown(f"**{slot}**: *Waiting for prior results...*")
        return None

    # Get seed displays
    t1_seed = ""
    t2_seed = ""
    for _, srow in sim.seeds.iterrows():
        if srow["TeamID"] == t1_id:
            t1_seed = get_seed_display(srow["Seed"])
        if srow["TeamID"] == t2_id:
            t2_seed = get_seed_display(srow["Seed"])

    # Determine current selection
    current_winner = sim.results.get(slot)

    # Format probabilities
    if prob is not None:
        t1_prob = prob
        t2_prob = 1 - prob
        color = prob_color(prob)
    else:
        t1_prob = t2_prob = 0.5
        color = "#95a5a6"

    # Radio button for team selection
    options = [
        f"{t1_seed} {t1_name} — {t1_prob:.1%}",
        f"{t2_seed} {t2_name} — {t2_prob:.1%}",
    ]

    # Determine default index
    if current_winner == t2_id:
        default_idx = 1
    else:
        default_idx = 0  # Default to favorite

    selected = st.radio(
        f"**{slot}**",
        options,
        index=default_idx,
        key=f"{key_prefix}_{slot}",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Parse selection
    if selected == options[0]:
        return t1_id
    else:
        return t2_id


def render_region(
    sim: BracketSimulator,
    region: str,
    region_name: str,
) -> None:
    """Render all rounds for a single region."""
    st.subheader(f"🏀 {region_name} Region")

    # Round 1
    with st.expander(f"Round of 64 — {region_name}", expanded=True):
        r1_slots = sim.slots[
            (sim.slots["Slot"].str.startswith("R1")) &
            (sim.slots["Slot"].str[2] == region)
        ].sort_values("Slot")

        for _, row in r1_slots.iterrows():
            slot = row["Slot"]
            strong_team = sim.get_team_for_code(row["StrongSeed"])
            weak_team = sim.get_team_for_code(row["WeakSeed"])

            prob = None
            if strong_team and weak_team:
                prob = sim.predictor.predict_matchup(
                    strong_team, weak_team, sim.season, sim.gender
                )

            matchup = {
                "strong_team": strong_team,
                "weak_team": weak_team,
                "strong_name": sim.predictor.team_name(strong_team) if strong_team else "TBD",
                "weak_name": sim.predictor.team_name(weak_team) if weak_team else "TBD",
                "probability": prob,
            }

            winner = render_matchup_card(matchup, slot, sim, f"r1_{region}")
            if winner is not None:
                sim.results[slot] = winner
                sim.probabilities[slot] = prob if prob else 0.5

            st.divider()

    # Rounds 2-4 (within region)
    for round_num in range(2, 5):
        round_name = ROUND_NAMES.get(f"R{round_num}", f"Round {round_num}")
        prefix = f"R{round_num}{region}"

        round_slots = sim.slots[
            sim.slots["Slot"].str.startswith(prefix)
        ].sort_values("Slot")

        if len(round_slots) == 0:
            continue

        with st.expander(f"{round_name} — {region_name}", expanded=(round_num <= 2)):
            for _, row in round_slots.iterrows():
                slot = row["Slot"]
                strong_team = sim.get_team_for_code(row["StrongSeed"])
                weak_team = sim.get_team_for_code(row["WeakSeed"])

                prob = None
                if strong_team and weak_team:
                    prob = sim.predictor.predict_matchup(
                        strong_team, weak_team, sim.season, sim.gender
                    )

                matchup = {
                    "strong_team": strong_team,
                    "weak_team": weak_team,
                    "strong_name": sim.predictor.team_name(strong_team) if strong_team else "TBD",
                    "weak_name": sim.predictor.team_name(weak_team) if weak_team else "TBD",
                    "probability": prob,
                }

                winner = render_matchup_card(matchup, slot, sim, f"r{round_num}_{region}")
                if winner is not None:
                    sim.results[slot] = winner
                    sim.probabilities[slot] = prob if prob else 0.5

                st.divider()


def render_final_four(sim: BracketSimulator) -> None:
    """Render Final Four and Championship."""
    st.subheader("🏆 Final Four & Championship")

    # Final Four (R5)
    r5_slots = sim.slots[sim.slots["Slot"].str.startswith("R5")].sort_values("Slot")
    with st.expander("Final Four", expanded=True):
        for _, row in r5_slots.iterrows():
            slot = row["Slot"]
            strong_team = sim.get_team_for_code(row["StrongSeed"])
            weak_team = sim.get_team_for_code(row["WeakSeed"])

            prob = None
            if strong_team and weak_team:
                prob = sim.predictor.predict_matchup(
                    strong_team, weak_team, sim.season, sim.gender
                )

            matchup = {
                "strong_team": strong_team,
                "weak_team": weak_team,
                "strong_name": sim.predictor.team_name(strong_team) if strong_team else "TBD",
                "weak_name": sim.predictor.team_name(weak_team) if weak_team else "TBD",
                "probability": prob,
            }

            winner = render_matchup_card(matchup, slot, sim, "r5")
            if winner is not None:
                sim.results[slot] = winner
                sim.probabilities[slot] = prob if prob else 0.5
            st.divider()

    # Championship (R6)
    r6_slots = sim.slots[sim.slots["Slot"].str.startswith("R6")].sort_values("Slot")
    with st.expander("🏆 Championship", expanded=True):
        for _, row in r6_slots.iterrows():
            slot = row["Slot"]
            strong_team = sim.get_team_for_code(row["StrongSeed"])
            weak_team = sim.get_team_for_code(row["WeakSeed"])

            prob = None
            if strong_team and weak_team:
                prob = sim.predictor.predict_matchup(
                    strong_team, weak_team, sim.season, sim.gender
                )

            matchup = {
                "strong_team": strong_team,
                "weak_team": weak_team,
                "strong_name": sim.predictor.team_name(strong_team) if strong_team else "TBD",
                "weak_name": sim.predictor.team_name(weak_team) if weak_team else "TBD",
                "probability": prob,
            }

            winner = render_matchup_card(matchup, slot, sim, "r6")
            if winner is not None:
                sim.results[slot] = winner
                sim.probabilities[slot] = prob if prob else 0.5

    # Show champion
    champion_id = sim.get_champion()
    if champion_id:
        champ_name = sim.predictor.team_name(champion_id)
        st.success(f"🏆 **Predicted Champion: {champ_name}**")


# ---------------------------------------------------------------------------
# Auto-fill bracket
# ---------------------------------------------------------------------------
def auto_fill_bracket(sim: BracketSimulator, mode: str = "deterministic") -> None:
    """Auto-fill the entire bracket using model predictions."""
    sim.results.clear()
    sim.probabilities.clear()
    sim.simulate_full_bracket(mode=mode)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("🏀 March Madness Bracket Predictor")
    st.caption("Powered by split M/W logistic regression trained on 2010–2025 tournament data")

    # Check if models exist
    if not check_models_exist():
        st.error(
            "**Models not found!** Run the export script first:\n\n"
            "```bash\npython -m src.export_models\n```\n\n"
            "This trains the final models and saves them to `models/`."
        )
        st.stop()

    # Load predictor
    predictor = load_predictor()

    # --------------- Sidebar ---------------
    st.sidebar.header("Settings")

    gender = st.sidebar.radio(
        "Tournament",
        ["Men's", "Women's"],
        index=0,
    )
    gender_code = "M" if gender == "Men's" else "W"

    current_season = predictor.config.get("current_season", 2026)
    season = st.sidebar.number_input(
        "Season",
        min_value=2011,
        max_value=current_season,
        value=current_season,
        help="Select a season to simulate. Use current year for predictions, or a past year to test.",
    )

    # Check if seeds exist for this season
    prefix = "M" if gender_code == "M" else "W"
    seeds_df = pd.read_csv(DATA_DIR / f"{prefix}NCAATourneySeeds.csv")
    available_seasons = sorted(seeds_df["Season"].unique())

    if season not in available_seasons:
        st.warning(
            f"⚠️ No seeds available for {season} {gender} tournament yet. "
            f"Seeds are announced on Selection Sunday. "
            f"Available seasons: {available_seasons[-5:]}"
        )
        st.info("Try a past season like 2025 to test the bracket predictor.")
        # Allow continuing with a different season
        season = st.sidebar.selectbox(
            "Or pick a past season:",
            sorted(available_seasons, reverse=True)[:10],
        )

    sim_mode = st.sidebar.radio(
        "Simulation Mode",
        ["Deterministic (pick favorites)", "Probabilistic (weighted random)"],
        help="Deterministic always picks the team with >50% win probability. "
             "Probabilistic randomly picks weighted by the probabilities.",
    )
    mode = "deterministic" if "Deterministic" in sim_mode else "probabilistic"

    # Initialize simulator in session state
    sim_key = f"sim_{gender_code}_{season}"
    if sim_key not in st.session_state:
        st.session_state[sim_key] = BracketSimulator(
            predictor, season, gender_code, DATA_DIR
        )

    sim = st.session_state[sim_key]

    # Auto-fill button
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("🤖 Auto-Fill", width="stretch"):
            auto_fill_bracket(sim, mode)
            st.rerun()
    with col2:
        if st.button("🔄 Reset", width="stretch"):
            st.session_state[sim_key] = BracketSimulator(
                predictor, season, gender_code, DATA_DIR
            )
            st.rerun()
    with col3:
        if st.button("🎲 Random", width="stretch"):
            auto_fill_bracket(sim, "probabilistic")
            st.rerun()

    # --------------- Seeded Teams ---------------
    with st.sidebar.expander("📋 Seeded Teams", expanded=False):
        seed_display = sim.seeds.copy()
        seed_display["Team"] = seed_display["TeamID"].apply(predictor.team_name)
        seed_display = seed_display[["Seed", "Team", "TeamID"]].sort_values("Seed")
        st.dataframe(seed_display, hide_index=True, width="stretch")

    # --------------- Model Info ---------------
    with st.sidebar.expander("📊 Model Info", expanded=False):
        st.write(f"**Model:** Split M/W Logistic Regression")
        st.write(f"**Features:** {', '.join(predictor.features)}")
        st.write(f"**Training:** {predictor.config['n_men_games']} men's + "
                 f"{predictor.config['n_women_games']} women's games")
        st.write(f"**Seasons:** 2010–2025 (excl. 2020)")
        st.write(f"**Validated MSE:** 0.1604 (3-year holdout avg)")

    # --------------- Main Bracket ---------------
    st.markdown(f"### {season} NCAA {gender} Tournament")

    # Tab layout: one tab per region + Final Four
    regions = sorted(sim.slots[
        sim.slots["Slot"].str.startswith("R1")
    ]["Slot"].str[2].unique())

    tab_names = [REGION_NAMES.get(r, r) for r in regions] + ["Final Four"]
    tabs = st.tabs(tab_names)

    for i, region in enumerate(regions):
        with tabs[i]:
            render_region(sim, region, REGION_NAMES.get(region, region))

    # Final Four tab
    with tabs[-1]:
        render_final_four(sim)

    # --------------- Bracket Summary Table ---------------
    if sim.results:
        st.markdown("---")
        st.subheader("📊 Full Bracket Results")

        summary = sim.get_bracket_summary()
        results_with_data = [s for s in summary if s["winner_id"] is not None]

        if results_with_data:
            df = pd.DataFrame(results_with_data)
            df = df[["round", "team1_name", "team2_name", "winner_name", "probability"]]
            df.columns = ["Round", "Team 1", "Team 2", "Winner", "P(Team 1)"]
            df["P(Team 1)"] = df["P(Team 1)"].apply(
                lambda x: f"{x:.1%}" if x is not None else "N/A"
            )
            st.dataframe(df, hide_index=True, width="stretch")

    # --------------- Matchup Explorer ---------------
    st.markdown("---")
    st.subheader("🔍 Head-to-Head Matchup Explorer")

    col1, col2 = st.columns(2)
    teams_list = sorted(sim.seeds["TeamID"].apply(predictor.team_name).tolist())

    with col1:
        team1_name = st.selectbox("Team 1", teams_list, key="explore_t1")
    with col2:
        team2_name = st.selectbox("Team 2",
                                  [t for t in teams_list if t != team1_name],
                                  key="explore_t2")

    if team1_name and team2_name:
        # Look up IDs
        t1_id = sim.seeds[
            sim.seeds["TeamID"].apply(predictor.team_name) == team1_name
        ]["TeamID"].values
        t2_id = sim.seeds[
            sim.seeds["TeamID"].apply(predictor.team_name) == team2_name
        ]["TeamID"].values

        if len(t1_id) > 0 and len(t2_id) > 0:
            prob = predictor.predict_matchup(
                int(t1_id[0]), int(t2_id[0]), season, gender_code
            )

            # Display
            mcol1, mcol2, mcol3 = st.columns([2, 1, 2])
            with mcol1:
                st.metric(team1_name, f"{prob:.1%}")
            with mcol2:
                st.markdown("### vs")
            with mcol3:
                st.metric(team2_name, f"{1 - prob:.1%}")

            # Confidence bar
            st.progress(prob)


if __name__ == "__main__":
    main()

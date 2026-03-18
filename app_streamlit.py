"""March Madness Bracket Predictor — Streamlit App with SVG Bracket.

Interactive tournament bracket with traditional bracket visualization.
Run with: streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bracket import BracketPredictor, BracketSimulator, ROUND_NAMES, REGION_NAMES

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"


# ═══════════════════════════════════════════════════════════════════════════
# SVG Layout Constants
# ═══════════════════════════════════════════════════════════════════════════
BOX_W = 95       # team-name box width
BOX_H = 18       # team-name box height
CONN_W = 10      # connector-line horizontal span
GAME_H = BOX_H * 2 + 2          # 38 px per game (2 teams + divider)
GAME_GAP = 6                     # vertical gap between R1 games
REGION_GAP = 30                  # gap between top / bottom regions
PAD = 20                         # SVG padding
COL_W = BOX_W + CONN_W          # 105 px per round column

# Standard NCAA bracket visual order for R1 within a region
# 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
R1_ORDER = [1, 8, 5, 4, 6, 3, 7, 2]
R2_ORDER = [1, 4, 3, 2]
R3_ORDER = [1, 2]
R4_ORDER = [1]

# Derived layout values
REGION_H = 8 * (GAME_H + GAME_GAP) - GAME_GAP  # 346
SVG_H = PAD * 2 + REGION_H * 2 + REGION_GAP     # 762

LEFT_X = [PAD + i * COL_W for i in range(5)]     # R1, R2, S16, E8, FF
CH_X = PAD + 5 * COL_W                            # championship column
RIGHT_X = [CH_X + BOX_W + CONN_W + i * COL_W for i in range(5)]  # FF..R1
SVG_W = RIGHT_X[4] + BOX_W + PAD

TOP_Y = PAD
BOT_Y = PAD + REGION_H + REGION_GAP

HEADERS_L = ["R64", "R32", "Sweet 16", "Elite 8", "Final Four"]
HEADERS_R = ["Final Four", "Elite 8", "Sweet 16", "R32", "R64"]


# ═══════════════════════════════════════════════════════════════════════════
# Page config & cached loading
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="March Madness Bracket Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> BracketPredictor:
    """Load trained models (cached across reruns)."""
    return BracketPredictor(MODELS_DIR)


def check_models_exist() -> bool:
    required = [
        "men_model.pkl", "women_model.pkl", "men_scaler.pkl",
        "women_scaler.pkl", "team_features.pkl", "teams.pkl", "config.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


# ═══════════════════════════════════════════════════════════════════════════
# SVG primitive helpers
# ═══════════════════════════════════════════════════════════════════════════
def _game_centers(n: int, y_start: float) -> list[float]:
    """Y-centres for *n* games starting at *y_start* (top of first game)."""
    return [y_start + i * (GAME_H + GAME_GAP) + GAME_H / 2 for i in range(n)]


def _midpoints(centers: list[float]) -> list[float]:
    return [(centers[2 * i] + centers[2 * i + 1]) / 2
            for i in range(len(centers) // 2)]


def _seed_num(sim: BracketSimulator, tid: int) -> Optional[int]:
    if tid is None:
        return None
    row = sim.seeds[sim.seeds["TeamID"] == tid]
    if len(row) == 0:
        return None
    try:
        return int(row.iloc[0]["Seed"][1:3])
    except (ValueError, IndexError):
        return None


def _trunc(name: str, maxlen: int = 12) -> str:
    return name if len(name) <= maxlen else name[: maxlen - 1] + "\u2026"


def _esc(text: str) -> str:
    """Escape text for SVG/XML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _team_box(x: float, y: float, name: str, seed: Optional[int],
              is_winner: bool = False) -> str:
    if name in ("TBD", None, ""):
        fill, stroke, tfill = "#f0f0f0", "#ddd", "#bbb"
        weight, display = "normal", "TBD"
    elif is_winner:
        fill, stroke, tfill = "#c3e6cb", "#28a745", "#155724"
        weight = "bold"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    else:
        fill, stroke, tfill = "#fff", "#999", "#333"
        weight = "normal"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    display = _esc(display)
    return (
        f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n'
        f'<text x="{x + 3}" y="{y + BOX_H - 5}" font-size="9" '
        f'fill="{tfill}" font-weight="{weight}" '
        f'font-family="Arial,sans-serif">{display}</text>\n'
    )


def _game_box(x: float, cy: float, gd: dict) -> str:
    """Two team boxes for one game, centred at *cy*."""
    top_y = cy - GAME_H / 2
    bot_y = top_y + BOX_H + 2
    winner = gd.get("winner_id")
    svg = _team_box(x, top_y, gd["strong_name"], gd["strong_seed"],
                    is_winner=(winner is not None and winner == gd["strong_id"]))
    svg += _team_box(x, bot_y, gd["weak_name"], gd["weak_seed"],
                     is_winner=(winner is not None and winner == gd["weak_id"]))
    return svg


def _connector(x_edge: float, y1: float, y2: float,
               x_next: float, direction: str = "right") -> str:
    """Bracket connector: two horizontals, one vertical, one horizontal."""
    ym = (y1 + y2) / 2
    if direction == "right":
        mx = x_edge + CONN_W / 2
    else:
        mx = x_edge - CONN_W / 2
    return (
        f'<line x1="{x_edge}" y1="{y1}" x2="{mx}" y2="{y1}" '
        f'stroke="#999" stroke-width="1"/>\n'
        f'<line x1="{x_edge}" y1="{y2}" x2="{mx}" y2="{y2}" '
        f'stroke="#999" stroke-width="1"/>\n'
        f'<line x1="{mx}" y1="{y1}" x2="{mx}" y2="{y2}" '
        f'stroke="#999" stroke-width="1"/>\n'
        f'<line x1="{mx}" y1="{ym}" x2="{x_next}" y2="{ym}" '
        f'stroke="#999" stroke-width="1"/>\n'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Game data collector
# ═══════════════════════════════════════════════════════════════════════════
def _collect_games(sim: BracketSimulator,
                   predictor: BracketPredictor) -> dict[str, dict]:
    games: dict[str, dict] = {}
    for _, row in sim.slots.iterrows():
        slot = row["Slot"]
        s_id = sim.get_team_for_code(row["StrongSeed"])
        w_id = sim.get_team_for_code(row["WeakSeed"])
        prob = sim.probabilities.get(slot)
        if prob is None and s_id is not None and w_id is not None:
            prob = predictor.predict_matchup(s_id, w_id, sim.season, sim.gender)
        games[slot] = {
            "slot": slot,
            "strong_id": s_id,
            "weak_id": w_id,
            "strong_name": predictor.team_name(s_id) if s_id else "TBD",
            "weak_name": predictor.team_name(w_id) if w_id else "TBD",
            "strong_seed": _seed_num(sim, s_id),
            "weak_seed": _seed_num(sim, w_id),
            "winner_id": sim.results.get(slot),
            "probability": prob,
        }
    return games


# ═══════════════════════════════════════════════════════════════════════════
# SVG bracket renderer
# ═══════════════════════════════════════════════════════════════════════════
def render_bracket_svg(sim: BracketSimulator,
                       predictor: BracketPredictor) -> str:
    """Build a full SVG bracket and return the HTML string."""
    games = _collect_games(sim, predictor)

    # Detect regions
    r1_slots = sim.slots[sim.slots["Slot"].str.match(r"^R1[A-Z]")]
    regions = sorted(r1_slots["Slot"].str[2].unique())
    if len(regions) < 4:
        return "<p style='color:red;'>Need 4 regions for bracket.</p>"
    left_top, left_bot, right_top, right_bot = regions

    parts: list[str] = []

    # ── y-centres per region ──────────────────────────────────────────────
    def _ypos(y0: float):
        r1 = _game_centers(8, y0)
        r2 = _midpoints(r1)
        r3 = _midpoints(r2)
        r4 = _midpoints(r3)
        return r1, r2, r3, r4

    lt1, lt2, lt3, lt4 = _ypos(TOP_Y)
    lb1, lb2, lb3, lb4 = _ypos(BOT_Y)
    rt1, rt2, rt3, rt4 = _ypos(TOP_Y)
    rb1, rb2, rb3, rb4 = _ypos(BOT_Y)

    # ── round headers ─────────────────────────────────────────────────────
    hy = PAD - 6
    for i, lbl in enumerate(HEADERS_L):
        cx = LEFT_X[i] + BOX_W / 2
        parts.append(
            f'<text x="{cx}" y="{hy}" text-anchor="middle" font-size="7" '
            f'fill="#888" font-family="Arial">{lbl}</text>\n')
    parts.append(
        f'<text x="{CH_X + BOX_W / 2}" y="{hy}" text-anchor="middle" '
        f'font-size="7" fill="#c0392b" font-weight="bold" '
        f'font-family="Arial">CHAMPION</text>\n')
    for i, lbl in enumerate(HEADERS_R):
        cx = RIGHT_X[i] + BOX_W / 2
        parts.append(
            f'<text x="{cx}" y="{hy}" text-anchor="middle" font-size="7" '
            f'fill="#888" font-family="Arial">{lbl}</text>\n')

    # ── region labels ─────────────────────────────────────────────────────
    for reg, ystart, anchor, xref in [
        (left_top,  TOP_Y, "start", LEFT_X[0]),
        (left_bot,  BOT_Y, "start", LEFT_X[0]),
        (right_top, TOP_Y, "end",   RIGHT_X[4] + BOX_W),
        (right_bot, BOT_Y, "end",   RIGHT_X[4] + BOX_W),
    ]:
        parts.append(
            f'<text x="{xref}" y="{ystart - 2}" text-anchor="{anchor}" '
            f'font-size="8" fill="#555" font-weight="bold" '
            f'font-family="Arial">{REGION_NAMES.get(reg, reg)}</text>\n')

    # ── helper: draw one round column for a region ────────────────────────
    def _draw_round(region, order, centers, col_x,
                    next_centers, next_col_x, direction, rnd):
        out = []
        for i, sn in enumerate(order):
            slot = f"{rnd}{region}{sn}"
            if slot in games:
                out.append(_game_box(col_x, centers[i], games[slot]))
        # connectors to next round
        if next_centers is not None and next_col_x is not None:
            if direction == "right":
                ex = col_x + BOX_W
                for j in range(len(centers) // 2):
                    out.append(_connector(ex, centers[2 * j],
                                          centers[2 * j + 1],
                                          next_col_x, "right"))
            else:
                ex = col_x
                for j in range(len(centers) // 2):
                    out.append(_connector(ex, centers[2 * j],
                                          centers[2 * j + 1],
                                          next_col_x + BOX_W, "left"))
        return "".join(out)

    # ── left side ─────────────────────────────────────────────────────────
    for reg, y1, y2, y3, y4 in [
        (left_top, lt1, lt2, lt3, lt4),
        (left_bot, lb1, lb2, lb3, lb4),
    ]:
        parts.append(_draw_round(reg, R1_ORDER, y1, LEFT_X[0],
                                 y2, LEFT_X[1], "right", "R1"))
        parts.append(_draw_round(reg, R2_ORDER, y2, LEFT_X[1],
                                 y3, LEFT_X[2], "right", "R2"))
        parts.append(_draw_round(reg, R3_ORDER, y3, LEFT_X[2],
                                 y4, LEFT_X[3], "right", "R3"))
        parts.append(_draw_round(reg, R4_ORDER, y4, LEFT_X[3],
                                 None, None, "right", "R4"))

    # ── right side ────────────────────────────────────────────────────────
    for reg, y1, y2, y3, y4 in [
        (right_top, rt1, rt2, rt3, rt4),
        (right_bot, rb1, rb2, rb3, rb4),
    ]:
        parts.append(_draw_round(reg, R1_ORDER, y1, RIGHT_X[4],
                                 y2, RIGHT_X[3], "left", "R1"))
        parts.append(_draw_round(reg, R2_ORDER, y2, RIGHT_X[3],
                                 y3, RIGHT_X[2], "left", "R2"))
        parts.append(_draw_round(reg, R3_ORDER, y3, RIGHT_X[2],
                                 y4, RIGHT_X[1], "left", "R3"))
        parts.append(_draw_round(reg, R4_ORDER, y4, RIGHT_X[1],
                                 None, None, "left", "R4"))

    # ── E8 → FF connectors ───────────────────────────────────────────────
    lt_e8, lb_e8 = lt4[0], lb4[0]
    rt_e8, rb_e8 = rt4[0], rb4[0]
    ff_l_y = (lt_e8 + lb_e8) / 2
    ff_r_y = (rt_e8 + rb_e8) / 2

    parts.append(_connector(LEFT_X[3] + BOX_W, lt_e8, lb_e8,
                            LEFT_X[4], "right"))
    parts.append(_connector(RIGHT_X[1], rt_e8, rb_e8,
                            RIGHT_X[0] + BOX_W, "left"))

    # ── Final Four games ──────────────────────────────────────────────────
    ff_slots = sim.slots[sim.slots["Slot"].str.startswith("R5")].sort_values("Slot")
    ff_names = ff_slots["Slot"].tolist()
    if len(ff_names) >= 1 and ff_names[0] in games:
        parts.append(_game_box(LEFT_X[4], ff_l_y, games[ff_names[0]]))
    if len(ff_names) >= 2 and ff_names[1] in games:
        parts.append(_game_box(RIGHT_X[0], ff_r_y, games[ff_names[1]]))

    # ── FF → Championship connectors ─────────────────────────────────────
    ch_y = (ff_l_y + ff_r_y) / 2
    parts.append(
        f'<line x1="{LEFT_X[4] + BOX_W}" y1="{ff_l_y}" '
        f'x2="{CH_X}" y2="{ch_y}" stroke="#999" stroke-width="1"/>\n')
    parts.append(
        f'<line x1="{RIGHT_X[0]}" y1="{ff_r_y}" '
        f'x2="{CH_X + BOX_W}" y2="{ch_y}" stroke="#999" stroke-width="1"/>\n')

    # ── Championship game ─────────────────────────────────────────────────
    ch_slots = sim.slots[sim.slots["Slot"].str.startswith("R6")]
    if len(ch_slots) > 0:
        ch_slot = ch_slots.iloc[0]["Slot"]
        if ch_slot in games:
            parts.append(_game_box(CH_X, ch_y, games[ch_slot]))
            champ = sim.get_champion()
            if champ:
                cname = _esc(predictor.team_name(champ))
                parts.append(
                    f'<text x="{CH_X + BOX_W / 2}" y="{ch_y + GAME_H / 2 + 15}" '
                    f'text-anchor="middle" font-size="10" fill="#c0392b" '
                    f'font-weight="bold" font-family="Arial">'
                    f'\U0001F3C6 {cname}</text>\n')

    # ── assemble SVG ──────────────────────────────────────────────────────
    svg = (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;background:#fafafa;'
        f'border:1px solid #eee;border-radius:8px;">\n'
        f'{"".join(parts)}'
        f'</svg>'
    )
    return svg


# ═══════════════════════════════════════════════════════════════════════════
# Downstream dependency tracking
# ═══════════════════════════════════════════════════════════════════════════
def _clear_downstream(sim: BracketSimulator, slot: str) -> None:
    """Recursively clear results that depend on *slot*."""
    for _, row in sim.slots.iterrows():
        if row["StrongSeed"] == slot or row["WeakSeed"] == slot:
            child = row["Slot"]
            sim.results.pop(child, None)
            sim.probabilities.pop(child, None)
            key = f"pick_{child}"
            if key in st.session_state:
                del st.session_state[key]
            _clear_downstream(sim, child)


# ═══════════════════════════════════════════════════════════════════════════
# Pick interface
# ═══════════════════════════════════════════════════════════════════════════
def _render_game_pick(sim: BracketSimulator,
                      predictor: BracketPredictor,
                      gd: dict, slot: str) -> bool:
    """Render a pick widget for one game. Returns True if pick changed."""
    t1 = gd["strong_id"]
    t2 = gd["weak_id"]
    if t1 is None or t2 is None:
        st.caption(f"*{slot}: Awaiting prior results*")
        return False

    s1 = f"({gd['strong_seed']}) " if gd.get("strong_seed") else ""
    s2 = f"({gd['weak_seed']}) " if gd.get("weak_seed") else ""
    prob = gd.get("probability") or 0.5
    opt1 = f"{s1}{gd['strong_name']} \u2014 {prob:.0%}"
    opt2 = f"{s2}{gd['weak_name']} \u2014 {1 - prob:.0%}"

    current = sim.results.get(slot)
    idx = 1 if current == t2 else 0

    pick_key = f"pick_{slot}"
    chosen = st.radio(
        slot,
        [opt1, opt2],
        index=idx,
        key=pick_key,
        horizontal=True,
        label_visibility="collapsed",
    )

    new_winner = t1 if chosen == opt1 else t2
    changed = (sim.results.get(slot) != new_winner)
    if changed:
        sim.results[slot] = new_winner
        sim.probabilities[slot] = prob
        _clear_downstream(sim, slot)
    return changed


def render_pick_interface(sim: BracketSimulator,
                          predictor: BracketPredictor) -> bool:
    """Render the round-by-round pick interface. Returns True if anything changed."""
    games = _collect_games(sim, predictor)
    changed = False

    # Detect regions
    r1_slots = sim.slots[sim.slots["Slot"].str.match(r"^R1[A-Z]")]
    regions = sorted(r1_slots["Slot"].str[2].unique())

    # Play-in slots
    playin_slots = sorted([s for s in games if not s.startswith("R")])

    tab_names = ["Play-In", "R64", "R32", "Sweet 16", "Elite 8",
                 "Final Four", "Championship"]
    tabs = st.tabs(tab_names)

    # ── Play-In ───────────────────────────────────────────────────────────
    with tabs[0]:
        if not playin_slots:
            st.info("No play-in games for this tournament.")
        else:
            cols = st.columns(2)
            for i, slot in enumerate(playin_slots):
                with cols[i % 2]:
                    if _render_game_pick(sim, predictor, games[slot], slot):
                        changed = True

    # ── Rounds 1-4 (region-based) ─────────────────────────────────────────
    round_cfgs = [
        (1, "R1", 4),
        (2, "R2", 4),
        (3, "R3", 4),
        (4, "R4", 4),
    ]
    for round_num, prefix, ncols in round_cfgs:
        with tabs[round_num]:
            for region in regions:
                rname = REGION_NAMES.get(region, region)
                st.markdown(f"**{rname} Region**")
                rslots = sorted(
                    s for s in games
                    if s.startswith(prefix) and len(s) > 2 and s[2] == region
                )
                if not rslots:
                    st.caption("No games.")
                    continue
                cols = st.columns(min(ncols, max(len(rslots), 1)))
                for j, slot in enumerate(rslots):
                    with cols[j % len(cols)]:
                        if _render_game_pick(sim, predictor, games[slot], slot):
                            changed = True
                st.divider()

    # ── Final Four ────────────────────────────────────────────────────────
    with tabs[5]:
        ff_slots = sorted(s for s in games if s.startswith("R5"))
        if not ff_slots:
            st.caption("No Final Four games yet.")
        else:
            cols = st.columns(max(len(ff_slots), 1))
            for j, slot in enumerate(ff_slots):
                with cols[j % len(cols)]:
                    if _render_game_pick(sim, predictor, games[slot], slot):
                        changed = True

    # ── Championship ──────────────────────────────────────────────────────
    with tabs[6]:
        ch_slots = sorted(s for s in games if s.startswith("R6"))
        for slot in ch_slots:
            if _render_game_pick(sim, predictor, games[slot], slot):
                changed = True
        champ = sim.get_champion()
        if champ:
            st.success(f"\U0001F3C6 **Champion: {predictor.team_name(champ)}**")

    return changed


# ═══════════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.title("\U0001F3C0 March Madness Bracket Predictor")
    st.caption("Split M/W logistic regression \u00b7 trained on 2010\u20132025 tournament data")

    if not check_models_exist():
        st.error(
            "**Models not found!** Run:\n\n"
            "```bash\npython -m src.export_models\n```"
        )
        st.stop()

    predictor = load_predictor()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    gender = st.sidebar.radio("Tournament", ["Men's", "Women's"])
    gender_code = "M" if gender == "Men's" else "W"

    current_season = predictor.config.get("current_season", 2026)
    season = st.sidebar.number_input(
        "Season", min_value=2011, max_value=current_season,
        value=current_season,
        help="Current year for predictions, or a past year to test.",
    )

    # Verify seeds
    pfx = "M" if gender_code == "M" else "W"
    seeds_df = pd.read_csv(DATA_DIR / f"{pfx}NCAATourneySeeds.csv")
    avail = sorted(seeds_df["Season"].unique())
    if season not in avail:
        st.warning(
            f"No seeds for {season} {gender} tournament. "
            f"Available: {avail[-5:]}"
        )
        season = st.sidebar.selectbox(
            "Pick a past season:",
            sorted(avail, reverse=True)[:10],
        )

    sim_mode = st.sidebar.radio(
        "Mode",
        ["Deterministic (favorites)", "Probabilistic (random)"],
    )
    mode = "deterministic" if "Deterministic" in sim_mode else "probabilistic"

    # Initialise simulator
    sim_key = f"sim_{gender_code}_{season}"
    if sim_key not in st.session_state:
        st.session_state[sim_key] = BracketSimulator(
            predictor, season, gender_code, DATA_DIR
        )
    sim = st.session_state[sim_key]

    # Action buttons
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("\U0001F916 Auto-Fill", key="btn_auto"):
            sim.results.clear()
            sim.probabilities.clear()
            sim.simulate_full_bracket(mode=mode)
            for k in list(st.session_state.keys()):
                if k.startswith("pick_"):
                    del st.session_state[k]
            st.rerun()
    with c2:
        if st.button("\U0001F504 Reset", key="btn_reset"):
            st.session_state[sim_key] = BracketSimulator(
                predictor, season, gender_code, DATA_DIR
            )
            for k in list(st.session_state.keys()):
                if k.startswith("pick_"):
                    del st.session_state[k]
            st.rerun()
    with c3:
        if st.button("\U0001F3B2 Random", key="btn_rand"):
            sim.results.clear()
            sim.probabilities.clear()
            sim.simulate_full_bracket(mode="probabilistic")
            for k in list(st.session_state.keys()):
                if k.startswith("pick_"):
                    del st.session_state[k]
            st.rerun()

    # Sidebar: seeded teams
    with st.sidebar.expander("\U0001F4CB Seeded Teams", expanded=False):
        sd = sim.seeds.copy()
        sd["Team"] = sd["TeamID"].apply(predictor.team_name)
        sd = sd[["Seed", "Team", "TeamID"]].sort_values("Seed")
        st.dataframe(sd, hide_index=True, width="stretch")

    # Sidebar: model info
    with st.sidebar.expander("\U0001F4CA Model Info", expanded=False):
        st.write(f"**Features:** {', '.join(predictor.features)}")
        st.write(f"**Training:** {predictor.config['n_men_games']}M + "
                 f"{predictor.config['n_women_games']}W games")

    # ── Bracket SVG ───────────────────────────────────────────────────────
    st.markdown(f"### {season} NCAA {gender} Tournament Bracket")

    svg_html = render_bracket_svg(sim, predictor)
    st.markdown(
        f'<div style="overflow-x:auto;overflow-y:auto;max-height:700px;'
        f'padding:4px 0;">{svg_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Pick interface ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001F4DD Fill Your Bracket")
    something_changed = render_pick_interface(sim, predictor)
    if something_changed:
        st.rerun()

    # ── Bracket summary table ─────────────────────────────────────────────
    if sim.results:
        st.markdown("---")
        st.subheader("\U0001F4CA Results Summary")
        summary = sim.get_bracket_summary()
        filled = [s for s in summary if s["winner_id"] is not None]
        if filled:
            df = pd.DataFrame(filled)
            df = df[["round", "team1_name", "team2_name",
                      "winner_name", "probability"]]
            df.columns = ["Round", "Team 1", "Team 2", "Winner", "P(Team 1)"]
            df["P(Team 1)"] = df["P(Team 1)"].apply(
                lambda x: f"{x:.1%}" if x is not None else "\u2014")
            st.dataframe(df, hide_index=True, width="stretch")

    # ── Matchup Explorer ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("\U0001F50D Head-to-Head Matchup Explorer")

    teams_list = sorted(
        sim.seeds["TeamID"].apply(predictor.team_name).tolist())
    mc1, mc2 = st.columns(2)
    with mc1:
        t1_name = st.selectbox("Team 1", teams_list, key="explore_t1")
    with mc2:
        t2_name = st.selectbox(
            "Team 2",
            [t for t in teams_list if t != t1_name],
            key="explore_t2",
        )

    if t1_name and t2_name:
        t1_ids = sim.seeds[
            sim.seeds["TeamID"].apply(predictor.team_name) == t1_name
        ]["TeamID"].values
        t2_ids = sim.seeds[
            sim.seeds["TeamID"].apply(predictor.team_name) == t2_name
        ]["TeamID"].values
        if len(t1_ids) > 0 and len(t2_ids) > 0:
            prob = predictor.predict_matchup(
                int(t1_ids[0]), int(t2_ids[0]), season, gender_code)
            mc1, mc2, mc3 = st.columns([2, 1, 2])
            with mc1:
                st.metric(t1_name, f"{prob:.1%}")
            with mc2:
                st.markdown("### vs")
            with mc3:
                st.metric(t2_name, f"{1 - prob:.1%}")
            st.progress(prob)


if __name__ == "__main__":
    main()

"""🏆 Interactive Bracket Predictor.

Full bracket visualization with pick interface.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bracket import (
    BracketPredictor, BracketSimulator, SubmissionPredictor,
    ROUND_NAMES, REGION_NAMES, MODEL_REGISTRY,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

st.set_page_config(
    page_title="Bracket Predictor",
    page_icon="\U0001f3c6",
    layout="wide",
)

# ── Shared dark theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); color: #e6edf3; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #f0f6fc !important; }
p, span, label, .stCaption, [data-testid="stCaptionContainer"] { color: #8b949e !important; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricValue"] { color: #58a6ff !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SVG Layout Constants
# ═══════════════════════════════════════════════════════════════════════════
BOX_W = 95
BOX_H = 18
CONN_W = 10
GAME_H = BOX_H * 2 + 2
GAME_GAP = 6
REGION_GAP = 30
PAD = 20
COL_W = BOX_W + CONN_W

R1_ORDER = [1, 8, 5, 4, 6, 3, 7, 2]
R2_ORDER = [1, 4, 3, 2]
R3_ORDER = [1, 2]
R4_ORDER = [1]

REGION_H = 8 * (GAME_H + GAME_GAP) - GAME_GAP
SVG_H = PAD * 2 + REGION_H * 2 + REGION_GAP
LEFT_X = [PAD + i * COL_W for i in range(5)]
CH_X = PAD + 5 * COL_W
RIGHT_X = [CH_X + BOX_W + CONN_W + i * COL_W for i in range(5)]
SVG_W = RIGHT_X[4] + BOX_W + PAD
TOP_Y = PAD
BOT_Y = PAD + REGION_H + REGION_GAP

HEADERS_L = ["R64", "R32", "Sweet 16", "Elite 8", "Final Four"]
HEADERS_R = ["Final Four", "Elite 8", "Sweet 16", "R32", "R64"]


@st.cache_resource
def load_predictor() -> BracketPredictor:
    return BracketPredictor(MODELS_DIR)


def check_models_exist() -> bool:
    required = [
        "men_model.pkl", "women_model.pkl", "men_scaler.pkl",
        "women_scaler.pkl", "team_features.pkl", "teams.pkl", "config.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


# ═══════════════════════════════════════════════════════════════════════════
# SVG helpers
# ═══════════════════════════════════════════════════════════════════════════
def _game_centers(n: int, y_start: float) -> list[float]:
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
    return name if len(name) <= maxlen else name[:maxlen - 1] + "\u2026"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _team_box(x: float, y: float, name: str, seed: Optional[int],
              is_winner: bool = False, score: Optional[int] = None,
              is_actual: bool = False) -> str:
    """Render one team box in SVG."""
    if name in ("TBD", None, ""):
        fill, stroke, tfill = "#21262d", "#30363d", "#484f58"
        weight, display = "normal", "TBD"
    elif is_winner and is_actual:
        fill, stroke, tfill = "#2d2000", "#ffc107", "#ffd866"
        weight = "bold"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    elif is_winner:
        fill, stroke, tfill = "#0d2818", "#28a745", "#7ee787"
        weight = "bold"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    elif is_actual and not is_winner:
        fill, stroke, tfill = "#2d0a0a", "#dc3545", "#f85149"
        weight = "normal"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    else:
        fill, stroke, tfill = "#161b22", "#30363d", "#c9d1d9"
        weight = "normal"
        display = f"{seed} {_trunc(name)}" if seed else _trunc(name)
    display = _esc(display)
    svg = (
        f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>\n'
        f'<text x="{x + 3}" y="{y + BOX_H - 5}" font-size="9" '
        f'fill="{tfill}" font-weight="{weight}" '
        f'font-family="Arial,sans-serif">{display}</text>\n'
    )
    if score is not None:
        score_color = "#ffd866" if is_actual else "#8b949e"
        svg += (
            f'<text x="{x + BOX_W - 3}" y="{y + BOX_H - 5}" font-size="8" '
            f'fill="{score_color}" font-weight="{weight}" text-anchor="end" '
            f'font-family="Arial,sans-serif">{score}</text>\n'
        )
    return svg


def _game_box(x: float, cy: float, gd: dict) -> str:
    top_y = cy - GAME_H / 2
    bot_y = top_y + BOX_H + 2
    winner = gd.get("winner_id")
    is_actual = gd.get("is_actual", False)
    svg = _team_box(x, top_y, gd["strong_name"], gd["strong_seed"],
                    is_winner=(winner is not None and winner == gd["strong_id"]),
                    score=gd.get("strong_score"), is_actual=is_actual)
    svg += _team_box(x, bot_y, gd["weak_name"], gd["weak_seed"],
                     is_winner=(winner is not None and winner == gd["weak_id"]),
                     score=gd.get("weak_score"), is_actual=is_actual)
    return svg


def _connector(x_edge: float, y1: float, y2: float,
               x_next: float, direction: str = "right") -> str:
    ym = (y1 + y2) / 2
    mx = x_edge + CONN_W / 2 if direction == "right" else x_edge - CONN_W / 2
    return (
        f'<line x1="{x_edge}" y1="{y1}" x2="{mx}" y2="{y1}" '
        f'stroke="#30363d" stroke-width="1"/>\n'
        f'<line x1="{x_edge}" y1="{y2}" x2="{mx}" y2="{y2}" '
        f'stroke="#30363d" stroke-width="1"/>\n'
        f'<line x1="{mx}" y1="{y1}" x2="{mx}" y2="{y2}" '
        f'stroke="#30363d" stroke-width="1"/>\n'
        f'<line x1="{mx}" y1="{ym}" x2="{x_next}" y2="{ym}" '
        f'stroke="#30363d" stroke-width="1"/>\n'
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
        strong_score, weak_score = None, None
        if prob is not None and s_id is not None and w_id is not None:
            try:
                strong_score, weak_score = predictor.predict_score(
                    s_id, w_id, sim.season, prob)
            except (AttributeError, Exception):
                pass

        # Overlay actual scores
        actual = sim.actual_results.get(slot)
        if actual:
            score_str = actual.get("score", "")
            if score_str and "-" in score_str:
                parts = score_str.split("-")
                w_score_actual = int(parts[0])
                l_score_actual = int(parts[1])
                winner = actual.get("winner_id")
                if winner == s_id:
                    strong_score, weak_score = w_score_actual, l_score_actual
                else:
                    strong_score, weak_score = l_score_actual, w_score_actual

        is_actual = slot in sim.actual_results

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
            "strong_score": strong_score,
            "weak_score": weak_score,
            "is_actual": is_actual,
        }
    return games


# ═══════════════════════════════════════════════════════════════════════════
# SVG bracket renderer
# ═══════════════════════════════════════════════════════════════════════════
def render_bracket_svg(sim: BracketSimulator,
                       predictor: BracketPredictor) -> str:
    games = _collect_games(sim, predictor)

    r1_slots = sim.slots[sim.slots["Slot"].str.match(r"^R1[A-Z]")]
    regions = sorted(r1_slots["Slot"].str[2].unique())
    if len(regions) < 4:
        return "<p style='color:red;'>Need 4 regions for bracket.</p>"

    region_layout = {"W": 0, "X": 1, "Z": 2, "Y": 3}
    ordered = sorted(regions, key=lambda r: region_layout.get(r, ord(r)))
    left_top, left_bot, right_top, right_bot = ordered

    parts: list[str] = []

    def _ypos(y0):
        r1 = _game_centers(8, y0)
        r2 = _midpoints(r1)
        r3 = _midpoints(r2)
        r4 = _midpoints(r3)
        return r1, r2, r3, r4

    lt1, lt2, lt3, lt4 = _ypos(TOP_Y)
    lb1, lb2, lb3, lb4 = _ypos(BOT_Y)
    rt1, rt2, rt3, rt4 = _ypos(TOP_Y)
    rb1, rb2, rb3, rb4 = _ypos(BOT_Y)

    # Headers
    hy = PAD - 6
    for i, lbl in enumerate(HEADERS_L):
        cx = LEFT_X[i] + BOX_W / 2
        parts.append(
            f'<text x="{cx}" y="{hy}" text-anchor="middle" font-size="7" '
            f'fill="#484f58" font-family="Arial">{lbl}</text>\n')
    parts.append(
        f'<text x="{CH_X + BOX_W / 2}" y="{hy}" text-anchor="middle" '
        f'font-size="7" fill="#f85149" font-weight="bold" '
        f'font-family="Arial">CHAMPION</text>\n')
    for i, lbl in enumerate(HEADERS_R):
        cx = RIGHT_X[i] + BOX_W / 2
        parts.append(
            f'<text x="{cx}" y="{hy}" text-anchor="middle" font-size="7" '
            f'fill="#484f58" font-family="Arial">{lbl}</text>\n')

    # Region labels
    for reg, ystart, anchor, xref in [
        (left_top,  TOP_Y, "start", LEFT_X[0]),
        (left_bot,  BOT_Y, "start", LEFT_X[0]),
        (right_top, TOP_Y, "end",   RIGHT_X[4] + BOX_W),
        (right_bot, BOT_Y, "end",   RIGHT_X[4] + BOX_W),
    ]:
        parts.append(
            f'<text x="{xref}" y="{ystart - 2}" text-anchor="{anchor}" '
            f'font-size="8" fill="#8b949e" font-weight="bold" '
            f'font-family="Arial">{REGION_NAMES.get(reg, reg)}</text>\n')

    def _draw_round(region, order, centers, col_x,
                    next_centers, next_col_x, direction, rnd):
        out = []
        for i, sn in enumerate(order):
            slot = f"{rnd}{region}{sn}"
            if slot in games:
                out.append(_game_box(col_x, centers[i], games[slot]))
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

    # Left side
    for reg, y1, y2, y3, y4 in [
        (left_top, lt1, lt2, lt3, lt4),
        (left_bot, lb1, lb2, lb3, lb4),
    ]:
        parts.append(_draw_round(reg, R1_ORDER, y1, LEFT_X[0], y2, LEFT_X[1], "right", "R1"))
        parts.append(_draw_round(reg, R2_ORDER, y2, LEFT_X[1], y3, LEFT_X[2], "right", "R2"))
        parts.append(_draw_round(reg, R3_ORDER, y3, LEFT_X[2], y4, LEFT_X[3], "right", "R3"))
        parts.append(_draw_round(reg, R4_ORDER, y4, LEFT_X[3], None, None, "right", "R4"))

    # Right side
    for reg, y1, y2, y3, y4 in [
        (right_top, rt1, rt2, rt3, rt4),
        (right_bot, rb1, rb2, rb3, rb4),
    ]:
        parts.append(_draw_round(reg, R1_ORDER, y1, RIGHT_X[4], y2, RIGHT_X[3], "left", "R1"))
        parts.append(_draw_round(reg, R2_ORDER, y2, RIGHT_X[3], y3, RIGHT_X[2], "left", "R2"))
        parts.append(_draw_round(reg, R3_ORDER, y3, RIGHT_X[2], y4, RIGHT_X[1], "left", "R3"))
        parts.append(_draw_round(reg, R4_ORDER, y4, RIGHT_X[1], None, None, "left", "R4"))

    # E8 → FF connectors
    lt_e8, lb_e8 = lt4[0], lb4[0]
    rt_e8, rb_e8 = rt4[0], rb4[0]
    ff_l_y = (lt_e8 + lb_e8) / 2
    ff_r_y = (rt_e8 + rb_e8) / 2

    parts.append(_connector(LEFT_X[3] + BOX_W, lt_e8, lb_e8, LEFT_X[4], "right"))
    parts.append(_connector(RIGHT_X[1], rt_e8, rb_e8, RIGHT_X[0] + BOX_W, "left"))

    # Final Four
    ff_slots = sim.slots[sim.slots["Slot"].str.startswith("R5")].sort_values("Slot")
    ff_names = ff_slots["Slot"].tolist()
    if len(ff_names) >= 1 and ff_names[0] in games:
        parts.append(_game_box(LEFT_X[4], ff_l_y, games[ff_names[0]]))
    if len(ff_names) >= 2 and ff_names[1] in games:
        parts.append(_game_box(RIGHT_X[0], ff_r_y, games[ff_names[1]]))

    # Championship
    ch_y = (ff_l_y + ff_r_y) / 2
    parts.append(
        f'<line x1="{LEFT_X[4] + BOX_W}" y1="{ff_l_y}" '
        f'x2="{CH_X}" y2="{ch_y}" stroke="#30363d" stroke-width="1"/>\n')
    parts.append(
        f'<line x1="{RIGHT_X[0]}" y1="{ff_r_y}" '
        f'x2="{CH_X + BOX_W}" y2="{ch_y}" stroke="#30363d" stroke-width="1"/>\n')

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
                    f'text-anchor="middle" font-size="10" fill="#f85149" '
                    f'font-weight="bold" font-family="Arial">'
                    f'🏆 {cname}</text>\n')

    svg = (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;background:#0d1117;'
        f'border:1px solid #30363d;border-radius:12px;">\n'
        f'{"".join(parts)}'
        f'</svg>'
    )
    return svg


# ═══════════════════════════════════════════════════════════════════════════
# Pick interface
# ═══════════════════════════════════════════════════════════════════════════
def _clear_downstream(sim: BracketSimulator, slot: str) -> None:
    for _, row in sim.slots.iterrows():
        if row["StrongSeed"] == slot or row["WeakSeed"] == slot:
            child = row["Slot"]
            sim.results.pop(child, None)
            sim.probabilities.pop(child, None)
            key = f"pick_{child}"
            if key in st.session_state:
                del st.session_state[key]
            _clear_downstream(sim, child)


def _slot_display(slot: str) -> str:
    """Convert cryptic slot codes to readable labels.

    Examples: R1W1 → 'East #1', R2X3 → 'South R32 #3', R5WX → 'FF #1'
    """
    if not slot.startswith("R"):
        # Play-in slot like "Y16", "Z11"
        return f"Play-In ({slot})"
    rnd = slot[:2]
    round_labels = {
        "R1": "R64", "R2": "R32", "R3": "Sweet 16",
        "R4": "Elite 8", "R5": "Final Four", "R6": "Championship",
    }
    rlabel = round_labels.get(rnd, rnd)
    if rnd in ("R5", "R6"):
        return rlabel
    region_code = slot[2] if len(slot) > 2 else ""
    game_num = slot[3:] if len(slot) > 3 else ""
    region_name = REGION_NAMES.get(region_code, region_code)
    return f"{region_name} {rlabel} #{game_num}" if game_num else f"{region_name} {rlabel}"


def _render_game_pick(sim: BracketSimulator,
                      predictor: BracketPredictor,
                      gd: dict, slot: str) -> bool:
    """Render a pick widget for one game. Returns True if pick changed."""
    t1 = gd["strong_id"]
    t2 = gd["weak_id"]
    readable = _slot_display(slot)
    if t1 is None or t2 is None:
        st.caption(f"*{readable}: Awaiting prior results*")
        return False

    # Lock actual results — display only, no interaction
    if gd.get("is_actual", False):
        winner = gd.get("winner_id")
        w_name = gd["strong_name"] if winner == t1 else gd["weak_name"]
        l_name = gd["weak_name"] if winner == t1 else gd["strong_name"]
        w_score = gd.get("strong_score") if winner == t1 else gd.get("weak_score")
        l_score = gd.get("weak_score") if winner == t1 else gd.get("strong_score")
        score_str = f" {w_score}-{l_score}" if w_score and l_score else ""
        st.success(f"✅ **{w_name}** def. {l_name}{score_str}", icon="🏀")
        return False

    s1 = f"({gd['strong_seed']}) " if gd.get("strong_seed") else ""
    s2 = f"({gd['weak_seed']}) " if gd.get("weak_seed") else ""
    prob = gd.get("probability") or 0.5
    label1 = f"{s1}{gd['strong_name']} — {prob:.0%}"
    label2 = f"{s2}{gd['weak_name']} — {1 - prob:.0%}"

    current = sim.results.get(slot)
    idx = 1 if current == t2 else 0

    labels = {t1: label1, t2: label2}
    pick_key = f"pick_{slot}"
    chosen = st.radio(
        readable, [t1, t2], index=idx,
        format_func=lambda tid: labels.get(tid, str(tid)),
        key=pick_key, horizontal=True, label_visibility="collapsed",
    )

    new_winner = chosen
    changed = (sim.results.get(slot) != new_winner)
    if changed:
        sim.results[slot] = new_winner
        sim.probabilities[slot] = prob
        _clear_downstream(sim, slot)
    return changed


def render_pick_interface(sim: BracketSimulator,
                          predictor: BracketPredictor) -> bool:
    games = _collect_games(sim, predictor)
    changed = False

    r1_slots = sim.slots[sim.slots["Slot"].str.match(r"^R1[A-Z]")]
    region_layout = {"W": 0, "X": 1, "Z": 2, "Y": 3}
    regions = sorted(r1_slots["Slot"].str[2].unique(),
                     key=lambda r: region_layout.get(r, ord(r)))

    playin_slots = sorted([s for s in games if not s.startswith("R")])
    tab_names = ["Play-In", "R64", "R32", "Sweet 16", "Elite 8",
                 "Final Four", "Championship"]
    tabs = st.tabs(tab_names)

    playin_changed = False
    with tabs[0]:
        if not playin_slots:
            st.info("No play-in games for this tournament.")
        else:
            cols = st.columns(2)
            for i, slot in enumerate(playin_slots):
                with cols[i % 2]:
                    if _render_game_pick(sim, predictor, games[slot], slot):
                        changed = True
                        playin_changed = True

    if playin_changed:
        games = _collect_games(sim, predictor)

    round_cfgs = [(1, "R1", 4), (2, "R2", 4), (3, "R3", 4), (4, "R4", 4)]
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

    with tabs[6]:
        ch_slots = sorted(s for s in games if s.startswith("R6"))
        for slot in ch_slots:
            if _render_game_pick(sim, predictor, games[slot], slot):
                changed = True
        champ = sim.get_champion()
        if champ:
            st.success(f"🏆 **Champion: {predictor.team_name(champ)}**")

    return changed


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.title("🏆 Bracket Predictor")
    st.markdown(
        "Build your bracket using our model's win probabilities. "
        "Use **Auto-Fill** for the model's picks, or choose your own winners below."
    )

    if not check_models_exist():
        st.error("**Models not found!** Run `python -m src.export_models` first.")
        st.stop()

    predictor = load_predictor()

    # Sidebar
    st.sidebar.header("Settings")
    gender = st.sidebar.radio("Tournament", ["Men's", "Women's"], key="brk_gender")
    gender_code = "M" if gender == "Men's" else "W"

    current_season = predictor.config.get("current_season", 2026)
    season = st.sidebar.number_input(
        "Season", min_value=2011, max_value=current_season,
        value=current_season,
    )

    pfx = "M" if gender_code == "M" else "W"
    seeds_df = pd.read_csv(DATA_DIR / f"{pfx}NCAATourneySeeds.csv")
    avail = sorted(seeds_df["Season"].unique())
    if season not in avail:
        st.warning(f"No seeds for {season}. Available: {avail[-5:]}")
        season = st.sidebar.selectbox("Pick a past season:",
                                       sorted(avail, reverse=True)[:10])

    # ── Model selector ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001f9e0 Model")
    selected_idx = st.sidebar.selectbox(
        "Choose a model",
        range(len(MODEL_REGISTRY)),
        format_func=lambda i: MODEL_REGISTRY[i]["name"],
        key="model_select",
    )
    sel = MODEL_REGISTRY[selected_idx]

    # Description card
    st.sidebar.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;'
        f'border-radius:8px;padding:10px 12px;margin-bottom:8px;">'
        f'<span style="color:#c9d1d9;font-size:13px;">{sel["description"]}</span>'
        f'<table style="margin-top:6px;font-size:12px;color:#8b949e;'
        f'border-collapse:collapse;width:100%;">'
        f'<tr><td style="padding:2px 6px 2px 0;color:#58a6ff;">Model</td>'
        f'<td style="padding:2px 0;">{sel["model"]}</td></tr>'
        f'<tr><td style="padding:2px 6px 2px 0;color:#58a6ff;">Split</td>'
        f'<td style="padding:2px 0;">{sel["split"]}</td></tr>'
        f'<tr><td style="padding:2px 6px 2px 0;color:#58a6ff;">Features</td>'
        f'<td style="padding:2px 0;">{sel["features"]}</td></tr>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # Build the predictor for the selected model
    sub_path = OUTPUT_DIR / sel["file"]
    if sub_path.exists() and season == current_season:
        active_predictor = SubmissionPredictor(predictor, sub_path, season)
    else:
        # Fall back to live model for historical seasons or missing CSV
        active_predictor = predictor

    sim_key = f"sim_{gender_code}_{season}_{sel['id']}"
    if sim_key not in st.session_state:
        st.session_state[sim_key] = BracketSimulator(
            active_predictor, season, gender_code, DATA_DIR)
    sim = st.session_state[sim_key]

    # Action buttons
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("\U0001f916 Auto-Fill", key="btn_auto"):
            sim.results.clear()
            sim.probabilities.clear()
            sim.simulate_full_bracket(mode="deterministic")
            sim._load_actual_results()
            for k in list(st.session_state.keys()):
                if k.startswith("pick_"):
                    del st.session_state[k]
            st.rerun()
    with c2:
        if st.button("\U0001f504 Reset", key="btn_reset"):
            st.session_state[sim_key] = BracketSimulator(
                active_predictor, season, gender_code, DATA_DIR)
            for k in list(st.session_state.keys()):
                if k.startswith("pick_"):
                    del st.session_state[k]
            st.rerun()

    with st.sidebar.expander("📋 Seeds", expanded=False):
        sd = sim.seeds.copy()
        sd["Team"] = sd["TeamID"].apply(active_predictor.team_name)
        sd = sd[["Seed", "Team", "TeamID"]].sort_values("Seed")
        st.dataframe(sd, hide_index=True, width='stretch')

    # SVG legend
    st.markdown(
        '<div style="font-size:12px;margin-bottom:8px;color:#c9d1d9;">'
        '<b>Legend:</b> '
        '<span style="background:#2d2000;border:1px solid #ffc107;color:#ffd866;padding:2px 6px;border-radius:3px;">\U0001f3c0 Actual Result</span> '
        '<span style="background:#0d2818;border:1px solid #28a745;color:#7ee787;padding:2px 6px;border-radius:3px;">Predicted Winner</span> '
        '<span style="background:#2d0a0a;border:1px solid #dc3545;color:#f85149;padding:2px 6px;border-radius:3px;">Actual Loser</span> '
        '<span style="color:#8b949e;margin-left:8px;">Scores in brackets are model-predicted scores</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Bracket SVG
    st.markdown(f"### {season} NCAA {gender} Tournament Bracket")
    st.caption(f"Model: {sel['name']}")
    bracket_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📝 Fill Your Bracket")
    st.caption(
        "Pick winners for each game below. The percentage next to each team is our model's "
        "predicted win probability. Gold-highlighted games are locked actual results."
    )
    something_changed = render_pick_interface(sim, active_predictor)

    with bracket_placeholder.container():
        svg_html = render_bracket_svg(sim, active_predictor)
        st.markdown(
            f'<div style="overflow-x:auto;overflow-y:auto;max-height:700px;'
            f'padding:4px 0;">{svg_html}</div>',
            unsafe_allow_html=True,
        )

    if something_changed:
        st.rerun()

    # Summary table
    if sim.results:
        st.markdown("---")
        st.subheader("📊 Results Summary")
        summary = sim.get_bracket_summary()
        filled = [s for s in summary if s["winner_id"] is not None]
        if filled:
            df = pd.DataFrame(filled)
            df = df[["round", "team1_name", "team2_name",
                      "winner_name", "probability"]]
            df.columns = ["Round", "Team 1", "Team 2", "Winner", "P(Team 1)"]
            df["P(Team 1)"] = df["P(Team 1)"].apply(
                lambda x: f"{x:.1%}" if x is not None else "—")
            st.dataframe(df, hide_index=True, width='stretch')

    # Matchup Explorer
    st.markdown("---")
    st.subheader("🔍 Head-to-Head Matchup Explorer")
    teams_list = sorted(sim.seeds["TeamID"].apply(active_predictor.team_name).tolist())
    mc1, mc2 = st.columns(2)
    with mc1:
        t1_name = st.selectbox("Team 1", teams_list, key="explore_t1")
    with mc2:
        t2_name = st.selectbox("Team 2",
                                [t for t in teams_list if t != t1_name],
                                key="explore_t2")
    if t1_name and t2_name:
        t1_ids = sim.seeds[sim.seeds["TeamID"].apply(active_predictor.team_name) == t1_name]["TeamID"].values
        t2_ids = sim.seeds[sim.seeds["TeamID"].apply(active_predictor.team_name) == t2_name]["TeamID"].values
        if len(t1_ids) > 0 and len(t2_ids) > 0:
            prob = active_predictor.predict_matchup(int(t1_ids[0]), int(t2_ids[0]), season, gender_code)
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

"""FastAPI backend for March Madness Bracket Predictor.

Serves the interactive bracket app and provides REST API endpoints
for predictions and bracket simulation.

Usage:
    python server.py                    # starts on http://localhost:8000
    uvicorn server:app --reload         # dev mode with hot-reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bracket import BracketPredictor, BracketSimulator, ROUND_NAMES, REGION_NAMES
from src.export_models import ensure_models

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── Global state ──────────────────────────────────────────────────────────
predictor: BracketPredictor | None = None


# ── Lifespan: auto-train on startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("\n🏀 Starting March Madness Bracket Predictor...")
    ensure_models(MODELS_DIR, DATA_DIR, max_age_hours=168)  # 7 days
    predictor = BracketPredictor(MODELS_DIR)
    print("✅ Models loaded. Server ready.\n")
    yield
    print("👋 Shutting down.")


app = FastAPI(
    title="March Madness Bracket Predictor",
    version="2.0",
    lifespan=lifespan,
)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic request/response models
# ═══════════════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    team1_id: int
    team2_id: int
    season: int
    gender: str = "M"


class SimulateRequest(BaseModel):
    gender: str = "M"
    season: int = 2025
    mode: str = "deterministic"
    overrides: dict[str, int] = {}  # slot → team_id


class TeamInfo(BaseModel):
    team_id: int
    team_name: str
    seed: str
    seed_num: int


class GameInfo(BaseModel):
    slot: str
    round_name: str
    strong_seed: str
    weak_seed: str
    strong_id: int | None
    weak_id: int | None
    strong_name: str
    weak_name: str
    strong_seed_num: int | None
    weak_seed_num: int | None
    winner_id: int | None
    winner_name: str | None
    probability: float | None  # P(strong wins)


class BracketResponse(BaseModel):
    season: int
    gender: str
    gender_label: str
    regions: dict[str, str]
    teams: list[TeamInfo]
    games: list[GameInfo]
    champion_id: int | None
    champion_name: str | None


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _get_available_seasons(gender: str) -> list[int]:
    prefix = "M" if gender == "M" else "W"
    seeds_path = DATA_DIR / f"{prefix}NCAATourneySeeds.csv"
    if not seeds_path.exists():
        return []
    df = pd.read_csv(seeds_path)
    return sorted(df["Season"].unique().tolist())


def _seed_num(seed_str: str) -> int:
    try:
        return int(seed_str[1:3])
    except (ValueError, IndexError):
        return 99


def _build_bracket_response(sim: BracketSimulator) -> BracketResponse:
    """Build a full BracketResponse from a simulated bracket."""
    # Teams
    teams = []
    for _, row in sim.seeds.iterrows():
        teams.append(TeamInfo(
            team_id=int(row["TeamID"]),
            team_name=predictor.team_name(int(row["TeamID"])),
            seed=row["Seed"],
            seed_num=int(row["seed_num"]),
        ))

    # Games
    games = []
    for _, row in sim.slots.sort_values("Slot").iterrows():
        slot = row["Slot"]
        s_code = row["StrongSeed"]
        w_code = row["WeakSeed"]
        s_id = sim.get_team_for_code(s_code)
        w_id = sim.get_team_for_code(w_code)

        prob = sim.probabilities.get(slot)
        if prob is None and s_id is not None and w_id is not None:
            prob = predictor.predict_matchup(s_id, w_id, sim.season, sim.gender)

        winner = sim.results.get(slot)

        # Round name
        if slot.startswith("R"):
            rkey = slot[:2]
            rname = ROUND_NAMES.get(rkey, rkey)
        else:
            rname = "Play-In"

        games.append(GameInfo(
            slot=slot,
            round_name=rname,
            strong_seed=s_code,
            weak_seed=w_code,
            strong_id=int(s_id) if s_id is not None else None,
            weak_id=int(w_id) if w_id is not None else None,
            strong_name=predictor.team_name(s_id) if s_id else "TBD",
            weak_name=predictor.team_name(w_id) if w_id else "TBD",
            strong_seed_num=_seed_num(s_code) if len(s_code) <= 4 else None,
            weak_seed_num=_seed_num(w_code) if len(w_code) <= 4 else None,
            winner_id=int(winner) if winner is not None else None,
            winner_name=predictor.team_name(winner) if winner else None,
            probability=round(prob, 4) if prob is not None else None,
        ))

    champion = sim.get_champion()
    return BracketResponse(
        season=sim.season,
        gender=sim.gender,
        gender_label="Men's" if sim.gender == "M" else "Women's",
        regions=dict(REGION_NAMES),
        teams=sorted(teams, key=lambda t: t.seed),
        games=games,
        champion_id=int(champion) if champion else None,
        champion_name=predictor.team_name(champion) if champion else None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def index():
    """Serve the main bracket page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/seasons")
async def get_seasons():
    """List available seasons per gender."""
    return {
        "M": _get_available_seasons("M"),
        "W": _get_available_seasons("W"),
    }


@app.get("/api/bracket/{gender}/{season}")
async def get_bracket(gender: str, season: int, mode: str = "deterministic"):
    """Get a full bracket (auto-filled by model)."""
    gender = gender.upper()
    if gender not in ("M", "W"):
        raise HTTPException(400, "gender must be M or W")

    avail = _get_available_seasons(gender)
    if season not in avail:
        raise HTTPException(404, f"No seeds for {season}. Available: {avail[-5:]}")

    sim = BracketSimulator(predictor, season, gender, DATA_DIR)
    sim.simulate_full_bracket(mode=mode)
    return _build_bracket_response(sim)


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Predict P(team1 wins) for a single matchup."""
    prob = predictor.predict_matchup(
        req.team1_id, req.team2_id, req.season, req.gender.upper()
    )
    return {
        "team1_id": req.team1_id,
        "team2_id": req.team2_id,
        "team1_name": predictor.team_name(req.team1_id),
        "team2_name": predictor.team_name(req.team2_id),
        "probability": round(prob, 4),
    }


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """Simulate a bracket with optional user overrides."""
    gender = req.gender.upper()
    avail = _get_available_seasons(gender)
    if req.season not in avail:
        raise HTTPException(404, f"No seeds for {req.season}")

    sim = BracketSimulator(predictor, req.season, gender, DATA_DIR)

    # Apply overrides first
    for slot, team_id in req.overrides.items():
        sim.set_override(slot, team_id)

    sim.simulate_full_bracket(mode=req.mode)
    return _build_bracket_response(sim)


@app.post("/api/retrain")
async def retrain():
    """Force re-download data and retrain models."""
    global predictor
    from src.export_models import train_and_export
    from src.data_loader import download_latest_data

    try:
        download_latest_data(DATA_DIR)
    except Exception as e:
        print(f"Download warning: {e}")

    config = train_and_export(MODELS_DIR)
    predictor = BracketPredictor(MODELS_DIR)
    return {"status": "ok", "config": config}


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

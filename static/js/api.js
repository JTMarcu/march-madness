/**
 * API client for March Madness Bracket Predictor backend.
 * All methods return Promises resolving to parsed JSON.
 */
const API = {
  /** @returns {{ M: number[], W: number[] }} */
  async getSeasons() {
    const res = await fetch('/api/seasons');
    if (!res.ok) throw new Error(`GET /api/seasons → ${res.status}`);
    return res.json();
  },

  /**
   * Fetch a fully-simulated bracket.
   * @param {'M'|'W'} gender
   * @param {number} season
   * @param {'deterministic'|'probabilistic'} mode
   * @returns {BracketResponse}
   */
  async getBracket(gender, season, mode = 'deterministic') {
    const res = await fetch(`/api/bracket/${gender}/${season}?mode=${mode}`);
    if (!res.ok) throw new Error(`GET /api/bracket → ${res.status}`);
    return res.json();
  },

  /**
   * Predict a single matchup.
   * @returns {{ team1_id, team2_id, team1_name, team2_name, probability }}
   */
  async predict(team1Id, team2Id, season, gender = 'M') {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        team1_id: team1Id,
        team2_id: team2Id,
        season,
        gender,
      }),
    });
    if (!res.ok) throw new Error(`POST /api/predict → ${res.status}`);
    return res.json();
  },

  /**
   * Simulate a bracket with user overrides.
   * @param {'M'|'W'} gender
   * @param {number} season
   * @param {'deterministic'|'probabilistic'} mode
   * @param {Object<string, number>} overrides  slot → teamId
   * @returns {BracketResponse}
   */
  async simulate(gender, season, mode, overrides = {}) {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gender, season, mode, overrides }),
    });
    if (!res.ok) throw new Error(`POST /api/simulate → ${res.status}`);
    return res.json();
  },

  /** Force re-download and retrain models. */
  async retrain() {
    const res = await fetch('/api/retrain', { method: 'POST' });
    if (!res.ok) throw new Error(`POST /api/retrain → ${res.status}`);
    return res.json();
  },
};

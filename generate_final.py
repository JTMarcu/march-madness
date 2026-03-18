"""Generate final optimized submissions (sub7 and sub8) and compare brackets."""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src import data_loader, features

# Load all data
rs = data_loader.load_regular_season()
tr = data_loader.load_tourney_results()
seeds = data_loader.load_tourney_seeds()
compact = data_loader.load_compact_results()
rs = rs[rs['Season'] >= 2010]
tr = tr[tr['Season'] >= 2010]

# Build features
gd = features.prepare_game_data(rs)
stats = features.compute_season_stats(gd)
wp = features.compute_win_pct(gd)
eff = features.compute_efficiency(gd)
mom = features.compute_last14_momentum(gd)
qual = features.compute_team_quality(gd, seeds)
elo = features.compute_elo_ratings(compact)
shooting = features.compute_shooting_pcts(stats)
tf = features.build_team_features(stats, wp, eff, mom, seeds,
                                   quality=qual, elo=elo, shooting=shooting)

# Build training matchups
tgd = features.prepare_game_data(tr)
tmatch = tgd[tgd['T1_TeamID'] < tgd['T2_TeamID']][
    ['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff']].copy()
tmatch['T1_Win'] = (tmatch['PointDiff'] > 0).astype(int)
enriched = features.create_matchup_df(tmatch, tf)
enriched, _ = features.compute_difference_features(enriched)
enriched = enriched[~enriched['Season'].isin([2020])].copy()

FEATS7 = ['Diff_seed', 'Diff_PointDiff', 'Diff_OffEff', 'Diff_WinPct',
           'Diff_Elo', 'Diff_FGPct', 'Diff_FTPct']

# Load sample submission & parse
sample = data_loader.load_sample_submission(stage=2)
sample[['Season', 'T1_TeamID', 'T2_TeamID']] = (
    sample['ID'].str.split('_', expand=True).astype(int))

# Build features for ALL 2026 matchups
all_matchups = features.create_matchup_df(sample, tf)
all_matchups, _ = features.compute_difference_features(all_matchups)

train_data = enriched.copy()

xgb_deeper = dict(max_depth=3, n_estimators=150, learning_rate=0.05,
    min_child_weight=30, gamma=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1, reg_lambda=3)
lgb_params = dict(num_leaves=8, max_depth=2, n_estimators=200,
    learning_rate=0.05, min_child_samples=30, reg_alpha=1, reg_lambda=3,
    subsample=0.8, colsample_bytree=0.8, verbose=-1)


def train_and_predict(train_df, test_matchups, feat_cols, mode='lr',
                      cm=0.25, cw=0.15):
    """Train split M/W models and predict."""
    preds = np.zeros(len(test_matchups))
    for is_men in [True, False]:
        tr_mask = train_df['T1_TeamID'] < 3000 if is_men else train_df['T1_TeamID'] >= 3000
        te_mask = test_matchups['T1_TeamID'] < 3000 if is_men else test_matchups['T1_TeamID'] >= 3000
        tr_s = train_df[tr_mask]
        te_s = test_matchups[te_mask]
        X_tr = tr_s[feat_cols].fillna(0).values
        y_tr = tr_s['T1_Win'].values
        X_te = te_s[feat_cols].fillna(0).values
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        C = cm if is_men else cw
        lbl = 'M' if is_men else 'W'

        if mode == 'lr':
            lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
            lr.fit(X_tr_s, y_tr)
            preds[te_mask] = lr.predict_proba(X_te_s)[:, 1]
            print(f"  {lbl} LR coefs: {np.round(lr.coef_[0], 3)}")

        elif mode == 'ensemble3':
            lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
            lr.fit(X_tr_s, y_tr)
            p_lr = lr.predict_proba(X_te_s)[:, 1]

            xgb = XGBClassifier(**xgb_deeper, random_state=42,
                                eval_metric='logloss', verbosity=0)
            xgb.fit(X_tr_s, y_tr)
            p_xgb = xgb.predict_proba(X_te_s)[:, 1]

            lgb = LGBMClassifier(**lgb_params, random_state=42)
            lgb.fit(X_tr_s, y_tr)
            p_lgb = lgb.predict_proba(X_te_s)[:, 1]

            preds[te_mask] = 0.6 * p_lr + 0.2 * p_xgb + 0.2 * p_lgb
            print(f"  {lbl} LR coefs: {np.round(lr.coef_[0], 3)}")
            print(f"  {lbl} blend: LR[{p_lr.min():.3f},{p_lr.max():.3f}] "
                  f"XGB[{p_xgb.min():.3f},{p_xgb.max():.3f}] "
                  f"LGB[{p_lgb.min():.3f},{p_lgb.max():.3f}]")

    return np.clip(preds, 0.025, 0.975)


# === GENERATE SUB7: Pure LR 7feat ===
print("=" * 60)
print("SUB7: Split LR, 7 features, C_m=0.25 C_w=0.15")
print("=" * 60)
preds7 = train_and_predict(train_data, all_matchups, FEATS7, mode='lr')
sub7 = sample[['ID']].copy()
sub7['Pred'] = preds7
sub7.to_csv('output/sub7_split_lr7_clean.csv', index=False)
print(f"\nSub7 saved: {len(sub7)} rows, "
      f"range [{preds7.min():.4f}, {preds7.max():.4f}]")

# === GENERATE SUB8: 3-model ensemble 60/20/20 ===
print("\n" + "=" * 60)
print("SUB8: 3-model ensemble (60% LR + 20% XGB + 20% LGBM)")
print("=" * 60)
preds8 = train_and_predict(train_data, all_matchups, FEATS7, mode='ensemble3')
sub8 = sample[['ID']].copy()
sub8['Pred'] = preds8
sub8.to_csv('output/sub8_3model_ensemble.csv', index=False)
print(f"\nSub8 saved: {len(sub8)} rows, "
      f"range [{preds8.min():.4f}, {preds8.max():.4f}]")

# === COMPARE BRACKETS ===
print("\n" + "=" * 60)
print("BRACKET COMPARISON (Men's)")
print("=" * 60)

teams = data_loader.load_teams()
tid2name = dict(zip(teams['TeamID'], teams['TeamName']))
mseeds_2026 = seeds[(seeds['Season'] == 2026) & (seeds['TeamID'] < 3000)]


def sim_bracket(sub_df, msdf):
    """Simulate men's bracket deterministically."""
    sub_dict = dict(zip(sub_df['ID'], sub_df['Pred']))

    def get_prob(t1, t2):
        lo, hi = min(t1, t2), max(t1, t2)
        key = f'2026_{lo}_{hi}'
        p = sub_dict.get(key, 0.5)
        return p if t1 == lo else 1 - p

    # Parse regions
    regions = {'W': [], 'X': [], 'Y': [], 'Z': []}
    for _, row in msdf.iterrows():
        reg = row['Seed'][0]
        snum = int(row['Seed'][1:3])
        regions[reg].append((snum, row['TeamID']))

    # Standard bracket ordering
    bracket_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    def sim_region(team_list):
        team_list.sort()
        seed_to_tid = {s: t for s, t in team_list}
        ordered = []
        for s in bracket_order:
            if s in seed_to_tid:
                ordered.append(seed_to_tid[s])
        # Handle play-in seeds (e.g., 16a/16b mapped to same seed num)
        remaining = [t for _, t in team_list if t not in ordered]
        ordered.extend(remaining)

        current = ordered[:16]
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    p = get_prob(current[i], current[i + 1])
                    winner = current[i] if p > 0.5 else current[i + 1]
                    nxt.append(winner)
                else:
                    nxt.append(current[i])
            current = nxt
        return current[0] if current else None

    rw = {}
    for reg in ['W', 'X', 'Y', 'Z']:
        rw[reg] = sim_region(regions[reg])

    # Final Four: W vs X, Y vs Z
    p_wx = get_prob(rw['W'], rw['X'])
    f1 = rw['W'] if p_wx > 0.5 else rw['X']
    p_yz = get_prob(rw['Y'], rw['Z'])
    f2 = rw['Y'] if p_yz > 0.5 else rw['Z']
    p_final = get_prob(f1, f2)
    champ = f1 if p_final > 0.5 else f2

    return rw, f1, f2, champ


sub1 = pd.read_csv('output/submission_phase1_baseline.csv')
sub5 = pd.read_csv('output/submission_phase3_ensemble.csv')

for name, sub_df in [('Sub1 (orig LR top6)', sub1),
                      ('Sub5 (combined LR)', sub5),
                      ('Sub7 (LR 7feat tuned)', sub7),
                      ('Sub8 (3-model ensemble)', sub8)]:
    rw, f1, f2, champ = sim_bracket(sub_df, mseeds_2026)
    print(f'\n{name}:')
    for reg in ['W', 'X', 'Y', 'Z']:
        tid = rw[reg]
        print(f'  {reg}: {tid2name.get(tid, tid)}')
    print(f'  FF: {tid2name.get(f1, f1)} vs {tid2name.get(f2, f2)}')
    print(f'  CHAMP: {tid2name.get(champ, champ)}')

# === DETAILED COMPARISON: Sub7 vs Sub8 disagreements ===
print("\n" + "=" * 60)
print("DISAGREEMENTS: Sub7 vs Sub8 (all matchups)")
print("=" * 60)
merged = pd.merge(sub7, sub8, on='ID', suffixes=('_7', '_8'))
merged['winner7'] = (merged['Pred_7'] > 0.5).astype(int)
merged['winner8'] = (merged['Pred_8'] > 0.5).astype(int)
disagree = merged[merged['winner7'] != merged['winner8']]
print(f"Total disagreements: {len(disagree)} out of {len(merged)}")
# Show the closest ones
disagree = disagree.copy()
disagree['margin7'] = abs(disagree['Pred_7'] - 0.5)
disagree['margin8'] = abs(disagree['Pred_8'] - 0.5)
disagree = disagree.sort_values('margin7')
for _, row in disagree.head(20).iterrows():
    parts = row['ID'].split('_')
    t1, t2 = int(parts[1]), int(parts[2])
    n1 = tid2name.get(t1, str(t1))
    n2 = tid2name.get(t2, str(t2))
    w7 = n1 if row['Pred_7'] > 0.5 else n2
    w8 = n1 if row['Pred_8'] > 0.5 else n2
    print(f"  {n1} vs {n2}: Sub7={row['Pred_7']:.4f}({w7}) Sub8={row['Pred_8']:.4f}({w8})")

# === Check actual 2026 results ===
print("\n" + "=" * 60)
print("ACTUAL 2026 RESULTS CHECK")
print("=" * 60)
# Howard(1178) vs UMBC(1370) => Howard won
# Texas(1345) vs NC State(1314) => Texas won
actuals = [
    ('Howard vs UMBC', '2026_1178_1370', 1178),   # Howard won
    ('Texas vs NC State', '2026_1314_1345', 1345),  # Texas won (T1=1314=NC State, so T1_win=0)
]
for desc, mid, winner in actuals:
    for sname, sdf in [('Sub1', sub1), ('Sub5', sub5), ('Sub7', sub7), ('Sub8', sub8)]:
        row = sdf[sdf['ID'] == mid]
        if len(row) > 0:
            p = row.iloc[0]['Pred']  # P(T1 wins)
            t1, t2 = int(mid.split('_')[1]), int(mid.split('_')[2])
            p_winner = p if winner == t1 else 1 - p
            correct = 'OK' if p_winner > 0.5 else 'MISS'
            print(f"  {sname} {desc}: P(winner)={p_winner:.4f} [{correct}]")
    print()

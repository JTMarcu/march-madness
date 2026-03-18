"""Print full bracket simulation for the final recommended model (sub8)."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src import data_loader

teams = data_loader.load_teams()
seeds = data_loader.load_tourney_seeds()
tid2name = dict(zip(teams['TeamID'], teams['TeamName']))

# Load sub8
sub8 = pd.read_csv('output/sub8_3model_ensemble.csv')
sub_dict = dict(zip(sub8['ID'], sub8['Pred']))

# 2026 men's seeds
ms = seeds[(seeds['Season'] == 2026) & (seeds['TeamID'] < 3000)].copy()

def get_prob(t1, t2):
    lo, hi = min(t1, t2), max(t1, t2)
    key = f'2026_{lo}_{hi}'
    p = sub_dict.get(key, 0.5)
    return p if t1 == lo else 1 - p

def name(tid):
    return tid2name.get(tid, str(tid))

# Parse seeds
regions = {'W': {}, 'X': {}, 'Y': {}, 'Z': {}}
region_names = {'W': 'East', 'X': 'South', 'Y': 'Midwest', 'Z': 'West'}
for _, row in ms.iterrows():
    reg = row['Seed'][0]
    snum = int(row['Seed'][1:3])
    suffix = row['Seed'][3:] if len(row['Seed']) > 3 else ''
    regions[reg][(snum, suffix)] = row['TeamID']

# Standard bracket order
bracket_matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

def sim_region_detailed(reg_code, seed_to_team):
    print(f"\n{'='*60}")
    print(f"  {region_names[reg_code]} REGION ({reg_code})")
    print(f"{'='*60}")
    
    # Handle play-in: if multiple teams at same seed number, pick the one
    # that won the play-in or just take whoever is available
    def get_team(seed_num):
        # Try exact match first
        if (seed_num, '') in seed_to_team:
            return seed_to_team[(seed_num, '')]
        # Try with suffix (play-in)
        candidates = [(k, v) for k, v in seed_to_team.items() if k[0] == seed_num]
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) == 2:
            t1 = candidates[0][1]
            t2 = candidates[1][1]
            p = get_prob(t1, t2)
            winner = t1 if p > 0.5 else t2
            loser = t2 if p > 0.5 else t1
            print(f"  Play-in: ({seed_num}) {name(t1)} vs {name(t2)} -> "
                  f"{name(winner)} ({p if winner == t1 else 1-p:.1%})")
            return winner
        return None

    # Round of 64
    print(f"\n  Round of 64:")
    r1_winners = []
    for hi, lo in bracket_matchups:
        t_hi = get_team(hi)
        t_lo = get_team(lo)
        if t_hi is None or t_lo is None:
            r1_winners.append(t_hi or t_lo)
            continue
        p = get_prob(t_hi, t_lo)
        winner = t_hi if p > 0.5 else t_lo
        marker = "" if abs(p - 0.5) > 0.15 else " *"
        print(f"    ({hi:2d}) {name(t_hi):20s} vs ({lo:2d}) {name(t_lo):20s} -> "
              f"{name(winner):20s} ({max(p, 1-p):.1%}){marker}")
        r1_winners.append(winner)

    # Round of 32
    print(f"\n  Round of 32:")
    r2_winners = []
    for i in range(0, len(r1_winners), 2):
        t1, t2 = r1_winners[i], r1_winners[i + 1]
        p = get_prob(t1, t2)
        winner = t1 if p > 0.5 else t2
        marker = "" if abs(p - 0.5) > 0.15 else " *"
        print(f"    {name(t1):20s} vs {name(t2):20s} -> "
              f"{name(winner):20s} ({max(p, 1-p):.1%}){marker}")
        r2_winners.append(winner)

    # Sweet 16
    print(f"\n  Sweet 16:")
    s16_winners = []
    for i in range(0, len(r2_winners), 2):
        t1, t2 = r2_winners[i], r2_winners[i + 1]
        p = get_prob(t1, t2)
        winner = t1 if p > 0.5 else t2
        marker = "" if abs(p - 0.5) > 0.15 else " *"
        print(f"    {name(t1):20s} vs {name(t2):20s} -> "
              f"{name(winner):20s} ({max(p, 1-p):.1%}){marker}")
        s16_winners.append(winner)

    # Elite 8
    print(f"\n  Elite 8:")
    t1, t2 = s16_winners[0], s16_winners[1]
    p = get_prob(t1, t2)
    reg_winner = t1 if p > 0.5 else t2
    marker = "" if abs(p - 0.5) > 0.15 else " *"
    print(f"    {name(t1):20s} vs {name(t2):20s} -> "
          f"{name(reg_winner):20s} ({max(p, 1-p):.1%}){marker}")

    return reg_winner


# Simulate all 4 regions
rw = {}
for reg in ['W', 'X', 'Y', 'Z']:
    rw[reg] = sim_region_detailed(reg, regions[reg])

# Final Four
print(f"\n{'='*60}")
print(f"  FINAL FOUR")
print(f"{'='*60}")

# W vs X
t1, t2 = rw['W'], rw['X']
p = get_prob(t1, t2)
ff1 = t1 if p > 0.5 else t2
print(f"\n  {region_names['W']} vs {region_names['X']}:")
print(f"    {name(t1):20s} vs {name(t2):20s} -> {name(ff1):20s} ({max(p, 1-p):.1%})")

# Y vs Z
t1, t2 = rw['Y'], rw['Z']
p = get_prob(t1, t2)
ff2 = t1 if p > 0.5 else t2
print(f"\n  {region_names['Y']} vs {region_names['Z']}:")
print(f"    {name(t1):20s} vs {name(t2):20s} -> {name(ff2):20s} ({max(p, 1-p):.1%})")

# Championship
print(f"\n{'='*60}")
print(f"  CHAMPIONSHIP")
print(f"{'='*60}")
p = get_prob(ff1, ff2)
champ = ff1 if p > 0.5 else ff2
print(f"\n    {name(ff1):20s} vs {name(ff2):20s} -> {name(champ):20s} ({max(p, 1-p):.1%})")
print(f"\n  {'*'*40}")
print(f"  * PREDICTED CHAMPION: {name(champ):^14s} *")
print(f"  {'*'*40}")

# Final Four summary
print(f"\n  FINAL FOUR TEAMS:")
for reg in ['W', 'X', 'Y', 'Z']:
    print(f"    {region_names[reg]:10s}: {name(rw[reg])}")

# Note coin-flip games (* = margin < 15%)
print(f"\n  (* = close game, margin < 15%)")

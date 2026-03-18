"""Quick validation of all 5 submission CSVs."""
import pandas as pd
import os

files = ['sub1_split_lr6.csv', 'sub2_split_lr4.csv', 'sub3_split_xgb6.csv',
         'sub4_ensemble_lr_xgb.csv', 'sub5_combined_lr6.csv']
sample = pd.read_csv('data/SampleSubmissionStage2.csv')
sample_ids = set(sample['ID'].values)

print(f'Expected: {len(sample)} rows, columns: {list(sample.columns)}\n')

all_good = True
for f in files:
    path = os.path.join('output', f)
    df = pd.read_csv(path)
    ids_match = set(df['ID'].values) == sample_ids
    cols_ok = list(df.columns) == ['ID', 'Pred']
    rows_ok = len(df) == len(sample)
    pred_range = df['Pred'].min() >= 0 and df['Pred'].max() <= 1
    no_nulls = df['Pred'].notna().all()
    ok = all([ids_match, cols_ok, rows_ok, pred_range, no_nulls])
    if not ok:
        all_good = False
    status = 'PASS' if ok else 'FAIL'
    null_count = df['Pred'].isna().sum()
    print(f'{status} | {f:30s} | rows={len(df)} | IDs_match={ids_match} | '
          f'range=[{df["Pred"].min():.4f}, {df["Pred"].max():.4f}] | nulls={null_count}')

print('\n' + ('ALL VALID' if all_good else 'SOME FAILURES'))

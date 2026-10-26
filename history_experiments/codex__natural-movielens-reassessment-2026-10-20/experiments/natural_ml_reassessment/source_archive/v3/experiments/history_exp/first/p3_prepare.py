"""P3: build the KuaiRand-Pure table that examples/run_kuaishou.py expects
(data_path/kuai_rand.csv), from the official KuaiRand-Pure release.

Source: official KuaiRand-Pure files (obtained via the Kaggle mirror
arashnic/an-unbiased-sequential-recommendation-dataset, which packages the
official Zenodo release; LICENSE CC BY-SA 4.0 preserved in the archive).

Join: logs (3 chronological files) -> user_features_pure (user_id)
      -> video_features_basic_pure + video_features_statistic_pure.counts (video_id)
Domain split (paper): tab in {0,1,4} -> domains 0/1/2.
"""
import os

import pandas as pd

SRC = "D:/CODE/data/kuairand/kaggle_raw/KuaiRand-Pure/KuaiRand-Pure/data"
DST = "D:/CODE/data/kuairand/processed"
os.makedirs(DST, exist_ok=True)

X_USED_COLS = ['tab', 'user_active_degree', 'is_live_streamer', 'is_video_author',
               'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
               'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
               'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7',
               'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11', 'onehot_feat12',
               'onehot_feat13', 'onehot_feat14', 'onehot_feat15', 'onehot_feat16',
               'onehot_feat17', 'video_type', 'upload_dt', 'upload_type', 'server_width',
               'server_height', 'music_type', 'tag', 'counts']

LOG_FILES = ['log_standard_4_08_to_4_21_pure.csv',   # chronological order
             'log_standard_4_22_to_5_08_pure.csv',
             'log_random_4_22_to_5_08_pure.csv']


def main():
    logs = []
    for f in LOG_FILES:
        df = pd.read_csv(os.path.join(SRC, f),
                         usecols=['user_id', 'video_id', 'tab', 'is_click', 'long_view',
                                  'date', 'time_ms'])
        print(f"{f}: {len(df)} rows")
        logs.append(df)
    data = pd.concat(logs, ignore_index=True)
    print("concatenated:", len(data))
    # Global chronological order. Rationale: the downstream pipeline splits
    # train:val:test = 8:1:1 BY ROW ORDER; naive file concatenation puts the
    # random-exposure log (tab=1 only) last, leaving val with a single domain.
    # Sorting by (date, time_ms) is deterministic, leak-free and keeps all
    # three domains present in every split (verified below).
    data = data.sort_values(['date', 'time_ms'], kind='mergesort').reset_index(drop=True)
    print("sorted by (date, time_ms)")

    n_raw = len(data)
    data = data[data['tab'].isin([0, 1, 4])].copy()
    print(f"tab in {{0,1,4}}: {len(data)} ({len(data)/n_raw*100:.2f}% of all logs)")

    users = pd.read_csv(os.path.join(SRC, 'user_features_pure.csv'))
    vbasic = pd.read_csv(os.path.join(SRC, 'video_features_basic_pure.csv'))
    vstat = pd.read_csv(os.path.join(SRC, 'video_features_statistic_pure.csv'),
                        usecols=['video_id', 'counts'])
    print(f"feature tables: users={len(users)}, videos_basic={len(vbasic)}, videos_stat={len(vstat)}")

    before = len(data)
    data = data.merge(users, on='user_id', how='inner')
    print(f"after user join: {len(data)} (lost {before-len(data)})")
    before = len(data)
    data = data.merge(vbasic, on='video_id', how='inner').merge(vstat, on='video_id', how='inner')
    print(f"after video join: {len(data)} (lost {before-len(data)})")
    # inner merges preserve left order in pandas, but re-sort defensively so
    # the row order written to disk is strictly chronological
    data = data.sort_values(['date', 'time_ms'], kind='mergesort').reset_index(drop=True)

    for c in X_USED_COLS + ['is_click', 'long_view']:
        if c in data.columns and data[c].isna().any():
            data[c] = data[c].fillna(-1)
    for c in X_USED_COLS:
        data[c] = data[c].astype(str)

    n = len(data)
    tr, va = int(n * 0.8), int(n * 0.9)
    print("")
    print("== per-split domain presence (rows) ==")
    for name, seg in [('train', data.iloc[:tr]), ('val', data.iloc[tr:va]), ('test', data.iloc[va:])]:
        c = seg['tab'].value_counts().to_dict()
        print(f"  {name}: " + ", ".join(f"tab{k}={v}" for k, v in sorted(c.items())))

    out = data[X_USED_COLS + ['is_click', 'long_view']]
    out_path = os.path.join(DST, 'kuai_rand.csv')
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows, {os.path.getsize(out_path)/1e6:.0f} MB)")

    print("\n== per-domain stats (before LabelEncoder; vs paper Table 1) ==")
    paper = {0: (15398, 6233, 178087, 7.10), 1: (27049, 7580, 2236414, 89.18),
             4: (11809, 4633, 93165, 3.72)}
    tot = len(data)
    for tab, dom in [(0, 0), (1, 1), (4, 2)]:
        sub = data[data['tab'] == str(tab)]
        pu, pi, pn, ps = paper[tab]
        u, i = sub['user_id'].nunique(), sub['video_id'].nunique()
        print(f"domain {dom} (tab={tab}): users {u} (paper {pu}, diff {u-pu:+d}), "
              f"items {i} (paper {pi}, diff {i-pi:+d}), instances {len(sub)} "
              f"(paper {pn}, diff {len(sub)-pn:+d}), share {len(sub)/tot*100:.2f}% (paper {ps}%)")


if __name__ == "__main__":
    main()

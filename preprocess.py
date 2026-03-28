# coding=utf-8
"""
将 TSMC2014 原始数据预处理为 H-DCHL-B 可直接读取的格式。

当前版本遵循以下设计：
1. 延续 DCHL 的 train/test 文件结构，方便训练入口复用；
2. 额外输出 POI-Region、POI-Category 两类异构语义映射；
3. 先支持 TKY/NYC 两类 TSMC2014 原始文件。
"""

import argparse
import datetime as dt
import os
from collections import defaultdict

from utils import save_dict_to_pkl, save_list_with_pkl


_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(latitude, longitude, precision=6):
    """使用纯 Python 实现 geohash 编码，避免额外依赖。"""
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    geohash_chars = []
    bit = 0
    ch = 0
    even = True

    while len(geohash_chars) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash_chars.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(geohash_chars)


def parse_time(timestr):
    """解析 TSMC2014 原始时间字段。"""
    return dt.datetime.strptime(timestr, "%a %b %d %H:%M:%S %z %Y")


def load_raw_events(raw_path):
    """读取原始事件，并按用户聚合。"""
    user_events = defaultdict(list)
    poi_users = defaultdict(set)
    poi_coos_raw = {}
    poi_cat_raw = {}

    with open(raw_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            user_id = parts[0]
            poi_id = parts[1]
            cat_id = parts[2]
            lat = float(parts[4])
            lon = float(parts[5])
            ts = parse_time(parts[7])

            user_events[user_id].append((ts, poi_id, lat, lon, cat_id))
            poi_users[poi_id].add(user_id)
            poi_coos_raw[poi_id] = (lat, lon)
            poi_cat_raw[poi_id] = cat_id

    return user_events, poi_users, poi_coos_raw, poi_cat_raw


def build_sessions(user_events, keep_poi_set, session_gap_hours=24, min_session_len=3):
    """按 24 小时时间窗切分 session，并过滤过短 session。"""
    split_delta = dt.timedelta(hours=session_gap_hours)
    user_sessions = {}
    for user_id, events in user_events.items():
        events_sorted = sorted(events, key=lambda x: x[0])
        filtered = [e for e in events_sorted if e[1] in keep_poi_set]
        if not filtered:
            continue

        sessions = []
        current = [filtered[0]]
        for i in range(1, len(filtered)):
            if filtered[i][0] - filtered[i - 1][0] > split_delta:
                sessions.append(current)
                current = [filtered[i]]
            else:
                current.append(filtered[i])
        sessions.append(current)

        valid_sessions = [s for s in sessions if len(s) >= min_session_len]
        if valid_sessions:
            user_sessions[user_id] = valid_sessions
    return user_sessions


def remap_and_split(user_sessions_raw, poi_coos_raw, poi_cat_raw, train_ratio=0.8, min_user_sessions=3):
    """
    remap 用户/POI/Category/Region 索引，并构造 train/test。

    这里仍延续 DCHL 的单标签格式：
    - train 输入为训练部分的前 n-1 个 session，label 为第 n 个 session 的首个 POI
    - test 输入为测试部分的前 n-1 个 session，label 为最后一个 session 的首个 POI
    """
    valid_users = []
    for uid, sessions in user_sessions_raw.items():
        if len(sessions) < min_user_sessions:
            continue
        train_cut = int(len(sessions) * train_ratio)
        train_raw = sessions[:train_cut]
        test_raw = sessions[train_cut:]
        if len(train_raw) < 2 or len(test_raw) < 2:
            continue
        valid_users.append(uid)
    valid_users = sorted(valid_users)

    poi_set = set()
    for uid in valid_users:
        for session in user_sessions_raw[uid]:
            for _, poi_id, _, _, _ in session:
                poi_set.add(poi_id)
    poi_list = sorted(list(poi_set))

    user2idx = {u: i for i, u in enumerate(valid_users)}
    poi2idx = {p: i for i, p in enumerate(poi_list)}

    cat_values = sorted({poi_cat_raw[p] for p in poi_list})
    cat2idx = {c: i for i, c in enumerate(cat_values)}

    geohash_values = sorted({geohash_encode(poi_coos_raw[p][0], poi_coos_raw[p][1], precision=6) for p in poi_list})
    geohash2idx = {g: i for i, g in enumerate(geohash_values)}

    train_sessions_dict, train_labels_dict = {}, {}
    test_sessions_dict, test_labels_dict = {}, {}
    poi_coos_idx, poi_cat_idx, poi_region_idx = {}, {}, {}

    for raw_poi, poi_idx in poi2idx.items():
        lat, lon = poi_coos_raw[raw_poi]
        poi_coos_idx[poi_idx] = [lat, lon]
        poi_cat_idx[poi_idx] = cat2idx[poi_cat_raw[raw_poi]]
        poi_region_idx[poi_idx] = geohash2idx[geohash_encode(lat, lon, precision=6)]

    for raw_user in valid_users:
        user_idx = user2idx[raw_user]
        sessions = user_sessions_raw[raw_user]
        train_cut = int(len(sessions) * train_ratio)
        train_raw = sessions[:train_cut]
        test_raw = sessions[train_cut:]

        train_sessions_dict[user_idx] = [[poi2idx[e[1]] for e in session] for session in train_raw[:-1]]
        train_labels_dict[user_idx] = poi2idx[train_raw[-1][0][1]]
        test_sessions_dict[user_idx] = [[poi2idx[e[1]] for e in session] for session in test_raw[:-1]]
        test_labels_dict[user_idx] = poi2idx[test_raw[-1][0][1]]

    meta = {
        "num_users": len(valid_users),
        "num_pois": len(poi_list),
        "padding_idx": len(poi_list),
        "num_categories": len(cat_values),
        "num_regions": len(geohash_values),
        "train_ratio": train_ratio,
        "session_gap_hours": 24,
        "min_user_sessions": min_user_sessions,
        "geohash_precision": 6,
        # "geohash_precision": 5,
    }

    return (
        train_sessions_dict,
        train_labels_dict,
        test_sessions_dict,
        test_labels_dict,
        poi_coos_idx,
        poi_cat_idx,
        poi_region_idx,
        meta,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="datasets/dataset_TSMC2014_TKY.txt")
    parser.add_argument("--output_dir", type=str, default="datasets/TKY")
    parser.add_argument("--min_poi_users", type=int, default=5)
    parser.add_argument("--min_session_len", type=int, default=3)
    parser.add_argument("--min_user_sessions", type=int, default=3)
    parser.add_argument("--session_gap_hours", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] 读取原始数据...")
    user_events, poi_users, poi_coos_raw, poi_cat_raw = load_raw_events(args.raw_path)

    print("[2/4] 过滤低频 POI 并切分 session...")
    keep_poi_set = {poi for poi, users in poi_users.items() if len(users) >= args.min_poi_users}
    user_sessions_raw = build_sessions(
        user_events,
        keep_poi_set,
        session_gap_hours=args.session_gap_hours,
        min_session_len=args.min_session_len,
    )

    print("[3/4] remap 与 train/test 构造...")
    outputs = remap_and_split(
        user_sessions_raw,
        poi_coos_raw,
        poi_cat_raw,
        train_ratio=args.train_ratio,
        min_user_sessions=args.min_user_sessions,
    )
    (
        train_sessions_dict,
        train_labels_dict,
        test_sessions_dict,
        test_labels_dict,
        poi_coos_idx,
        poi_cat_idx,
        poi_region_idx,
        meta,
    ) = outputs

    print("[4/4] 保存文件...")
    save_list_with_pkl(os.path.join(args.output_dir, "train_poi_zero.pkl"), [train_sessions_dict, train_labels_dict])
    save_list_with_pkl(os.path.join(args.output_dir, "test_poi_zero.pkl"), [test_sessions_dict, test_labels_dict])
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_coos.pkl"), poi_coos_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_category.pkl"), poi_cat_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_region.pkl"), poi_region_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "meta.pkl"), meta)
    print("预处理完成：", meta)


if __name__ == "__main__":
    main()

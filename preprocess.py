# coding=utf-8
"""
将 TSMC2014 原始数据预处理为“标准 Next POI 预测”所需的数据格式。

本版本与此前最大的不同是：
1. 按用户内时间顺序切分训练/测试 session；
2. 在每个 session 内展开 prefix -> next POI 样本；
3. 同时保留训练阶段用户历史，用于构造协同图、转移图等全局结构；
4. 额外保存 label_category / label_region，便于后续多任务或分析使用。
"""

import argparse
import datetime as dt
import os
from collections import defaultdict

from utils import save_dict_to_pkl, save_list_with_pkl


_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(latitude, longitude, precision=5):
    """使用纯 Python 实现 geohash 编码。"""
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
    """读取原始事件并按用户聚合。"""
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


def build_entity_mappings(valid_users, user_sessions_raw, poi_coos_raw, poi_cat_raw, geohash_precision=5):
    """统一 remap 用户、POI、类别和区域索引。"""
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

    geohash_values = sorted({geohash_encode(poi_coos_raw[p][0], poi_coos_raw[p][1], precision=geohash_precision) for p in poi_list})
    geohash2idx = {g: i for i, g in enumerate(geohash_values)}

    poi_coos_idx, poi_cat_idx, poi_region_idx = {}, {}, {}
    for raw_poi, poi_idx in poi2idx.items():
        lat, lon = poi_coos_raw[raw_poi]
        poi_coos_idx[poi_idx] = [lat, lon]
        poi_cat_idx[poi_idx] = cat2idx[poi_cat_raw[raw_poi]]
        poi_region_idx[poi_idx] = geohash2idx[geohash_encode(lat, lon, precision=geohash_precision)]

    return user2idx, poi2idx, poi_coos_idx, poi_cat_idx, poi_region_idx, len(cat_values), len(geohash_values)


def build_prefix_samples(remapped_sessions, poi_cat_idx, poi_region_idx):
    """将 session 内部展开为 prefix -> next POI 样本。"""
    samples = []
    for user_idx, sessions in remapped_sessions.items():
        for session_idx, session in enumerate(sessions):
            # session 长度已在预处理前保证 >= 3
            for prefix_end in range(1, len(session)):
                prefix = session[:prefix_end]
                label_poi = session[prefix_end]
                sample = {
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "prefix_pois": prefix,
                    "label_poi": label_poi,
                    "label_category": poi_cat_idx[label_poi],
                    "label_region": poi_region_idx[label_poi],
                }
                samples.append(sample)
    return samples


def remap_sessions_for_users(user_ids, user_sessions_raw, user2idx, poi2idx):
    """将指定用户集合的 session 重映射为整数索引。"""
    remapped = {}
    for raw_user in user_ids:
        user_idx = user2idx[raw_user]
        remapped[user_idx] = [[poi2idx[e[1]] for e in session] for session in user_sessions_raw[raw_user]]
    return remapped


def split_users_sessions(user_sessions_raw, train_ratio=0.8, min_user_sessions=3):
    """
    对每个用户按时间顺序做 80/20 session 切分。

    这里只要求用户总 session 数 >= min_user_sessions，
    不再额外施加 train_raw>=2、test_raw>=2 的限制，
    因为标准 next POI 样本会在 session 内部展开 prefix -> next POI。
    """
    valid_users = []
    train_user_sessions_raw = {}
    test_user_sessions_raw = {}
    total_train_sessions = 0
    total_test_sessions = 0

    for uid, sessions in user_sessions_raw.items():
        if len(sessions) < min_user_sessions:
            continue
        train_cut = max(1, int(len(sessions) * train_ratio))
        if train_cut >= len(sessions):
            train_cut = len(sessions) - 1
        train_raw = sessions[:train_cut]
        test_raw = sessions[train_cut:]

        # 只要训练和测试至少各有 1 个 session 就保留用户
        if len(train_raw) < 1 or len(test_raw) < 1:
            continue
        valid_users.append(uid)
        train_user_sessions_raw[uid] = train_raw
        test_user_sessions_raw[uid] = test_raw
        total_train_sessions += len(train_raw)
        total_test_sessions += len(test_raw)

    return valid_users, train_user_sessions_raw, test_user_sessions_raw, total_train_sessions, total_test_sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="datasets/dataset_TSMC2014_TKY.txt")
    parser.add_argument("--output_dir", type=str, default="datasets/TKY")
    parser.add_argument("--min_poi_users", type=int, default=5)
    parser.add_argument("--min_session_len", type=int, default=3)
    parser.add_argument("--min_user_sessions", type=int, default=3)
    parser.add_argument("--session_gap_hours", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--geohash_precision", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] 读取原始数据...")
    user_events, poi_users, poi_coos_raw, poi_cat_raw = load_raw_events(args.raw_path)

    print("[2/5] 过滤低频 POI 并切分 session...")
    keep_poi_set = {poi for poi, users in poi_users.items() if len(users) >= args.min_poi_users}
    user_sessions_raw = build_sessions(
        user_events,
        keep_poi_set,
        session_gap_hours=args.session_gap_hours,
        min_session_len=args.min_session_len,
    )

    print("[3/5] 用户内按时间做 80/20 session 切分...")
    valid_users, train_user_sessions_raw, test_user_sessions_raw, total_train_sessions, total_test_sessions = split_users_sessions(
        user_sessions_raw,
        train_ratio=args.train_ratio,
        min_user_sessions=args.min_user_sessions,
    )

    user2idx, poi2idx, poi_coos_idx, poi_cat_idx, poi_region_idx, num_categories, num_regions = build_entity_mappings(
        valid_users,
        user_sessions_raw,
        poi_coos_raw,
        poi_cat_raw,
        geohash_precision=args.geohash_precision,
    )

    train_user_sessions = remap_sessions_for_users(valid_users, train_user_sessions_raw, user2idx, poi2idx)
    test_user_sessions = remap_sessions_for_users(valid_users, test_user_sessions_raw, user2idx, poi2idx)

    print("[4/5] 生成标准 prefix -> next POI 训练/测试样本...")
    train_samples = build_prefix_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
    test_samples = build_prefix_samples(test_user_sessions, poi_cat_idx, poi_region_idx)

    meta = {
        "num_users": len(valid_users),
        "num_pois": len(poi2idx),
        "padding_idx": len(poi2idx),
        "num_categories": num_categories,
        "num_regions": num_regions,
        "train_ratio": args.train_ratio,
        "session_gap_hours": args.session_gap_hours,
        "min_session_len": args.min_session_len,
        "min_user_sessions": args.min_user_sessions,
        "geohash_precision": args.geohash_precision,
        "num_train_sessions": total_train_sessions,
        "num_test_sessions": total_test_sessions,
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
    }

    print("[5/5] 保存文件...")
    save_list_with_pkl(os.path.join(args.output_dir, "train_samples.pkl"), train_samples)
    save_list_with_pkl(os.path.join(args.output_dir, "test_samples.pkl"), test_samples)
    save_dict_to_pkl(os.path.join(args.output_dir, "train_user_sessions.pkl"), train_user_sessions)
    save_dict_to_pkl(os.path.join(args.output_dir, "test_user_sessions.pkl"), test_user_sessions)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_coos.pkl"), poi_coos_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_category.pkl"), poi_cat_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_region.pkl"), poi_region_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "meta.pkl"), meta)

    print("预处理完成：")
    print(meta)


if __name__ == "__main__":
    main()

# coding=utf-8
"""
H-DCHL-B 数据集定义。

设计原则：
1. 延续 DCHL 的 batch 组织方式，样本仍以用户为中心；
2. 但在 dataset 内部额外构建异构语义结构：POI-Region、POI-Category；
3. 第一版仅做纯结构增益，因此暂不加入掩码任务所需的额外字段。
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import (
    csr_matrix_drop_edge,
    gen_sparse_H_poi_category,
    gen_sparse_H_poi_region,
    gen_sparse_H_user,
    gen_sparse_directed_H_poi,
    get_all_users_seqs,
    get_hyper_deg,
    get_user_complete_traj,
    get_user_reverse_traj,
    load_dict_from_pkl,
    load_list_with_pkl,
    transform_csr_matrix_to_tensor,
)


class HDCHLBDataset(Dataset):
    """
    H-DCHL-B 的核心数据集。

    每条样本仍然对应一个用户，但 dataset 对象内部缓存全局异构图结构，
    这样模型 forward 时可以像 DCHL 一样同时使用 batch 与全图信息。
    """

    def __init__(self, data_filename, data_dir, args, device):
        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]
        self.meta = load_dict_from_pkl(f"{data_dir}/meta.pkl")
        self.pois_coos_dict = load_dict_from_pkl(f"{data_dir}/poi_coos.pkl")
        self.poi_category_dict = load_dict_from_pkl(f"{data_dir}/poi_category.pkl")
        self.poi_region_dict = load_dict_from_pkl(f"{data_dir}/poi_region.pkl")

        self.num_users = self.meta["num_users"]
        self.num_pois = self.meta["num_pois"]
        self.num_categories = self.meta["num_categories"]
        self.num_regions = self.meta["num_regions"]
        self.padding_idx = self.meta["padding_idx"]
        self.keep_rate = args.keep_rate
        self.keep_rate_poi = args.keep_rate_poi
        self.device = device

        # 将多 session 用户行为整理为完整轨迹，供转移分支使用
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.sessions_dict)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # 构建长期协同分支：POI-User 关联矩阵
        self.H_pu = gen_sparse_H_user(self.sessions_dict, self.num_pois, self.num_users)
        self.H_pu = csr_matrix_drop_edge(self.H_pu, self.keep_rate)
        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = transform_csr_matrix_to_tensor(self.Deg_H_pu * self.H_pu).to(device)

        # User -> POI 归一化矩阵，用于从 POI 分支聚合用户表示
        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = transform_csr_matrix_to_tensor(self.Deg_H_up * self.H_up).to(device)

        # 构建 Region 异构语义分支
        self.H_pr = gen_sparse_H_poi_region(self.poi_region_dict, self.num_pois, self.num_regions)
        self.Deg_H_pr = get_hyper_deg(self.H_pr)
        self.HG_pr = transform_csr_matrix_to_tensor(self.Deg_H_pr * self.H_pr).to(device)

        self.H_rp = self.H_pr.T
        self.Deg_H_rp = get_hyper_deg(self.H_rp)
        self.HG_rp = transform_csr_matrix_to_tensor(self.Deg_H_rp * self.H_rp).to(device)

        # 构建 Category 异构语义分支
        self.H_pc = gen_sparse_H_poi_category(self.poi_category_dict, self.num_pois, self.num_categories)
        self.Deg_H_pc = get_hyper_deg(self.H_pc)
        self.HG_pc = transform_csr_matrix_to_tensor(self.Deg_H_pc * self.H_pc).to(device)

        self.H_cp = self.H_pc.T
        self.Deg_H_cp = get_hyper_deg(self.H_cp)
        self.HG_cp = transform_csr_matrix_to_tensor(self.Deg_H_cp * self.H_cp).to(device)

        # 保留 DCHL 风格的有向 POI 转移分支
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, self.num_pois)
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, self.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.Deg_H_poi_src * self.H_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.Deg_H_poi_tar * self.H_poi_tar).to(device)

        # 用于 batch 中反向/顺序轨迹输入，保持与原 DCHL 接口一致
        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        self.pad_all_train_sessions = pad_sequence(
            self.all_train_sessions, batch_first=True, padding_value=self.padding_idx
        ).to(device)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]
        return {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }


def collate_fn(batch, padding_value):
    """将一个 batch 中不同长度的用户轨迹 padding 到统一长度。"""
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])

    pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=padding_value)
    pad_user_rev_seq = pad_sequence(batch_user_rev_seq, batch_first=True, padding_value=padding_value)
    pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=0)

    return {
        "user_idx": torch.stack(batch_user_idx),
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": torch.stack(batch_user_seq_len),
        "user_seq_mask": pad_user_seq_mask,
        "label": torch.stack(batch_label),
    }

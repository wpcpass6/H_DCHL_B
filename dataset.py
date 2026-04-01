# coding=utf-8
"""
H-DCHL-B 数据集定义（标准 Next POI 版本）。

核心变化：
1. 每个样本不再对应一个用户，而是对应一个 session 内的 prefix -> next POI；
2. 图结构统一基于训练阶段用户历史构建；
3. Category/Region 仍作为全局静态属性图使用。
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import (
    build_poi_region_from_coos,
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
    标准 Next POI 样本级数据集。

    说明：
    - 一个样本 = 一个 prefix -> next POI；
    - 但图结构仍然基于训练阶段全量用户历史缓存一次，避免每个样本单独构图。
    """

    def __init__(self, samples_filename, graph_data_dir, args, device):
        self.samples = load_list_with_pkl(samples_filename)
        self.meta = load_dict_from_pkl(f"{graph_data_dir}/meta.pkl")
        self.train_user_sessions = load_dict_from_pkl(f"{graph_data_dir}/train_user_sessions.pkl")
        self.pois_coos_dict = load_dict_from_pkl(f"{graph_data_dir}/poi_coos.pkl")
        self.poi_category_dict = load_dict_from_pkl(f"{graph_data_dir}/poi_category.pkl")

        # Region 映射支持两种来源：显式文件优先，否则基于坐标动态生成。
        if getattr(args, "poi_region_path", None):
            self.poi_region_dict = load_dict_from_pkl(args.poi_region_path)
            self.num_regions = max(self.poi_region_dict.values()) + 1 if self.poi_region_dict else 0
        else:
            self.poi_region_dict, self.num_regions, _ = build_poi_region_from_coos(
                self.pois_coos_dict,
                precision=args.region_precision,
            )

        self.num_users = self.meta["num_users"]
        self.num_pois = self.meta["num_pois"]
        self.num_categories = self.meta["num_categories"]
        self.padding_idx = self.meta["padding_idx"]
        self.keep_rate = args.keep_rate
        self.keep_rate_poi = args.keep_rate_poi
        self.device = device

        # 预先构造 poi->category 与 poi->region 的张量索引，便于在训练时快速取标签
        self.poi_category_tensor = torch.zeros(self.num_pois, dtype=torch.long, device=device)
        for poi_idx, cat_idx in self.poi_category_dict.items():
            self.poi_category_tensor[poi_idx] = cat_idx

        self.poi_region_tensor = torch.zeros(self.num_pois, dtype=torch.long, device=device)
        for poi_idx, region_idx in self.poi_region_dict.items():
            self.poi_region_tensor[poi_idx] = region_idx

        # 图构建只基于训练阶段用户历史，避免测试信息泄漏
        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(self.train_user_sessions)
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        # User-POI 协同图：只使用训练历史中的用户访问行为
        self.H_pu = gen_sparse_H_user(self.train_user_sessions, self.num_pois, self.num_users)
        self.H_pu = csr_matrix_drop_edge(self.H_pu, self.keep_rate)
        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = transform_csr_matrix_to_tensor(self.Deg_H_pu * self.H_pu).to(device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = transform_csr_matrix_to_tensor(self.Deg_H_up * self.H_up).to(device)

        # Region/Category 为静态属性图，允许基于全局 POI 属性构建
        self.H_pr = gen_sparse_H_poi_region(self.poi_region_dict, self.num_pois, self.num_regions)
        self.Deg_H_pr = get_hyper_deg(self.H_pr)
        self.HG_pr = transform_csr_matrix_to_tensor(self.Deg_H_pr * self.H_pr).to(device)

        self.H_rp = self.H_pr.T
        self.Deg_H_rp = get_hyper_deg(self.H_rp)
        self.HG_rp = transform_csr_matrix_to_tensor(self.Deg_H_rp * self.H_rp).to(device)

        self.H_pc = gen_sparse_H_poi_category(self.poi_category_dict, self.num_pois, self.num_categories)
        self.Deg_H_pc = get_hyper_deg(self.H_pc)
        self.HG_pc = transform_csr_matrix_to_tensor(self.Deg_H_pc * self.H_pc).to(device)

        self.H_cp = self.H_pc.T
        self.Deg_H_cp = get_hyper_deg(self.H_cp)
        self.HG_cp = transform_csr_matrix_to_tensor(self.Deg_H_cp * self.H_cp).to(device)

        # 转移图只基于训练行为构建，且采用相邻 POI 转移更符合标准 next POI 任务
        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, self.num_pois, only_adjacent=True)
        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, self.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.Deg_H_poi_src * self.H_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.Deg_H_poi_tar * self.H_poi_tar).to(device)

        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)
        self.pad_all_train_sessions = pad_sequence(
            self.all_train_sessions, batch_first=True, padding_value=self.padding_idx
        ).to(device)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prefix = sample["prefix_pois"]
        return {
            "user_idx": torch.tensor(sample["user_idx"]).to(self.device),
            "user_seq": torch.tensor(prefix).to(self.device),
            "user_rev_seq": torch.tensor(prefix[::-1]).to(self.device),
            "user_seq_len": torch.tensor(len(prefix)).to(self.device),
            "user_seq_mask": torch.tensor([1] * len(prefix)).to(self.device),
            "label": torch.tensor(sample["label_poi"]).to(self.device),
            "label_category": torch.tensor(sample["label_category"]).to(self.device),
            "label_region": torch.tensor(sample["label_region"]).to(self.device),
        }


def collate_fn(batch, padding_value):
    """将一个 batch 中不同长度的 prefix padding 到统一长度。"""
    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    batch_label_category = []
    batch_label_region = []

    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_label_category.append(item["label_category"])
        batch_label_region.append(item["label_region"])
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
        "label_category": torch.stack(batch_label_category),
        "label_region": torch.stack(batch_label_region),
    }

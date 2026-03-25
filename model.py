# coding=utf-8
"""
H-DCHL-B 模型定义。

设计思想：
1. 延续 DCHL 的多分支结构风格；
2. 将原本的地理分支替换为 Region 异构语义分支；
3. 新增 Category 异构语义分支；
4. 第一版仅比较结构增益，因此不包含对比学习与掩码训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroHyperConvLayer(nn.Module):
    """
    异构超图卷积层。

    该层用于处理“实体节点 + 语义节点（或用户节点）”组成的二部超图：
    - 节点到超边：先对 POI 聚合，再与超边所属实体节点融合
    - 超边到节点：将超边消息回传到 POI 侧
    """

    def __init__(self, emb_dim, device):
        super().__init__()
        self.poi_linear = nn.Linear(emb_dim, emb_dim, bias=False, device=device)
        self.edge_linear = nn.Linear(emb_dim, emb_dim, bias=False, device=device)
        self.fusion_linear = nn.Linear(2 * emb_dim, emb_dim, bias=False, device=device)

    def forward(self, poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge):
        # 1) 先将 POI 聚合到超边侧
        poi_msg = torch.sparse.mm(hg_poi_to_edge, self.poi_linear(poi_embs))

        # 2) 超边既接收 POI 聚合消息，也保留自身实体语义（user/region/category）
        edge_msg = self.edge_linear(edge_embs)
        fused_edge = self.fusion_linear(torch.cat([poi_msg, edge_msg], dim=1))

        # 3) 再将超边消息回传到 POI 侧
        propagated_poi = torch.sparse.mm(hg_edge_to_poi, fused_edge)
        return propagated_poi, fused_edge


class HeteroHyperConvNetwork(nn.Module):
    """堆叠多层异构超图卷积，并使用残差连接。"""

    def __init__(self, num_layers, emb_dim, dropout, device):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv = HeteroHyperConvLayer(emb_dim, device)

    def forward(self, poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge):
        final_poi_embs = [poi_embs]
        final_edge_embs = [edge_embs]
        for _ in range(self.num_layers):
            poi_embs, edge_embs = self.conv(poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge)
            poi_embs = F.dropout(poi_embs + final_poi_embs[-1], self.dropout)
            edge_embs = F.dropout(edge_embs + final_edge_embs[-1], self.dropout)
            final_poi_embs.append(poi_embs)
            final_edge_embs.append(edge_embs)
        poi_output = torch.mean(torch.stack(final_poi_embs), dim=0)
        edge_output = torch.mean(torch.stack(final_edge_embs), dim=0)
        return poi_output, edge_output


class DirectedHyperConvLayer(nn.Module):
    """保留 DCHL 原有风格的有向 POI 转移卷积层。"""

    def forward(self, poi_embs, hg_poi_src, hg_poi_tar):
        msg_tar = torch.sparse.mm(hg_poi_tar, poi_embs)
        msg_src = torch.sparse.mm(hg_poi_src, msg_tar)
        return msg_src


class DirectedHyperConvNetwork(nn.Module):
    """堆叠多层有向转移卷积。"""

    def __init__(self, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer = DirectedHyperConvLayer()

    def forward(self, poi_embs, hg_poi_src, hg_poi_tar):
        final_poi_embs = [poi_embs]
        for _ in range(self.num_layers):
            poi_embs = self.layer(poi_embs, hg_poi_src, hg_poi_tar)
            poi_embs = F.dropout(poi_embs + final_poi_embs[-1], self.dropout)
            final_poi_embs.append(poi_embs)
        return torch.mean(torch.stack(final_poi_embs), dim=0)


class HDCHLB(nn.Module):
    """
    H-DCHL-B 主模型。

    分支说明：
    - collaborative branch: User-POI 长期协同
    - transition branch: 有向 POI 转移
    - region branch: POI-Region 异构语义
    - category branch: POI-Category 异构语义
    """

    def __init__(self, num_users, num_pois, num_regions, num_categories, padding_idx, args, device):
        super().__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_regions = num_regions
        self.num_categories = num_categories
        self.emb_dim = args.emb_dim
        self.device = device

        # 四类节点的基础 embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=padding_idx)
        self.region_embedding = nn.Embedding(num_regions, self.emb_dim)
        self.category_embedding = nn.Embedding(num_categories, self.emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)

        # 节点自门控：延续 DCHL 的 disentangled 输入风格，但现在对应四个结构分支
        self.w_gate_col = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_trans = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_trans = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_reg = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_reg = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_cat = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_cat = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        for weight in [
            self.w_gate_col, self.b_gate_col, self.w_gate_trans, self.b_gate_trans,
            self.w_gate_reg, self.b_gate_reg, self.w_gate_cat, self.b_gate_cat,
        ]:
            nn.init.xavier_normal_(weight.data)

        # 四个结构分支
        self.col_network = HeteroHyperConvNetwork(args.num_col_layers, args.emb_dim, args.dropout, device)
        self.reg_network = HeteroHyperConvNetwork(args.num_reg_layers, args.emb_dim, args.dropout, device)
        self.cat_network = HeteroHyperConvNetwork(args.num_cat_layers, args.emb_dim, args.dropout, device)
        self.trans_network = DirectedHyperConvNetwork(args.num_trans_layers, args.dropout)

        # 用户侧自适应融合门，维持 DCHL 风格
        self.col_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.reg_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.cat_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

    def forward(self, dataset, batch):
        # 1) 为不同分支生成独立的输入门控 POI 表示
        base_poi_embs = self.poi_embedding.weight[:-1]
        col_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_col + self.b_gate_col))
        trans_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_trans + self.b_gate_trans))
        reg_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_reg + self.b_gate_reg))
        cat_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_cat + self.b_gate_cat))

        # 2) 协同分支：User-POI 异构超图
        col_edge_embs = self.user_embedding.weight
        col_poi_out, col_user_out = self.col_network(col_poi_embs, col_edge_embs, dataset.HG_pu, dataset.HG_up)

        # 3) 区域分支：POI-Region 异构超图
        reg_edge_embs = self.region_embedding.weight
        reg_poi_out, reg_region_out = self.reg_network(reg_poi_embs, reg_edge_embs, dataset.HG_pr, dataset.HG_rp)

        # 4) 类别分支：POI-Category 异构超图
        cat_edge_embs = self.category_embedding.weight
        cat_poi_out, cat_category_out = self.cat_network(cat_poi_embs, cat_edge_embs, dataset.HG_pc, dataset.HG_cp)

        # 5) 转移分支：保持原 DCHL 风格
        trans_poi_out = self.trans_network(trans_poi_embs, dataset.HG_poi_src, dataset.HG_poi_tar)

        # 6) 从不同 POI 分支回聚到 User，得到用户的多分支偏好表示
        reg_user_out = torch.sparse.mm(dataset.HG_up, reg_poi_out)
        cat_user_out = torch.sparse.mm(dataset.HG_up, cat_poi_out)
        trans_user_out = torch.sparse.mm(dataset.HG_up, trans_poi_out)

        batch_user_idx = batch["user_idx"]
        col_batch_user = col_user_out[batch_user_idx]
        reg_batch_user = reg_user_out[batch_user_idx]
        cat_batch_user = cat_user_out[batch_user_idx]
        trans_batch_user = trans_user_out[batch_user_idx]

        # 7) 归一化后做用户侧自适应融合
        norm_col_user = F.normalize(col_batch_user, p=2, dim=1)
        norm_reg_user = F.normalize(reg_batch_user, p=2, dim=1)
        norm_cat_user = F.normalize(cat_batch_user, p=2, dim=1)
        norm_trans_user = F.normalize(trans_batch_user, p=2, dim=1)

        col_coef = self.col_gate(norm_col_user)
        reg_coef = self.reg_gate(norm_reg_user)
        cat_coef = self.cat_gate(norm_cat_user)
        trans_coef = self.trans_gate(norm_trans_user)

        final_user_embs = (
            col_coef * norm_col_user + reg_coef * norm_reg_user + cat_coef * norm_cat_user + trans_coef * norm_trans_user
        )

        # 8) POI 侧先直接求和，后续如果需要可再扩展成更细的层次融合
        final_poi_embs = (
            F.normalize(col_poi_out, p=2, dim=1)
            + F.normalize(reg_poi_out, p=2, dim=1)
            + F.normalize(cat_poi_out, p=2, dim=1)
            + F.normalize(trans_poi_out, p=2, dim=1)
        )

        # 9) 计算下一 POI 的打分
        prediction = final_user_embs @ final_poi_embs.T

        # 第一版只做结构增益，不引入额外自监督项
        aux_loss = prediction.new_tensor(0.0)
        return prediction, aux_loss

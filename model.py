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
    """有向 POI 转移卷积层"""

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
        self.lambda_cat_cls = args.lambda_cat_cls
        self.lambda_reg_cls = args.lambda_reg_cls
        self.alpha_cat_prior = args.alpha_cat_prior
        self.beta_reg_prior = args.beta_reg_prior
        self.use_collaborative = args.use_collaborative
        self.use_category = args.use_category
        self.use_region = args.use_region

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

        # 方向A：Category/Region 意图预测头
        # 这里不再做 mask 重建，而是让 coarse-grained 意图直接进入最终决策链。
        self.category_predictor = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.ELU(),
            nn.Linear(args.emb_dim, num_categories),
        )
        self.region_predictor = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.ELU(),
            nn.Linear(args.emb_dim, num_regions),
        )
        self.category_criterion = nn.CrossEntropyLoss()
        self.region_criterion = nn.CrossEntropyLoss()

    @staticmethod
    def masked_mean_pooling(seq_embs, seq_mask):
        """
        对 prefix 序列做 masked mean pooling。

        seq_embs: [B, T, d]
        seq_mask: [B, T]
        """
        mask = seq_mask.unsqueeze(-1).float()
        summed = torch.sum(seq_embs * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / denom

    def forward(self, dataset, batch):
        # 1) 为不同分支生成独立的输入门控 POI 表示
        base_poi_embs = self.poi_embedding.weight[:-1]

        col_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_col + self.b_gate_col))
        trans_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_trans + self.b_gate_trans))
        reg_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_reg + self.b_gate_reg))
        cat_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_cat + self.b_gate_cat))

        # 2) 协同分支：User-POI 异构超图
        # 支持 w/o U-P 消融：当关闭协同分支时，不再使用 User-POI 图传播。
        if self.use_collaborative:
            col_edge_embs = self.user_embedding.weight
            col_poi_out, col_user_out = self.col_network(col_poi_embs, col_edge_embs, dataset.HG_pu, dataset.HG_up)
        else:
            col_poi_out = col_poi_embs
            col_user_out = self.user_embedding.weight

        # 3) 区域分支：POI-Region 异构超图
        if self.use_region:
            reg_edge_embs = self.region_embedding.weight
            reg_poi_out, reg_region_out = self.reg_network(reg_poi_embs, reg_edge_embs, dataset.HG_pr, dataset.HG_rp)
        else:
            reg_poi_out = reg_poi_embs
            reg_region_out = self.region_embedding.weight

        # 4) 类别分支：POI-Category 异构超图
        if self.use_category:
            cat_edge_embs = self.category_embedding.weight
            cat_poi_out, cat_category_out = self.cat_network(cat_poi_embs, cat_edge_embs, dataset.HG_pc, dataset.HG_cp)
        else:
            cat_poi_out = cat_poi_embs
            cat_category_out = self.category_embedding.weight

        # 5) 转移分支：保持原 DCHL 风格
        if dataset.HG_poi_src is not None and dataset.HG_poi_tar is not None:
            trans_poi_out = self.trans_network(trans_poi_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
            use_transition = True
        else:
            trans_poi_out = None
            use_transition = False

        # 6) 标准 next POI 版本：
        # 不再只依赖 user_idx 对应的静态用户表示，而是结合当前 prefix 序列生成样本级表示。
        batch_user_idx = batch["user_idx"]
        batch_prefix = batch["user_seq"]
        batch_prefix_mask = batch["user_seq_mask"]

        # prefix 经过 padding 后会包含 padding_idx，而四个分支输出只覆盖真实 POI [0, num_pois-1]。
        # 因此这里为每个分支补一行全零 padding 向量，避免索引越界并保证 pooling 正确忽略 padding 位。
        zero_pad = torch.zeros(1, self.emb_dim, device=self.device)
        col_poi_out_with_pad = torch.cat([col_poi_out, zero_pad], dim=0)
        reg_poi_out_with_pad = torch.cat([reg_poi_out, zero_pad], dim=0)
        cat_poi_out_with_pad = torch.cat([cat_poi_out, zero_pad], dim=0)
        if use_transition:
            trans_poi_out_with_pad = torch.cat([trans_poi_out, zero_pad], dim=0)

        col_prefix_embs = col_poi_out_with_pad[batch_prefix]
        reg_prefix_embs = reg_poi_out_with_pad[batch_prefix]
        cat_prefix_embs = cat_poi_out_with_pad[batch_prefix]
        if use_transition:
            trans_prefix_embs = trans_poi_out_with_pad[batch_prefix]

        col_prefix_user = self.masked_mean_pooling(col_prefix_embs, batch_prefix_mask)
        reg_prefix_user = self.masked_mean_pooling(reg_prefix_embs, batch_prefix_mask)
        cat_prefix_user = self.masked_mean_pooling(cat_prefix_embs, batch_prefix_mask)
        if use_transition:
            trans_prefix_user = self.masked_mean_pooling(trans_prefix_embs, batch_prefix_mask)

        # 协同分支额外保留用户静态节点信息，其余分支主要依赖 prefix 序列语义。
        if self.use_collaborative:
            col_batch_user = col_user_out[batch_user_idx] + col_prefix_user
        else:
            col_batch_user = None
        reg_batch_user = reg_prefix_user
        cat_batch_user = cat_prefix_user
        trans_batch_user = trans_prefix_user if use_transition else None

        # 7) 归一化后做用户侧自适应融合
        if self.use_collaborative:
            norm_col_user = F.normalize(col_batch_user, p=2, dim=1)
        norm_reg_user = F.normalize(reg_batch_user, p=2, dim=1)
        norm_cat_user = F.normalize(cat_batch_user, p=2, dim=1)
        if use_transition:
            norm_trans_user = F.normalize(trans_batch_user, p=2, dim=1)

        if self.use_collaborative:
            col_coef = self.col_gate(norm_col_user)
        reg_coef = self.reg_gate(norm_reg_user)
        cat_coef = self.cat_gate(norm_cat_user)
        if use_transition:
            trans_coef = self.trans_gate(norm_trans_user)

        if not self.use_region:
            reg_coef = torch.zeros_like(reg_coef)
        if not self.use_category:
            cat_coef = torch.zeros_like(cat_coef)

        if self.use_collaborative and use_transition:
            final_user_embs = (
                col_coef * norm_col_user + reg_coef * norm_reg_user + cat_coef * norm_cat_user + trans_coef * norm_trans_user
            )
        elif self.use_collaborative and not use_transition:
            final_user_embs = col_coef * norm_col_user + reg_coef * norm_reg_user + cat_coef * norm_cat_user
        elif (not self.use_collaborative) and use_transition:
            final_user_embs = reg_coef * norm_reg_user + cat_coef * norm_cat_user + trans_coef * norm_trans_user
        else:
            final_user_embs = reg_coef * norm_reg_user + cat_coef * norm_cat_user

        # 8) POI 侧基础融合：先得到原始 POI 打分主干。
        norm_col_poi = F.normalize(col_poi_out, p=2, dim=1)
        norm_reg_poi = F.normalize(reg_poi_out, p=2, dim=1)
        norm_cat_poi = F.normalize(cat_poi_out, p=2, dim=1)
        poi_parts = []
        if self.use_collaborative:
            poi_parts.append(norm_col_poi)
        poi_parts.append(norm_reg_poi)
        poi_parts.append(norm_cat_poi)
        if use_transition:
            norm_trans_poi = F.normalize(trans_poi_out, p=2, dim=1)
            poi_parts.append(norm_trans_poi)
        final_poi_embs = poi_parts[0]
        for poi_part in poi_parts[1:]:
            final_poi_embs = final_poi_embs + poi_part
        base_prediction = final_user_embs @ final_poi_embs.T

        # 方向A：先预测 coarse-grained intent（category / region），再将其作为 soft prior 注入 POI 评分。
        category_logits = self.category_predictor(final_user_embs)  # [B, C]
        region_logits = self.region_predictor(final_user_embs)      # [B, R]

        # 将 category / region 的意图分数映射回每个候选 POI。
        poi_cat_idx = dataset.poi_category_tensor.to(self.device)
        poi_reg_idx = dataset.poi_region_tensor.to(self.device)
        cat_prior_scores = category_logits[:, poi_cat_idx] if self.use_category else torch.zeros_like(base_prediction)
        reg_prior_scores = region_logits[:, poi_reg_idx] if self.use_region else torch.zeros_like(base_prediction)

        prediction = base_prediction + self.alpha_cat_prior * cat_prior_scores + self.beta_reg_prior * reg_prior_scores

        aux_loss = prediction.new_tensor(0.0)
        if self.training:
            if self.use_category and self.lambda_cat_cls > 0:
                aux_loss = aux_loss + self.lambda_cat_cls * self.category_criterion(category_logits, batch["label_category"])
            if self.use_region and self.lambda_reg_cls > 0:
                aux_loss = aux_loss + self.lambda_reg_cls * self.region_criterion(region_logits, batch["label_region"])

        return prediction, aux_loss

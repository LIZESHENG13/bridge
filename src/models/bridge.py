# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_frequency_encoder import DualFrequencyEncoder


class BRIDGE(DualFrequencyEncoder):
    """Behavior-guided candidate calibration model.

    The clean BRIDGE configuration keeps the dual-frequency multimodal encoder
    and uses co-user behavior evidence to calibrate only the base top-K
    candidates. Behavior prototype injection and private residual correction
    remain available as optional ablation switches, but are disabled by default.
    """

    def __init__(self, config, dataset):
        super(BRIDGE, self).__init__(config, dataset)

        self.use_behavior_proto = self._as_bool(self._cfg(config, "use_behavior_proto", False))
        self.use_behavior_correction = self._as_bool(self._cfg(config, "use_behavior_correction", True))
        self.use_private_residual = self._as_bool(self._cfg(config, "use_private_residual", False))
        self.use_topk_correction = self._as_bool(self._cfg(config, "use_topk_correction", True))
        self.use_behavior_gate = self._as_bool(self._cfg(config, "use_behavior_gate", True))
        self.use_calibration_gate = self._as_bool(self._cfg(config, "use_calibration_gate", True))
        self.use_residual_gate = self._as_bool(self._cfg(config, "use_residual_gate", False))

        self.behavior_topk = int(self._cfg(config, "behavior_topk", 1000))
        self.behavior_weight = float(self._cfg(config, "behavior_weight", 0.4))
        self.behavior_eval_topk = int(self._cfg(config, "behavior_eval_topk", 200))
        self.correction_scope = str(self._cfg(config, "correction_scope", "topk")).lower()
        self.train_candidate_aware = self._as_bool(self._cfg(config, "train_candidate_aware", True))
        self.train_candidate_topk = int(self._cfg(config, "train_candidate_topk", self.behavior_eval_topk))
        self.behavior_sim_method = str(self._cfg(config, "behavior_sim_method", "cosine")).lower()
        self.behavior_aggregation = str(self._cfg(config, "behavior_aggregation", "sum_sqrt")).lower()
        self.behavior_score_norm = str(self._cfg(config, "behavior_score_norm", "z")).lower()

        self.private_residual_weight = float(self._cfg(config, "private_residual_weight", 0.0))
        self.private_residual_transform = str(self._cfg(config, "private_residual_transform", "raw")).lower()
        self.low_band_count = int(self._cfg(config, "low_band_count", 1))

        self.aux_base_weight = float(self._cfg(config, "aux_base_weight", 0.2))
        self.aux_low_weight = float(self._cfg(config, "aux_low_weight", 0.0))
        self.aux_high_weight = float(self._cfg(config, "aux_high_weight", 0.0))
        self.gate_reg_weight = float(self._cfg(config, "gate_reg_weight", 0.001))
        self.gate_target = float(self._cfg(config, "gate_target", 0.5))
        self.gate_floor = float(self._cfg(config, "gate_floor", 0.0))
        self.gate_behavior_init = float(self._cfg(config, "gate_behavior_init", 1.0))
        self.behavior_proto_weight = float(self._cfg(config, "behavior_proto_weight", 0.0))
        self.behavior_proto_loss_weight = float(self._cfg(config, "behavior_proto_loss_weight", 0.0))
        self.behavior_proto_gate_reg_weight = float(self._cfg(config, "behavior_proto_gate_reg_weight", 0.0))
        self.behavior_proto_gate_target = float(self._cfg(config, "behavior_proto_gate_target", 0.5))

        self.behavior_user_bias = nn.Embedding(self.n_users, 1)
        self.behavior_item_bias = nn.Embedding(self.n_items, 1)
        nn.init.zeros_(self.behavior_user_bias.weight)
        nn.init.zeros_(self.behavior_item_bias.weight)
        self.gate_base_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gate_behavior_scale = nn.Parameter(torch.tensor(self.gate_behavior_init, dtype=torch.float32))
        self.gate_residual_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.final_embed_dim = self.dim_latent * 3
        if self.use_behavior_proto and (
            self.behavior_proto_weight != 0.0 or self.behavior_proto_loss_weight != 0.0
        ):
            self.behavior_proto_adapter = nn.Linear(self.final_embed_dim, self.final_embed_dim, bias=False)
            self.behavior_proto_gate = nn.Sequential(
                nn.Linear(self.final_embed_dim * 2, self.final_embed_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.final_embed_dim // 2, 1),
            )
            nn.init.zeros_(self.behavior_proto_gate[-1].weight)
            nn.init.zeros_(self.behavior_proto_gate[-1].bias)
        else:
            self.use_behavior_proto = False
            self.behavior_proto_adapter = None
            self.behavior_proto_gate = None

        dense_entries = self.n_users * self.n_items
        dense_threshold = int(self._cfg(config, "behavior_dense_threshold", 1_000_000_000))
        if dense_entries <= dense_threshold:
            behavior_scores, item_pop, user_degree = self._build_behavior_score_matrix(dataset)
            self.behavior_score_matrix = torch.from_numpy(behavior_scores).to(self.device)
            self.behavior_train_csr = None
            self.behavior_item_sim = None
            behavior_backend = "dense"
        else:
            behavior_item_sim, behavior_train_csr, item_pop, user_degree = self._build_sparse_behavior_evidence(dataset)
            self.behavior_score_matrix = None
            self.behavior_train_csr = behavior_train_csr
            self.behavior_item_sim = behavior_item_sim
            behavior_backend = "sparse"
        self.item_pop_feature = torch.from_numpy(item_pop).to(self.device)
        self.user_degree_feature = torch.from_numpy(user_degree).to(self.device)
        self.history_adj = self._build_history_adj(dataset)
        print(
            "BRIDGE behavior branch: "
            f"BPI={self.use_behavior_proto}, BC={self.use_behavior_correction}, "
            f"FRC={self.use_private_residual}, TCR={self.use_topk_correction}, "
            f"Gate={self.use_calibration_gate}, BGate={self.use_behavior_gate}, RGate={self.use_residual_gate}, "
            f"topk={self.behavior_topk}, weight={self.behavior_weight}, "
            f"eval_topk={self.behavior_eval_topk}, train_candidate_aware={self.train_candidate_aware}, "
            f"train_topk={self.train_candidate_topk}, scope={self.correction_scope}, norm={self.behavior_score_norm}, "
            f"proto_weight={self.behavior_proto_weight}, backend={behavior_backend}"
        )

    @staticmethod
    def _cfg(config, key, default):
        value = config[key] if key in config else default
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    @staticmethod
    def _as_bool(value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def _build_behavior_score_matrix(self, dataset):
        train_mat = dataset.inter_matrix(form="csr").astype(np.float32)
        n_users, n_items = train_mat.shape

        user_degree = np.asarray(train_mat.sum(axis=1)).reshape(-1).astype(np.float32)
        item_pop = np.asarray(train_mat.sum(axis=0)).reshape(-1).astype(np.float32)

        cooc = (train_mat.T @ train_mat).astype(np.float32)
        cooc.setdiag(0.0)
        cooc.eliminate_zeros()
        sim = cooc.toarray().astype(np.float32, copy=False)

        if self.behavior_sim_method == "cosine":
            denom = np.sqrt(np.maximum(item_pop, 1e-6))
            sim = sim / denom[:, None] / denom[None, :]
        elif self.behavior_sim_method == "jaccard":
            union = item_pop[:, None] + item_pop[None, :] - sim
            sim = np.divide(sim, union, out=np.zeros_like(sim, dtype=np.float32), where=union > 0)
        elif self.behavior_sim_method == "logcooc":
            sim = np.log1p(sim)
        elif self.behavior_sim_method not in ("cooc", "raw"):
            raise ValueError(f"Unknown behavior_sim_method: {self.behavior_sim_method}")

        np.fill_diagonal(sim, 0.0)
        if 0 < self.behavior_topk < n_items:
            top_idx = np.argpartition(sim, -self.behavior_topk, axis=1)[:, -self.behavior_topk:]
            top_val = np.take_along_axis(sim, top_idx, axis=1)
            pruned = np.zeros_like(sim, dtype=np.float32)
            np.put_along_axis(pruned, top_idx, top_val, axis=1)
            sim = pruned

        scores = train_mat.dot(sim.T)
        if not isinstance(scores, np.ndarray):
            scores = scores.toarray()
        scores = scores.astype(np.float32, copy=False)

        if self.behavior_aggregation == "mean":
            denom = np.maximum(user_degree, 1.0).reshape(-1, 1)
            scores = scores / denom
        elif self.behavior_aggregation == "sum_sqrt":
            denom = np.sqrt(np.maximum(user_degree, 1.0)).reshape(-1, 1)
            scores = scores / denom
        elif self.behavior_aggregation != "sum":
            raise ValueError(f"Unknown behavior_aggregation: {self.behavior_aggregation}")

        if self.behavior_score_norm == "z":
            mean = scores.mean(axis=1, keepdims=True)
            std = scores.std(axis=1, keepdims=True)
            scores = (scores - mean) / np.maximum(std, 1e-6)
        elif self.behavior_score_norm == "minmax":
            lo = scores.min(axis=1, keepdims=True)
            hi = scores.max(axis=1, keepdims=True)
            scores = (scores - lo) / np.maximum(hi - lo, 1e-6)
        elif self.behavior_score_norm != "none":
            raise ValueError(f"Unknown behavior_score_norm: {self.behavior_score_norm}")

        user_degree = np.log1p(user_degree)
        user_degree = (user_degree - user_degree.mean()) / max(user_degree.std(), 1e-6)
        item_pop = np.log1p(item_pop)
        item_pop = (item_pop - item_pop.mean()) / max(item_pop.std(), 1e-6)
        return scores.astype(np.float32, copy=False), item_pop.astype(np.float32), user_degree.astype(np.float32)

    @staticmethod
    def _prune_csr_topk(matrix, topk):
        matrix = matrix.tocsr()
        topk = int(topk)
        if topk <= 0:
            return matrix

        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
        new_indptr = [0]
        new_indices = []
        new_data = []
        for row in range(matrix.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            row_indices = indices[start:end]
            row_data = data[start:end]
            if row_data.size > topk:
                keep = np.argpartition(row_data, -topk)[-topk:]
                order = np.argsort(row_indices[keep])
                keep = keep[order]
                row_indices = row_indices[keep]
                row_data = row_data[keep]
            new_indices.extend(row_indices.tolist())
            new_data.extend(row_data.tolist())
            new_indptr.append(len(new_indices))

        pruned = sp.csr_matrix(
            (
                np.asarray(new_data, dtype=np.float32),
                np.asarray(new_indices, dtype=np.int32),
                np.asarray(new_indptr, dtype=np.int64),
            ),
            shape=matrix.shape,
            dtype=np.float32,
        )
        pruned.eliminate_zeros()
        return pruned

    def _build_sparse_behavior_evidence(self, dataset):
        train_mat = dataset.inter_matrix(form="csr").astype(np.float32)
        n_users, n_items = train_mat.shape

        user_degree = np.asarray(train_mat.sum(axis=1)).reshape(-1).astype(np.float32)
        item_pop_raw = np.asarray(train_mat.sum(axis=0)).reshape(-1).astype(np.float32)

        cooc = (train_mat.T @ train_mat).astype(np.float32).tocsr()
        cooc.setdiag(0.0)
        cooc.eliminate_zeros()

        if self.behavior_sim_method == "cosine":
            denom = np.sqrt(np.maximum(item_pop_raw, 1e-6)).astype(np.float32)
            inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom > 0)
            sim = cooc.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
        elif self.behavior_sim_method == "jaccard":
            coo = cooc.tocoo()
            union = item_pop_raw[coo.row] + item_pop_raw[coo.col] - coo.data
            data = np.divide(
                coo.data,
                union,
                out=np.zeros_like(coo.data, dtype=np.float32),
                where=union > 0,
            )
            sim = sp.csr_matrix((data, (coo.row, coo.col)), shape=(n_items, n_items), dtype=np.float32)
        elif self.behavior_sim_method == "logcooc":
            sim = cooc.copy()
            sim.data = np.log1p(sim.data)
        elif self.behavior_sim_method in ("cooc", "raw"):
            sim = cooc
        else:
            raise ValueError(f"Unknown behavior_sim_method: {self.behavior_sim_method}")

        sim = self._prune_csr_topk(sim, self.behavior_topk)
        sim.sort_indices()

        user_degree_norm = np.log1p(user_degree)
        user_degree_norm = (user_degree_norm - user_degree_norm.mean()) / max(user_degree_norm.std(), 1e-6)
        item_pop = np.log1p(item_pop_raw)
        item_pop = (item_pop - item_pop.mean()) / max(item_pop.std(), 1e-6)
        return sim, train_mat, item_pop.astype(np.float32), user_degree_norm.astype(np.float32)

    def _pair_behavior_scores(self, users, items):
        if self.behavior_score_matrix is not None:
            return self.behavior_score_matrix[users, items]

        device = users.device
        users_np = users.detach().cpu().numpy().astype(np.int64, copy=False)
        items_np = items.detach().cpu().numpy().astype(np.int64, copy=False)
        item_rows = self.behavior_item_sim[items_np]
        history_rows = self.behavior_train_csr[users_np]
        scores = np.asarray(item_rows.multiply(history_rows).sum(axis=1)).reshape(-1).astype(np.float32)
        if self.behavior_aggregation in ("mean", "sum_sqrt"):
            hist_len = np.diff(history_rows.indptr).astype(np.float32)
            hist_len = np.maximum(hist_len, 1.0)
            if self.behavior_aggregation == "mean":
                scores = scores / hist_len
            else:
                scores = scores / np.sqrt(hist_len)
        elif self.behavior_aggregation != "sum":
            raise ValueError(f"Unknown behavior_aggregation: {self.behavior_aggregation}")
        return torch.from_numpy(scores).to(device)

    @staticmethod
    def _gather_csr_rows(matrix, col_idx):
        matrix = matrix.tocsr()
        matrix.sort_indices()
        out = np.zeros(col_idx.shape, dtype=np.float32)
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data
        for row in range(col_idx.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue
            cols = indices[start:end]
            vals = data[start:end]
            targets = col_idx[row]
            pos = np.searchsorted(cols, targets)
            valid = (pos < cols.size) & (cols[pos.clip(max=max(cols.size - 1, 0))] == targets)
            if np.any(valid):
                out[row, valid] = vals[pos[valid]]
        return out

    def _candidate_behavior_scores(self, users, candidate_idx):
        if self.behavior_score_matrix is not None:
            return self.behavior_score_matrix[users].gather(1, candidate_idx)

        device = candidate_idx.device
        users_np = users.detach().cpu().numpy().astype(np.int64, copy=False)
        candidate_np = candidate_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        history_rows = self.behavior_train_csr[users_np]
        score_rows = history_rows.dot(self.behavior_item_sim.T).tocsr()
        if self.behavior_aggregation in ("mean", "sum_sqrt"):
            hist_len = np.diff(history_rows.indptr).astype(np.float32)
            hist_len = np.maximum(hist_len, 1.0)
            scale = 1.0 / hist_len if self.behavior_aggregation == "mean" else 1.0 / np.sqrt(hist_len)
            score_rows = sp.diags(scale).dot(score_rows).tocsr()
        elif self.behavior_aggregation != "sum":
            raise ValueError(f"Unknown behavior_aggregation: {self.behavior_aggregation}")

        scores = self._gather_csr_rows(score_rows, candidate_np)
        scores = torch.from_numpy(scores).to(device)
        if self.behavior_score_norm == "z":
            scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
        elif self.behavior_score_norm == "minmax":
            lo = scores.min(dim=1, keepdim=True).values
            hi = scores.max(dim=1, keepdim=True).values
            scores = (scores - lo) / (hi - lo).clamp_min(1e-6)
        elif self.behavior_score_norm != "none":
            raise ValueError(f"Unknown behavior_score_norm: {self.behavior_score_norm}")
        return scores

    def _build_history_adj(self, dataset):
        train_mat = dataset.inter_matrix(form="coo").astype(np.float32)
        rows = train_mat.row.astype(np.int64)
        cols = train_mat.col.astype(np.int64)
        user_degree = np.bincount(rows, minlength=self.n_users).astype(np.float32)
        if self.behavior_aggregation == "mean":
            values = 1.0 / np.maximum(user_degree[rows], 1.0)
        elif self.behavior_aggregation == "sum_sqrt":
            values = 1.0 / np.sqrt(np.maximum(user_degree[rows], 1.0))
        else:
            values = np.ones_like(rows, dtype=np.float32)
        indices = torch.from_numpy(np.stack([rows, cols], axis=0)).long().to(self.device)
        values = torch.from_numpy(values.astype(np.float32, copy=False)).to(self.device)
        return torch.sparse_coo_tensor(
            indices,
            values,
            (self.n_users, self.n_items),
            device=self.device,
        ).coalesce()

    def _apply_behavior_proto(self, user_emb_all, item_emb_all):
        if not self.use_behavior_proto or self.behavior_proto_weight == 0.0:
            proto = torch.zeros_like(user_emb_all)
            gate = torch.zeros((user_emb_all.size(0), 1), device=user_emb_all.device, dtype=user_emb_all.dtype)
            return user_emb_all, proto, gate

        proto = torch.sparse.mm(self.history_adj, item_emb_all)
        proto = self.behavior_proto_adapter(proto)
        proto_norm = F.normalize(proto, dim=-1)
        user_norm = user_emb_all.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proto = proto_norm * user_norm
        gate = torch.sigmoid(self.behavior_proto_gate(torch.cat([user_emb_all, proto], dim=-1)))
        enhanced_user = user_emb_all + self.behavior_proto_weight * gate * proto
        return enhanced_user, proto, gate

    def _frequency_views(self, all_user, all_item, user_frequencies, item_frequencies):
        low_count = min(max(self.low_band_count, 1), len(user_frequencies))
        low_user = torch.stack(user_frequencies[:low_count], dim=0).sum(dim=0)
        low_item = torch.stack(item_frequencies[:low_count], dim=0).sum(dim=0)

        if low_count < len(user_frequencies):
            high_user = torch.stack(user_frequencies[low_count:], dim=0).sum(dim=0)
            high_item = torch.stack(item_frequencies[low_count:], dim=0).sum(dim=0)
        else:
            high_user = all_user - low_user
            high_item = all_item - low_item
        return low_user, low_item, high_user, high_item

    def _transform_residual(self, residual):
        if self.private_residual_transform == "raw":
            return residual
        if self.private_residual_transform == "neg":
            return torch.clamp(residual, max=0.0)
        if self.private_residual_transform == "pos":
            return torch.clamp(residual, min=0.0)
        if self.private_residual_transform == "tanh":
            return torch.tanh(residual)
        raise ValueError(f"Unknown private_residual_transform: {self.private_residual_transform}")

    def _pair_gate(self, users, items, base_scores, behavior_scores, residual_scores):
        if not self.use_calibration_gate:
            return torch.ones_like(base_scores)

        user_bias = self.behavior_user_bias(users).view(-1, 1)
        if items.dim() == 1:
            item_bias = self.behavior_item_bias(items).view(-1)
            user_bias = user_bias.view(-1)
        else:
            item_bias = self.behavior_item_bias(items).squeeze(-1)

        logits = user_bias + item_bias
        logits = logits + self.gate_base_scale * torch.tanh(base_scores.detach())
        if self.use_behavior_gate:
            logits = logits + self.gate_behavior_scale * torch.tanh(behavior_scores.detach())
        if self.use_residual_gate:
            logits = logits + self.gate_residual_scale * torch.tanh(residual_scores.detach())
        gate = torch.sigmoid(logits)
        if self.gate_floor > 0:
            gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        return gate

    def _combine_score_components(self, users, items, base_scores, behavior_scores, residual_scores, correction_mask=None):
        """Combine the clean score with optional ablation components.

        base_scores: encoder score after optional behavior prototype injection.
        behavior_scores: co-user ItemKNN evidence.
        residual_scores: optional high-frequency residual ablation signal.
        """
        gate = self._pair_gate(users, items, base_scores, behavior_scores, residual_scores)
        behavior_weight = self.behavior_weight if self.use_behavior_correction else 0.0
        residual_weight = self.private_residual_weight if self.use_private_residual else 0.0
        behavior_correction = behavior_weight * gate * behavior_scores
        residual_correction = residual_weight * gate * residual_scores
        if correction_mask is not None:
            correction_mask = correction_mask.to(base_scores.device, dtype=base_scores.dtype)
            behavior_correction = behavior_correction * correction_mask
            residual_correction = residual_correction * correction_mask
        final_scores = base_scores + behavior_correction + residual_correction
        return final_scores, gate, behavior_correction, residual_correction

    def _sampled_candidate_masks(self, user_emb_batch, item_emb_all, pos_items, neg_items):
        """Return whether sampled items are inside the current base top-K set.

        Inference applies calibration only inside the base top-K candidates.  This
        mask applies the same scope during training while keeping the top-K
        selection detached from the encoder gradients.
        """
        if (
            not self.train_candidate_aware
            or not self.use_topk_correction
            or self.behavior_eval_topk <= 0
            or self.correction_scope != "topk"
        ):
            ones = torch.ones_like(pos_items, dtype=user_emb_batch.dtype, device=user_emb_batch.device)
            return ones, ones

        candidate_k = min(max(self.train_candidate_topk, 50), self.n_items)
        with torch.no_grad():
            base_scores = torch.matmul(user_emb_batch.detach(), item_emb_all.detach().transpose(0, 1))
            _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
            pos_mask = (candidate_idx == pos_items.view(-1, 1)).any(dim=1)
            neg_mask = (candidate_idx == neg_items.view(-1, 1)).any(dim=1)
        return pos_mask.to(user_emb_batch.dtype), neg_mask.to(user_emb_batch.dtype)

    @staticmethod
    def _bpr_from_scores(pos_scores, neg_scores):
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def calculate_loss(self, interaction):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        raw_user_emb_all, item_emb_all, ib_loss, user_frequencies, item_frequencies = DualFrequencyEncoder.forward(self)
        user_emb_all, proto_user_emb_all, proto_gate = self._apply_behavior_proto(raw_user_emb_all, item_emb_all)
        low_user, low_item, high_user, high_item = self._frequency_views(
            raw_user_emb_all, item_emb_all, user_frequencies, item_frequencies
        )

        user_emb_batch = user_emb_all[user_indices]
        pos_item_emb_batch = item_emb_all[pos_item_indices]
        neg_item_emb_batch = item_emb_all[neg_item_indices]

        pos_base = torch.sum(user_emb_batch * pos_item_emb_batch, dim=1)
        neg_base = torch.sum(user_emb_batch * neg_item_emb_batch, dim=1)

        low_user_batch = low_user[user_indices]
        high_user_batch = high_user[user_indices]
        pos_low = torch.sum(low_user_batch * low_item[pos_item_indices], dim=1)
        neg_low = torch.sum(low_user_batch * low_item[neg_item_indices], dim=1)
        pos_high = torch.sum(high_user_batch * high_item[pos_item_indices], dim=1)
        neg_high = torch.sum(high_user_batch * high_item[neg_item_indices], dim=1)

        pos_residual = self._transform_residual(pos_high - pos_low)
        neg_residual = self._transform_residual(neg_high - neg_low)
        pos_behavior = self._pair_behavior_scores(user_indices, pos_item_indices)
        neg_behavior = self._pair_behavior_scores(user_indices, neg_item_indices)
        if self.behavior_score_matrix is None and self.behavior_score_norm == "z":
            behavior_pair = torch.cat([pos_behavior, neg_behavior], dim=0)
            behavior_pair = (behavior_pair - behavior_pair.mean()) / behavior_pair.std(unbiased=False).clamp_min(1e-6)
            pos_behavior, neg_behavior = torch.chunk(behavior_pair, 2, dim=0)
        elif self.behavior_score_matrix is None and self.behavior_score_norm == "minmax":
            behavior_pair = torch.cat([pos_behavior, neg_behavior], dim=0)
            lo = behavior_pair.min()
            hi = behavior_pair.max()
            behavior_pair = (behavior_pair - lo) / (hi - lo).clamp_min(1e-6)
            pos_behavior, neg_behavior = torch.chunk(behavior_pair, 2, dim=0)

        pos_candidate_mask, neg_candidate_mask = self._sampled_candidate_masks(
            user_emb_batch, item_emb_all, pos_item_indices, neg_item_indices
        )
        pos_final, pos_gate, _, _ = self._combine_score_components(
            user_indices, pos_item_indices, pos_base, pos_behavior, pos_residual, pos_candidate_mask
        )
        neg_final, neg_gate, _, _ = self._combine_score_components(
            user_indices, neg_item_indices, neg_base, neg_behavior, neg_residual, neg_candidate_mask
        )

        main_loss = self._bpr_from_scores(pos_final, neg_final)
        base_loss = self._bpr_from_scores(pos_base, neg_base)
        low_loss = self._bpr_from_scores(pos_low, neg_low)
        high_loss = self._bpr_from_scores(pos_high, neg_high)
        zero_loss = pos_base.new_tensor(0.0)
        if self.use_behavior_proto and self.behavior_proto_loss_weight != 0.0:
            proto_user_batch = proto_user_emb_all[user_indices]
            pos_proto = torch.sum(proto_user_batch * pos_item_emb_batch, dim=1)
            neg_proto = torch.sum(proto_user_batch * neg_item_emb_batch, dim=1)
            proto_loss = self._bpr_from_scores(pos_proto, neg_proto)
        else:
            proto_loss = zero_loss

        user_band_emb_batch = [band[user_indices] for band in user_frequencies]
        item_band_emb_batch = [band[pos_item_indices] for band in item_frequencies]
        cl_loss = self.frequency_contrastive_loss(user_band_emb_batch, item_band_emb_batch)

        active_gate_mask = torch.cat([pos_candidate_mask, neg_candidate_mask], dim=0) > 0
        gate_values = torch.cat([pos_gate, neg_gate], dim=0)
        if active_gate_mask.any():
            gate_mean = gate_values[active_gate_mask].mean()
            gate_reg = (gate_mean - self.gate_target) ** 2
        else:
            gate_reg = zero_loss
        if self.use_behavior_proto and self.behavior_proto_gate_reg_weight != 0.0:
            proto_gate_reg = (proto_gate.mean() - self.behavior_proto_gate_target) ** 2
        else:
            proto_gate_reg = zero_loss

        return (
            main_loss
            + ib_loss * self.ib_weight
            + cl_loss * self.reg_weight
            + self.aux_base_weight * base_loss
            + self.aux_low_weight * low_loss
            + self.aux_high_weight * high_loss
            + self.gate_reg_weight * gate_reg
            + self.behavior_proto_loss_weight * proto_loss
            + self.behavior_proto_gate_reg_weight * proto_gate_reg
        )

    def full_sort_predict(self, interaction):
        users = interaction[0]
        raw_user_emb_all, item_emb_all, _, user_frequencies, item_frequencies = DualFrequencyEncoder.forward(self)
        user_emb_all, _, _ = self._apply_behavior_proto(raw_user_emb_all, item_emb_all)
        low_user, low_item, high_user, high_item = self._frequency_views(
            raw_user_emb_all, item_emb_all, user_frequencies, item_frequencies
        )

        user_emb = user_emb_all[users]
        base_scores = torch.matmul(user_emb, item_emb_all.transpose(0, 1))
        if (
            not self.use_topk_correction
            or self.behavior_eval_topk <= 0
            or (
                (not self.use_behavior_correction or self.behavior_weight == 0.0)
                and (not self.use_private_residual or self.private_residual_weight == 0.0)
            )
        ):
            return base_scores

        if self.correction_scope == "global":
            all_items = torch.arange(self.n_items, device=self.device).unsqueeze(0).expand(users.size(0), -1)
            behavior_scores = self._candidate_behavior_scores(users, all_items)
            high_scores = torch.matmul(high_user[users], high_item.transpose(0, 1))
            low_scores = torch.matmul(low_user[users], low_item.transpose(0, 1))
            residual_scores = self._transform_residual(high_scores - low_scores)
            _, _, behavior_correction, residual_correction = self._combine_score_components(
                users, all_items, base_scores, behavior_scores, residual_scores
            )
            return base_scores + behavior_correction + residual_correction

        if self.correction_scope != "topk":
            raise ValueError(f"Unknown correction_scope: {self.correction_scope}")

        candidate_k = min(max(self.behavior_eval_topk, 50), self.n_items)
        _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
        candidate_base = base_scores.gather(1, candidate_idx)
        candidate_behavior = self._candidate_behavior_scores(users, candidate_idx)

        high_scores = torch.matmul(high_user[users], high_item.transpose(0, 1)).gather(1, candidate_idx)
        low_scores = torch.matmul(low_user[users], low_item.transpose(0, 1)).gather(1, candidate_idx)
        candidate_residual = self._transform_residual(high_scores - low_scores)
        _, _, behavior_correction, residual_correction = self._combine_score_components(
            users, candidate_idx, candidate_base, candidate_behavior, candidate_residual
        )
        correction = behavior_correction + residual_correction

        final_scores = base_scores.clone()
        final_scores.scatter_add_(1, candidate_idx, correction)
        return final_scores

    @torch.no_grad()
    def export_diagnostics(self, max_users=256):
        was_training = self.training
        self.eval()

        raw_user, item_emb, _, user_frequencies, item_frequencies = DualFrequencyEncoder.forward(self)
        user_emb, proto_emb, proto_gate = self._apply_behavior_proto(raw_user, item_emb)
        low_user, low_item, high_user, high_item = self._frequency_views(
            raw_user, item_emb, user_frequencies, item_frequencies
        )

        n_diag_users = min(max_users, self.n_users)
        users = torch.arange(n_diag_users, device=self.device)
        base_scores = torch.matmul(user_emb[users], item_emb.transpose(0, 1))
        candidate_k = min(max(self.behavior_eval_topk, 50), self.n_items)
        _, candidate_idx = torch.topk(base_scores, k=candidate_k, dim=1)
        candidate_base = base_scores.gather(1, candidate_idx)
        candidate_behavior = self._candidate_behavior_scores(users, candidate_idx)
        high_scores = torch.matmul(high_user[users], high_item.transpose(0, 1)).gather(1, candidate_idx)
        low_scores = torch.matmul(low_user[users], low_item.transpose(0, 1)).gather(1, candidate_idx)
        candidate_residual = self._transform_residual(high_scores - low_scores)
        _, candidate_gate, behavior_correction, residual_correction = self._combine_score_components(
            users, candidate_idx, candidate_base, candidate_behavior, candidate_residual
        )
        correction = behavior_correction + residual_correction

        low_high_cos = F.cosine_similarity(low_item, high_item, dim=-1)
        proto_user_cos = F.cosine_similarity(raw_user, user_emb, dim=-1)
        diagnostics = {
            "use_behavior_proto": float(self.use_behavior_proto),
            "use_behavior_correction": float(self.use_behavior_correction),
            "use_private_residual": float(self.use_private_residual),
            "use_topk_correction": float(self.use_topk_correction),
            "use_calibration_gate": float(self.use_calibration_gate),
            "proto_gate_mean": float(proto_gate.mean().detach().cpu()),
            "proto_gate_std": float(proto_gate.std().detach().cpu()),
            "proto_norm_mean": float(proto_emb.norm(dim=-1).mean().detach().cpu()),
            "raw_enhanced_user_cos": float(proto_user_cos.mean().detach().cpu()),
            "pair_gate_mean": float(candidate_gate.mean().detach().cpu()),
            "pair_gate_std": float(candidate_gate.std().detach().cpu()),
            "behavior_score_mean": float(candidate_behavior.mean().detach().cpu()),
            "behavior_score_std": float(candidate_behavior.std().detach().cpu()),
            "residual_score_mean": float(candidate_residual.mean().detach().cpu()),
            "residual_score_std": float(candidate_residual.std().detach().cpu()),
            "behavior_correction_mean": float(behavior_correction.mean().detach().cpu()),
            "behavior_correction_abs_mean": float(behavior_correction.abs().mean().detach().cpu()),
            "residual_correction_mean": float(residual_correction.mean().detach().cpu()),
            "residual_correction_abs_mean": float(residual_correction.abs().mean().detach().cpu()),
            "total_correction_abs_mean": float(correction.abs().mean().detach().cpu()),
            "low_item_norm_mean": float(low_item.norm(dim=-1).mean().detach().cpu()),
            "high_item_norm_mean": float(high_item.norm(dim=-1).mean().detach().cpu()),
            "low_high_item_cos": float(low_high_cos.mean().detach().cpu()),
        }

        if was_training:
            self.train()
        return diagnostics

    def diagnostic_info(self):
        diagnostics = self.export_diagnostics()
        keys = [
            "proto_gate_mean",
            "raw_enhanced_user_cos",
            "pair_gate_mean",
            "behavior_correction_abs_mean",
            "residual_correction_abs_mean",
            "low_item_norm_mean",
            "high_item_norm_mean",
            "low_high_item_cos",
        ]
        details = ", ".join(f"{key}={diagnostics[key]:.4f}" for key in keys)
        return "BRIDGE diagnostics: " + details

    def export_embeddings(self):
        raw_user, all_item, _, user_frequencies, item_frequencies = DualFrequencyEncoder.forward(self)
        all_user, _, _ = self._apply_behavior_proto(raw_user, all_item)
        low_user, low_item, high_user, high_item = self._frequency_views(
            raw_user, all_item, user_frequencies, item_frequencies
        )
        return {
            "all_user": all_user,
            "all_item": all_item,
            "content_user": low_user,
            "content_item": low_item,
            "mm_user": high_user,
            "mm_item": high_item,
        }

# coding: utf-8
#
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class DualFrequencyEncoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DualFrequencyEncoder, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True
        self.num_freq_bands = config['num_freq_bands']
        self.freq_decomp_method = str(self._cfg_value(config, 'freq_decomp_method', 'svd')).lower()
        self.ib_direction = config['ib_direction']
        self.ib_alpha = config['ib_alpha']
        self.ib_mu = config['ib_mu']
        self.ib_phi_plus = config['ib_phi_plus']
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = config['num_layers']
        self.cold_start = 0
        self.dataset = dataset
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.ib_weight = config['ib_weight']
        self.feature_ablation = self._cfg_value(config, 'feature_ablation', 'all')
        self.use_item_graph = self._cfg_bool(config, 'use_item_graph', True)
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.mm_adj = None

        self.user_id_embedding = nn.Embedding(self.n_users, self.dim_latent)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim_latent)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = None
        self.use_user_graph = False
        user_emb_path = os.path.join(dataset_path, config['user_emb_file'])
        user_graph_path = os.path.join(dataset_path, config['user_graph_dict_file'])

        # The optional user graph branch needs both artifacts.  A standalone
        # user_emb.npy is not enough for topk_sample(), which consumes the
        # precomputed neighbor dictionary.
        if os.path.isfile(user_emb_path) and os.path.isfile(user_graph_path):
            self.user_graph_dict = np.load(user_graph_path, allow_pickle=True).item()
            user_emb = np.load(user_emb_path, allow_pickle=True)
            self.user_emb = torch.from_numpy(user_emb).to(self.device)
            print(">>>>self.user_emb.shape=", self.user_emb.shape)

            indices, user_adj = self.get_knn_adj_mat(self.user_emb)
            self.user_adj = user_adj

            self.use_user_graph = True
            print(">>>> Loaded optional user graph artifacts")
        elif os.path.isfile(user_emb_path):
            print(">>>> user_graph_dict_file not found, skip optional user graph: ", user_graph_path)
        else:
            print(f">>>> user_emb_file not found, skip loading: {user_emb_path}")

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        self.use_image_feature = self.v_feat is not None and self.feature_ablation not in ('no_image', 'no_content')
        self.use_text_feature = self.t_feat is not None and self.feature_ablation not in ('no_text', 'no_content')
        if self.feature_ablation not in ('all', 'no_image', 'no_text', 'no_content'):
            raise ValueError(f"Unknown feature_ablation: {self.feature_ablation}")
        if self.v_feat is not None and not self.use_image_feature:
            self.v_feat = torch.zeros_like(self.v_feat)
        if self.t_feat is not None and not self.use_text_feature:
            self.t_feat = torch.zeros_like(self.t_feat)
        print(
            "Dual-frequency feature controls: "
            f"feature_ablation={self.feature_ablation}, "
            f"use_image_feature={self.use_image_feature}, "
            f"use_text_feature={self.use_text_feature}, "
            f"use_item_graph={self.use_item_graph}"
        )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=not self.use_image_feature)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=not self.use_text_feature)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if not self.use_item_graph:
            self.mm_adj = None
            print(">>>>>Skip multimodal item graph propagation")
        elif self.feature_ablation == 'all' and os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
            print(">>>>>Loaded from: ", mm_adj_file)
        else:
            image_adj, text_adj = None, None
            if self.v_feat is not None and self.use_image_feature:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None and self.use_text_feature:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.use_image_feature and self.use_text_feature:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            elif not self.use_image_feature and not self.use_text_feature:
                self.mm_adj = self._identity_sparse(self.num_item)
            if self.feature_ablation == 'all':
                torch.save(self.mm_adj, mm_adj_file)
                print(">>>>>Save mm_adj to: ", mm_adj_file)
            else:
                print(">>>>>Built ablation-specific mm_adj without overwriting cache")
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.v_feat.size(1)).to(self.device)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)  # 256)
        if self.t_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.t_feat.size(1)).to(self.device)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)
        
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                             device=self.device, features=self.item_id_embedding.weight)

        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.FeqDecomposeOperator = FrequencyDecompositionModule(
            self.dim_latent,
            dim_x,
            dim_x,
            dim_x,
            self.num_freq_bands,
            self.ib_direction,
            self.ib_alpha,
            self.ib_mu,
            self.ib_phi_plus,
            hidden_dim=16,
            decomp_method=self.freq_decomp_method,
        )

    @staticmethod
    def _unwrap_cfg(value):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    @classmethod
    def _cfg_value(cls, config, key, default):
        return cls._unwrap_cfg(config[key]) if key in config else default

    @classmethod
    def _cfg_bool(cls, config, key, default):
        value = cls._cfg_value(config, key, default)
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        return bool(value)

    def _identity_sparse(self, size):
        idx = torch.arange(size, device=self.device, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(size, device=self.device, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (size, size), device=self.device).coalesce()

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(1e-12))
        n_items = context_norm.size(0)
        adj_size = (n_items, n_items)

        # Electronics has 63k items; building the full dense item-item
        # similarity matrix would require more than 15GB before top-k.
        # Chunked top-k keeps the exact same KNN graph construction while
        # bounding peak memory.
        chunk_size = int(self.config["knn_chunk_size"] or 4096) if hasattr(self, "config") else 4096
        chunk_size = max(1, chunk_size)
        if n_items <= chunk_size:
            sim = torch.mm(context_norm, context_norm.transpose(1, 0))
            _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
            del sim
        else:
            knn_chunks = []
            context_t = context_norm.transpose(1, 0)
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                sim = torch.mm(context_norm[start:end], context_t)
                _, chunk_ind = torch.topk(sim, self.knn_k, dim=-1)
                knn_chunks.append(chunk_ind)
                del sim
            knn_ind = torch.cat(knn_chunks, dim=0)
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        if self.use_user_graph:
            self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
            self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        pass

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self):
        representation = None
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index_dropv, self.edge_index, self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.image_embedding.weight)
            representation = self.v_rep
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.text_embedding.weight)
            if representation is None:
                representation = self.t_rep
            else:
                representation = torch.cat((self.id_rep, self.v_rep, self.t_rep), dim=1)

        if self.construction == 'weighted_sum':
            if self.v_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                                        self.weight_u)
            user_rep = torch.squeeze(user_rep)

        if self.construction == 'weighted_max':
            # pdb.set_trace()
            self.v_rep = torch.unsqueeze(self.v_rep, 2)

            self.t_rep = torch.unsqueeze(self.t_rep, 2)

            user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1, 2) * user_rep
            user_rep = torch.max(user_rep, dim=2).values
        
        if self.construction == 'cat':
            # pdb.set_trace()
            if self.v_rep is not None:
                user_rep = self.v_rep[:self.num_user]
            if self.t_rep is not None:
                user_rep = self.t_rep[:self.num_user]
            if self.v_rep is not None and self.t_rep is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                self.id_rep = torch.unsqueeze(self.id_rep, 2)
                user_rep = torch.cat((self.id_rep[:self.num_user], self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep

                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

        item_rep = representation[self.num_user:]

        # ---- Item-side graph propagation ----
        if self.use_item_graph:
            item_graph_input = item_rep
            item_graph_features = item_graph_input
            for layer_idx in range(self.n_layers):
                item_graph_features = torch.sparse.mm(self.mm_adj, item_graph_features)
        else:
            item_graph_features = item_rep

        # ---- User-side graph propagation ----
        user_modal_fused_input = torch.cat(
            (self.id_preference, self.v_preference, self.t_preference), dim=1
        )  # (num_users, dim_total)

        if self.use_user_graph:
            user_graph_features = user_modal_fused_input
            for layer_idx in range(self.n_layers):
                user_graph_features = torch.sparse.mm(self.user_adj, user_graph_features)
        else:
            user_graph_features = user_modal_fused_input

        # ---- Frequency-aware Decomposition fusion ----
        user_rep_multi, item_rep_multi, ib_loss, user_frequencies, item_frequencies = self.FeqDecomposeOperator(
            user_rep, item_rep,
            item_graph_features, user_graph_features
        )
        return user_rep_multi, item_rep_multi, ib_loss, user_frequencies, item_frequencies


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        return loss_value

    def bpr_margin_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
        margin: float = 2.0,
    ) -> torch.Tensor:
        """
        Margin-augmented BPR loss.
        Penalize when pos_score <= neg_score + margin.

        Args:
            user_emb:      (B, D)
            pos_item_emb:  (B, D)
            neg_item_emb:  (B, D)
            margin:        float, e.g., 0.5 ~ 2.0

        Returns:
            loss: scalar tensor
        """
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)  # (B,)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)  # (B,)

        # hinge: max(0, margin - (pos - neg))
        hinge_terms = F.relu(margin - (pos_scores - neg_scores))  # (B,)
        loss = hinge_terms.mean()
        return loss

    def cross_entropy_loss(self, u_emb, pos_i_emb, neg_i_emb):
        """
        Treat (user, pos_item) as positive samples and (user, neg_item) as negative samples,
        then concatenate them and optimize using a binary cross-entropy loss.
        """
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)  # [batch_size]
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)  # [batch_size]

        pos_labels = torch.ones_like(pos_scores)  # [batch_size],
        neg_labels = torch.zeros_like(neg_scores) # [batch_size],

        all_scores = torch.cat([pos_scores, neg_scores], dim=0)   # [2 * batch_size]
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)   # [2 * batch_size]

        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        
        return loss

    def frequency_contrastive_loss(self, user_frequencies, item_frequencies, tau=1.0):
        """
        Frequency-domain contrastive learning:
        - Encourage different frequency bands of the same user/item to be as similar as possible
        - Encourage high-frequency features of different users/items to be more discriminative
        """
        # user_frequencies: List[M], each of shape (num_nodes, embed_dim)
        # item_frequencies: List[M], each of shape (num_nodes, embed_dim)
        contrastive_loss = 0
        self.M = self.num_freq_bands

        # For all freq_bands
        for m in range(self.M):
            user_high = user_frequencies[m]
            item_high = item_frequencies[m]

            for n in range(self.M):
                if m != n:
                    user_low = user_frequencies[n]
                    item_low = item_frequencies[n]

                    user_sim = F.cosine_similarity(user_high, user_low, dim=-1)
                    item_sim = F.cosine_similarity(item_high, item_low, dim=-1)

                    contrastive_loss += F.mse_loss(user_sim, torch.ones_like(user_sim))
                    contrastive_loss += F.mse_loss(item_sim, torch.ones_like(item_sim))

            if m >= self.M // 2:
                user_contrast = torch.mm(user_high, user_high.T) / tau
                item_contrast = torch.mm(item_high, item_high.T) / tau

                user_contrast = F.log_softmax(user_contrast, dim=-1)
                item_contrast = F.log_softmax(item_contrast, dim=-1)

                contrastive_loss -= torch.mean(user_contrast)
                contrastive_loss -= torch.mean(item_contrast)

        return contrastive_loss / (self.M * (self.M - 1))

    def modality_frequency_contrastive_loss(
        self,
        user_band_embs,
        item_band_embs,
        temperature: float = 1.0,
    ):
        """
        Align (ID / visual / text) modalities within each frequency band, and encourage
        discriminability for higher-frequency bands.

        Args:
            user_band_embs: List[num_bands], each (N_user, 3*D)
            item_band_embs: List[num_bands], each (N_item, 3*D)
            temperature:    float

        Returns:
            loss: scalar tensor
        """
        num_bands = len(user_band_embs)
        loss = 0.0

        for band_idx in range(num_bands):
            user_band = user_band_embs[band_idx]  # (N_user, 3D)
            item_band = item_band_embs[band_idx]  # (N_item, 3D)

            user_id_emb, user_vis_emb, user_txt_emb = torch.chunk(user_band, 3, dim=-1)
            item_id_emb, item_vis_emb, item_txt_emb = torch.chunk(item_band, 3, dim=-1)

            user_cos_id_vis = F.cosine_similarity(user_id_emb, user_vis_emb, dim=-1)
            user_cos_id_txt = F.cosine_similarity(user_id_emb, user_txt_emb, dim=-1)
            user_cos_vis_txt = F.cosine_similarity(user_vis_emb, user_txt_emb, dim=-1)

            item_cos_id_vis = F.cosine_similarity(item_id_emb, item_vis_emb, dim=-1)
            item_cos_id_txt = F.cosine_similarity(item_id_emb, item_txt_emb, dim=-1)
            item_cos_vis_txt = F.cosine_similarity(item_vis_emb, item_txt_emb, dim=-1)

            ones_u = torch.ones_like(user_cos_id_vis)
            ones_i = torch.ones_like(item_cos_id_vis)

            loss = loss + F.mse_loss(user_cos_id_vis, ones_u)
            loss = loss + F.mse_loss(user_cos_id_txt, ones_u)
            loss = loss + F.mse_loss(user_cos_vis_txt, ones_u)

            loss = loss + F.mse_loss(item_cos_id_vis, ones_i)
            loss = loss + F.mse_loss(item_cos_id_txt, ones_i)
            loss = loss + F.mse_loss(item_cos_vis_txt, ones_i)

            if band_idx >= num_bands // 2:
                user_logits = (user_band @ user_band.T) / temperature
                item_logits = (item_band @ item_band.T) / temperature
                loss = loss - F.log_softmax(user_logits, dim=-1).mean()
                loss = loss - F.log_softmax(item_logits, dim=-1).mean()

        # 6 = 3 modality pairs * (user + item)
        normalizer = max(1, num_bands * 6)
        return loss / normalizer

    def calculate_loss(self, interaction):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]
        
        user_emb_all, item_emb_all, ib_loss, user_frequencies, item_frequencies = self.forward()
        self.build_item_graph = False

        user_emb_batch = user_emb_all[user_indices]
        pos_item_emb_batch = item_emb_all[pos_item_indices]
        neg_item_emb_batch = item_emb_all[neg_item_indices]

        user_band_emb_batch = []
        item_band_emb_batch = []
        for item_emb in user_frequencies:
            temp = item_emb[user_indices]
            user_band_emb_batch.append(temp)

        for user_emb in item_frequencies:
            temp = user_emb[pos_item_indices]
            item_band_emb_batch.append(temp)
        cl_loss = self.frequency_contrastive_loss(user_band_emb_batch, item_band_emb_batch)

        batch_mf_loss = self.cross_entropy_loss(user_emb_batch, pos_item_emb_batch, neg_item_emb_batch)

        return batch_mf_loss + ib_loss * self.ib_weight + cl_loss * self.reg_weight

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _, _, _ = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_layer = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_layer = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        outs = [x]
        h = x
        for _ in range(self.num_layer):
            h = self.conv_embed_layer(h, edge_index)
            outs.append(h)
        x_hat = sum(outs)
        return x_hat, self.preference

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class MKMMDLoss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        """
        MK-MMD Loss function to align two distributions.
        :param kernel_mul: The scale factor between different Gaussian kernels.
        :param kernel_num: Number of Gaussian kernels used in multi-kernel MMD.
        """
        super(MKMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target, bandwidth_list):
        """
        Compute multi-kernel Gaussian kernel values between two embeddings.
        :param source: Tensor of shape [Batch, Dimension]
        :param target: Tensor of shape [Batch, Dimension]
        :param bandwidth_list: List of bandwidths for multiple Gaussian kernels.
        :return: Computed multi-kernel Gram matrix.
        """
        batch_size = source.shape[0]
        
        # Compute pairwise L2 distance (||X - Y||^2)
        source_sq = torch.sum(source ** 2, dim=1, keepdim=True)
        target_sq = torch.sum(target ** 2, dim=1, keepdim=True)

        dist_matrix = (
            source_sq + target_sq.T - 2.0 * torch.matmul(source, target.T)
        )

        # Compute multi-kernel MMD using Gaussian kernels
        kernel_val = torch.zeros_like(dist_matrix)
        for bandwidth in bandwidth_list:
            kernel_val += torch.exp(-dist_matrix / (2.0 * bandwidth))

        return kernel_val

    def forward(self, i_id_emb, i_v_emb):
        """
        Compute the MK-MMD loss between ID embeddings and visual embeddings.
        :param i_id_emb: Tensor [Batch, Dimension] - GNN-learned ID embeddings
        :param i_v_emb: Tensor [Batch, Dimension] - GNN-learned visual embeddings
        :return: MK-MMD loss (scalar tensor)
        """
        batch_size = i_id_emb.shape[0]

        # Compute median distance as base bandwidth
        combined = torch.cat([i_id_emb, i_v_emb], dim=0)
        pairwise_distances = torch.cdist(combined, combined, p=2)
        median_dist = torch.median(pairwise_distances[pairwise_distances > 0])
        bandwidth_list = [median_dist * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        
        # Compute MK-MMD loss
        K_ii = self.gaussian_kernel(i_id_emb, i_id_emb, bandwidth_list)
        K_vv = self.gaussian_kernel(i_v_emb, i_v_emb, bandwidth_list)
        K_iv = self.gaussian_kernel(i_id_emb, i_v_emb, bandwidth_list)

        loss = torch.mean(K_ii) + torch.mean(K_vv) - 2 * torch.mean(K_iv)
        return loss

class InfoNCE_Loss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        """
        Contrastive learning InfoNCE loss
        :param temperature: Controls the smoothness of softmax, typical values are 0.07~0.2
        """
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        """
        Compute the InfoNCE contrastive loss
        :param user_emb: Tensor [Batch, Dim] - user embeddings
        :param item_emb: Tensor [Batch, Dim] - item embeddings
        :return: contrastive loss (scalar)
        """
        batch_size = user_emb.shape[0]

        # Compute cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)  # normalize
        item_emb = F.normalize(item_emb, p=2, dim=1)  # normalize

        # Compute positive-pair similarity (matched pairs on the diagonal)
        pos_sim = torch.sum(user_emb * item_emb, dim=1)  # [Batch]
        pos_sim = pos_sim / self.temperature  # temperature scaling

        # Compute global similarity (pairwise similarities across the batch)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature  # [Batch, Batch]

        # Compute InfoNCE loss (diagonal entries are positives, others are negatives)
        labels = torch.arange(batch_size).to(user_emb.device)  # construct ground-truth matching indices
        loss = F.cross_entropy(logits, labels)

        return loss

class FrequencyDecompositionModule(nn.Module):
    def __init__(
        self,
        dim_latent,
        id_dim,
        v_dim,
        t_dim,
        M,
        ib_direction,
        ib_alpha,
        ib_mu,
        ib_phi_plus,
        hidden_dim=128,
        decomp_method='svd',
    ):
        """
        Frequency decomposition + MoE structure:
        - id_dim: dimension of the ID representation
        - v_dim: dimension of the visual representation
        - t_dim: dimension of the textual representation
        - M: number of frequency bands
        - num_experts: number of experts in each MoE
        - hidden_dim: hidden dimension of the MLP in the MoE
        - use_gft: whether to use GFT for decomposition (otherwise use SVD)
        """
        super(FrequencyDecompositionModule, self).__init__()
        self.M = M
        self.id_dim = id_dim
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.all_dim = id_dim + v_dim + t_dim
        self.dim_latent = dim_latent
        self.decomp_method = str(decomp_method).lower()
        aliases = {
            'none': 'none_equal_capacity',
            'none_equalcap': 'none_equal_capacity',
            'no_decomp': 'none_equal_capacity',
            'no_decomp_equalcap': 'none_equal_capacity',
            'no_decomp_equal_capacity': 'none_equal_capacity',
        }
        self.decomp_method = aliases.get(self.decomp_method, self.decomp_method)
        valid_methods = {'svd', 'gram', 'dct', 'random', 'none_equal_capacity'}
        if self.decomp_method not in valid_methods:
            raise ValueError(f"Unknown freq_decomp_method: {decomp_method}")
        self._basis_cache = {}
        # M Freq_Bands Fusion
        self.freq_weights = nn.Parameter(torch.ones(M) / M)
        self.fusion_layer_user = TaskAwareFrequencyFusion(self.M, self.all_dim, ib_direction, ib_alpha, ib_mu, ib_phi_plus)
        self.fusion_layer_item = TaskAwareFrequencyFusion(self.M, self.all_dim, ib_direction, ib_alpha, ib_mu, ib_phi_plus)

    def _split_sizes(self, feat_dim):
        base = feat_dim // self.M
        remainder = feat_dim % self.M
        return [base + (1 if i < remainder else 0) for i in range(self.M)]

    def frequency_decompose_svd(self, rep):
        """
        Perform SVD on `rep` and split (U, S, Vh) into M parts according to the singular values.
        Finally, return M partial-sum matrices, each having the same shape as `rep`.
        """
        M = self.M
        # rep: [N, F]
        try:
            U, S, Vh = torch.linalg.svd(rep, full_matrices=False)
        except RuntimeError:
            return self.frequency_decompose_gram(rep)
        # U:  [N, F]
        # S:  [F, ]  (one dimension)
        # Vh: [F, F]

        N, F_ = U.shape  # F_ should be as F
        assert F_ == S.shape[0] and F_ == Vh.shape[0] == Vh.shape[1], \
            f"Shape mismatch in SVD: U({U.shape}), S({S.shape}), Vh({Vh.shape})"

        # calculate each band size
        # e.g. F=192, M=3, split_sizes = [64,64,64]
        split_sizes = self._split_sizes(F_)
        # sum(split_sizes) == F_

        # splitting S, U, Vh one by one
        freq_components = []
        start = 0
        for size_i in split_sizes:
            end = start + size_i

            # S_i: [size_i, ]
            S_i = S[start:end]  
            # U_i: [N, size_i]
            U_i = U[:, start:end]  
            # V_i: [size_i, F]
            V_i = Vh[start:end, :]

            # reconstruct corresponding matrix for the band = sum_{k in [start, end]}(S_k * U_{:,k} outer Vh_{k,:})
            # U_i @ diag(S_i) @ V_i
            diag_S_i = torch.diag(S_i)              # (size_i, size_i)
            partial_rep = U_i @ diag_S_i @ V_i      # (N, F)
            freq_components.append(partial_rep)

            start = end

        return freq_components  # List[M], with each element shape (N, F)

    def frequency_decompose_gram(self, rep):
        """Stable fallback for ill-conditioned tall matrices.

        For right singular vectors V, the band reconstruction
        U_i S_i V_i^T is equivalent to rep V_i V_i^T. Computing V from
        the small F x F Gram matrix avoids CUDA SVD convergence failures.
        """
        M = self.M
        feat_dim = rep.shape[1]
        rep = torch.nan_to_num(rep, nan=0.0, posinf=1e4, neginf=-1e4)
        gram = rep.transpose(0, 1) @ rep
        gram = 0.5 * (gram + gram.transpose(0, 1))
        gram = torch.nan_to_num(gram, nan=0.0, posinf=1e8, neginf=-1e8)
        diag_scale = gram.diagonal().abs().mean().clamp_min(1.0)
        eye = torch.eye(feat_dim, dtype=rep.dtype, device=rep.device)
        gram = gram + eye * (diag_scale * 1e-6)
        try:
            eigvals, eigvecs = torch.linalg.eigh(gram)
            order = torch.argsort(eigvals, descending=True)
            eigvecs = eigvecs[:, order]
        except RuntimeError:
            try:
                gram_cpu = gram.detach().double().cpu()
                gram_cpu = 0.5 * (gram_cpu + gram_cpu.transpose(0, 1))
                diag_cpu = gram_cpu.diagonal().abs().mean().clamp_min(1.0)
                gram_cpu = gram_cpu + torch.eye(feat_dim, dtype=torch.float64) * (diag_cpu * 1e-4)
                eigvals, eigvecs_cpu = torch.linalg.eigh(gram_cpu)
                order = torch.argsort(eigvals, descending=True)
                eigvecs = eigvecs_cpu[:, order].to(device=rep.device, dtype=rep.dtype)
            except RuntimeError:
                eigvecs = torch.eye(feat_dim, dtype=rep.dtype, device=rep.device)

        split_sizes = self._split_sizes(feat_dim)

        freq_components = []
        start = 0
        for size_i in split_sizes:
            end = start + size_i
            basis = eigvecs[:, start:end]
            freq_components.append((rep @ basis) @ basis.transpose(0, 1))
            start = end

        return freq_components

    def _dct_basis(self, feat_dim, device, dtype):
        cache_key = ('dct', feat_dim, str(device), dtype)
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]

        n = torch.arange(feat_dim, device=device, dtype=dtype)
        k = torch.arange(feat_dim, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos((torch.pi / feat_dim) * (n + 0.5) * k)
        basis[0] = basis[0] * (1.0 / feat_dim) ** 0.5
        if feat_dim > 1:
            basis[1:] = basis[1:] * (2.0 / feat_dim) ** 0.5
        basis = basis.transpose(0, 1).contiguous()
        self._basis_cache[cache_key] = basis
        return basis

    def _random_basis(self, feat_dim, device, dtype):
        cache_key = ('random', feat_dim, str(device), dtype)
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]

        gen = torch.Generator(device='cpu')
        gen.manual_seed(20260519 + feat_dim)
        mat = torch.randn(feat_dim, feat_dim, generator=gen, dtype=torch.float64)
        q, _ = torch.linalg.qr(mat)
        basis = q.to(device=device, dtype=dtype).contiguous()
        self._basis_cache[cache_key] = basis
        return basis

    def _basis_decompose(self, rep, basis):
        split_sizes = self._split_sizes(rep.shape[1])
        freq_components = []
        start = 0
        for size_i in split_sizes:
            end = start + size_i
            basis_band = basis[:, start:end]
            freq_components.append((rep @ basis_band) @ basis_band.transpose(0, 1))
            start = end
        return freq_components

    def frequency_decompose_dct(self, rep):
        basis = self._dct_basis(rep.shape[1], rep.device, rep.dtype)
        return self._basis_decompose(rep, basis)

    def frequency_decompose_random(self, rep):
        basis = self._random_basis(rep.shape[1], rep.device, rep.dtype)
        return self._basis_decompose(rep, basis)

    def frequency_decompose_none_equal_capacity(self, rep):
        component = rep / float(self.M)
        return [component for _ in range(self.M)]

    def frequency_decompose(self, rep):
        if self.decomp_method == 'svd':
            return self.frequency_decompose_svd(rep)
        if self.decomp_method == 'gram':
            return self.frequency_decompose_gram(rep)
        if self.decomp_method == 'dct':
            return self.frequency_decompose_dct(rep)
        if self.decomp_method == 'random':
            return self.frequency_decompose_random(rep)
        if self.decomp_method == 'none_equal_capacity':
            return self.frequency_decompose_none_equal_capacity(rep)
        raise ValueError(f"Unknown freq_decomp_method: {self.decomp_method}")

    def frequency_decompose_svd_separate(self, rep):
        """ Split modalities first, then perform SVD decomposition separately for each modality. """
        M = self.M

        id_rep, visual_rep, text_rep = torch.split(rep, [self.dim_latent, self.dim_latent, self.dim_latent], dim=-1)

        # Decomposition for different modalities with the configured method.
        id_frequencies = self.frequency_decompose(id_rep)    # List[M], with each element (num_samples, each_dim)
        visual_frequencies = self.frequency_decompose(visual_rep)
        text_frequencies = self.frequency_decompose(text_rep)

        # re-concat to original dimensions
        freq_components = [torch.cat([id_frequencies[i], visual_frequencies[i], text_frequencies[i]], dim=-1) for i in range(M)]

        return freq_components  # List[M], with each element shape (num_samples, all_dim)

    def frequency_kl_loss(self, gate_values):
        """
        gate_values: (num_nodes, M, 1), indicating the importance of each frequency band
        """
        # kl_loss = torch.mean(torch.norm(gate_values, dim=-1))  # push unimportant bands close to 0
        # kl_loss = torch.mean((1 - gate_values) * torch.norm(gate_values, dim=-1))  # apply stronger KL constraint to low-weight bands only
        gate_values = gate_values.squeeze(-1)  # convert to (num_nodes, M)
        kl_loss = torch.mean((1 - gate_values) * torch.norm(gate_values, dim=-1, keepdim=True))  # match (num_nodes, M) shape
        return kl_loss

    def forward(self, user_rep, item_rep, h, h_u):
        """
        - user_rep: [num_user, id_dim + v_dim + t_dim], GCN user representations
        - item_rep: [num_item, id_dim + v_dim + t_dim], GCN item representations
        """
        user_frequencies = self.frequency_decompose_svd_separate(user_rep)
        item_frequencies = self.frequency_decompose_svd_separate(item_rep)

        user_fre = self.frequency_decompose_svd_separate(h_u)
        item_fre = self.frequency_decompose_svd_separate(h)

        user_fre_new = []
        for i in range(self.M):
            user_fre_new.append(user_fre[i]+user_frequencies[i])

        item_fre_new = []
        for i in range(self.M):
            item_fre_new.append(item_fre[i]+item_frequencies[i])

        user_rep_fuse, ib_loss_user = self.fusion_layer_user(user_fre_new, user_rep)
        item_rep_fuse, ib_loss_item = self.fusion_layer_item(item_fre_new, item_rep)

        #user_rep_reconstructed = sum(user_frequencies)  # (num_user, embed_dim)
        #item_rep_reconstructed = sum(item_frequencies)  # (num_user, embed_dim)

        ib_loss = ib_loss_user + ib_loss_item
        return user_rep_fuse, item_rep_fuse, ib_loss, user_fre_new, item_fre_new

class TaskAwareFrequencyFusion(nn.Module):
    def __init__(self, M, embed_dim, ib_direction="Pos", ib_alpha=1.0, ib_mu=1.0, ib_phi_plus=0.0):
        super().__init__()
        self.M = M
        self.embed_dim = embed_dim

        self.ib_alpha = ib_alpha
        self.ib_mu = ib_mu
        self.ib_phi_plus = ib_phi_plus

        self.num_bands = M
        self.ib_direction = ib_direction
        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.ones(M))  # (M,)

        # Frequency gate
        if self.ib_direction == "Dual":
            # we provide a new direction to allow both positive/negative deviations for expanding
            self.freq_gate = nn.Sequential(
                nn.Linear(embed_dim, M),
                nn.Tanh()
            )
        elif self.ib_direction == "Pos":
            self.freq_gate = nn.Sequential(
                nn.Linear(embed_dim, M),
                nn.Sigmoid()
            )
        else:
            print(self.ib_direction)
            raise Exception("Invalid Setting")
        self.freq_gate_raw = nn.Linear(embed_dim, M)

        # Learnable Residual Weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.ib_layer = IB_Layer(embed_dim, embed_dim//4)

        self.gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def ib_surrogate_loss_from_gate(
        self,
        gate_values: torch.Tensor,   # (N, M, 1) or (N, M)
        alpha: float = 1.0,
        mu: float = 1.0,
        phi_plus: float = 0.0,
        eps: float = 1e-12,
        enforce_nonneg: bool = True,
    ) -> torch.Tensor:
        if gate_values.dim() == 3:
            gate_values = gate_values.squeeze(-1)  # (N, M)
        assert gate_values.dim() == 2, f"Expected (N,M), got {gate_values.shape}"

        delta = gate_values - 1.0  # (N, M)

        # ---- direction handling ----
        # self.ib_direction in {"pos", "dual"}
        if self.ib_direction == "Pos":
            if enforce_nonneg:
                delta = F.relu(delta)
            delta_for_threshold = delta
        elif self.ib_direction == "Dual":
            # we provide a new direction to allow both positive/negative deviations for expanding
            # adjustment here correspondingly
            delta_for_threshold = delta.abs()
        else:
            raise ValueError(f"Unknown ib_direction: {self.ib_direction}")

        delta_norm_sq = torch.sum(delta * delta, dim=1)  # (N,)
        term1 = alpha * delta_norm_sq.mean()

        delta_norm = torch.sqrt(delta_norm_sq + eps)  # (N,)
        exceed = F.relu(delta_for_threshold - phi_plus)  # (N, M)
        exceed_sum = torch.sum(exceed, dim=1)            # (N,)
        term2 = mu * (delta_norm * exceed_sum).mean()

        return term1 + term2

    def forward(
        self,
        band_components,                # List[M], each (N, D)
        task_emb: torch.Tensor,         # (N, D)
        enforce_nonneg: bool = True,
    ):
        """
        Returns:
            fused_emb: (N, D)
            ib_loss:   scalar tensor
        """
        band_tensor = torch.stack(band_components, dim=1)  # (N, M, D)

        band_gates = 1.0 + self.gate_scale * self.freq_gate(task_emb)  # (N, M)
        ib_loss = self.ib_surrogate_loss_from_gate(
            band_gates,
            alpha=self.ib_alpha,
            mu=self.ib_mu,
            phi_plus=self.ib_phi_plus,
            enforce_nonneg=enforce_nonneg,
        )
        
        band_gates = band_gates.unsqueeze(-1)  # (N, M, 1)
    
        band_weights = torch.sigmoid(self.freq_weights).view(1, self.num_bands, 1)  # (1, M, 1)
        gated_bands = band_gates * band_tensor  # (N, M, D)

        fused_emb = torch.sum(band_weights * gated_bands, dim=1)  # (N, D)
        return fused_emb, ib_loss
    

class IB_Layer(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, embed_dim)

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return compressed, reconstructed

class TaskAwareFrequencyFusionMulti(nn.Module):
    def __init__(self, num_bands: int, embed_dim: int, num_modality: int):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.modality_dim = embed_dim // num_modality

        self.band_logits_id = nn.Parameter(torch.ones(num_bands))  # (M,)
        self.band_logits_vis = nn.Parameter(torch.ones(num_bands))  # (M,)
        self.band_logits_txt = nn.Parameter(torch.ones(num_bands))  # (M,)

        self.band_gate_id = nn.Sequential(
            nn.Linear(self.modality_dim, num_bands),
            nn.LayerNorm(num_bands),
            nn.GELU(),
            nn.Linear(num_bands, num_bands),
            nn.Sigmoid(),
        )
        self.band_gate_vis = nn.Sequential(
            nn.Linear(self.modality_dim, num_bands),
            nn.LayerNorm(num_bands),
            nn.GELU(),
            nn.Linear(num_bands, num_bands),
            nn.Sigmoid(),
        )
        self.band_gate_txt = nn.Sequential(
            nn.Linear(self.modality_dim, num_bands),
            nn.LayerNorm(num_bands),
            nn.GELU(),
            nn.Linear(num_bands, num_bands),
            nn.Sigmoid(),
        )

        self.residual_mix = nn.Parameter(torch.tensor(0.5))  # alpha in [0,1] (learnable)

    def forward(self, band_components, task_emb: torch.Tensor):
        """
        Args:
            band_components: List[M], each (N, 3*D)
            task_emb:        (N, 3*D)

        Returns:
            fused_emb:  (N, 3*D)
            band_gates: (N, 3*M)
        """
        num_nodes = task_emb.size(0)
        d = self.modality_dim
        m = self.num_bands

        band_tensor = torch.stack(band_components, dim=1)  # (N, M, 3D)
        band_id, band_vis, band_txt = torch.split(band_tensor, d, dim=-1)  # each (N, M, D)

        gate_id = self.band_gate_id(task_emb[:, :d]).unsqueeze(-1)          # (N, M, 1)
        gate_vis = self.band_gate_vis(task_emb[:, d:2 * d]).unsqueeze(-1)   # (N, M, 1)
        gate_txt = self.band_gate_txt(task_emb[:, 2 * d:]).unsqueeze(-1)    # (N, M, 1)

        weight_id = torch.softmax(self.band_logits_id, dim=0).view(1, m, 1)   # (1, M, 1)
        weight_vis = torch.softmax(self.band_logits_vis, dim=0).view(1, m, 1)
        weight_txt = torch.softmax(self.band_logits_txt, dim=0).view(1, m, 1)

        gated_id = gate_id * band_id
        gated_vis = gate_vis * band_vis
        gated_txt = gate_txt * band_txt

        fused_id = (weight_id * gated_id).sum(dim=1)      # (N, D)
        fused_vis = (weight_vis * gated_vis).sum(dim=1)
        fused_txt = (weight_txt * gated_txt).sum(dim=1)

        residual_id = band_id.sum(dim=1)                  # (N, D)
        residual_vis = band_vis.sum(dim=1)
        residual_txt = band_txt.sum(dim=1)

        alpha = self.residual_mix
        fused_id = alpha * fused_id + (1.0 - alpha) * residual_id
        fused_vis = alpha * fused_vis + (1.0 - alpha) * residual_vis
        fused_txt = alpha * fused_txt + (1.0 - alpha) * residual_txt

        fused_emb = torch.cat([fused_id, fused_vis, fused_txt], dim=-1)  # (N, 3D)

        band_gates = torch.cat(
            [gate_id.squeeze(-1), gate_vis.squeeze(-1), gate_txt.squeeze(-1)],
            dim=-1,
        )  # (N, 3M)

        return fused_emb, band_gates

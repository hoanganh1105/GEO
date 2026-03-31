import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np
from loss import loss_function



class MLPFusion(nn.Module):
    def __init__(self, hidden_dim=16):
        super(MLPFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, l_nll, l_energy, l_one_class):
        if l_nll.dim() == 1:
            l_nll = l_nll.unsqueeze(1)
            l_energy = l_energy.unsqueeze(1)
            l_one_class = l_one_class.unsqueeze(1)
        x = torch.cat([l_nll, l_energy, l_one_class], dim=-1)
        return self.mlp(x)

class GNNSafe(nn.Module):
    '''
    The model class of energy-based models for out-of-distribution detection
    The parameter args.use_reg and args.use_prop control the model versions:
        Energy: args.use_reg = False, args.use_prop = False
        Energy FT: args.use_reg = True, args.use_prop = False
        GNNSafe: args.use_reg = False, args.use_prop = True
        GNNSafe++ args.use_reg = True, args.use_prop = True
    '''
    def __init__(self, d, c, args):
        super(GNNSafe, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)
        elif args.backbone == 'gen': 
            self.encoder = GEN(in_channels=d, hidden_channels=args.hidden_channels, out_channels=c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.center = None 
        self.radius = nn.Parameter(torch.tensor(0.0))
        self.fusion = MLPFusion(hidden_dim=16)
    
    def init_center(self, dataset_ind, device):
        """Khởi tạo tâm c dựa trên giá trị trung bình của các node training"""
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        with torch.no_grad():
            # Dùng feature_list để lấy embeddings thay vì logits
            logits, out_features = self.encoder.feature_list(x, edge_index)
            embeddings = out_features[-1] # Lấy đặc trưng ở lớp ẩn cuối cùng
            
            train_idx = dataset_ind.splits['train']
            self.center = torch.mean(embeddings[train_idx], dim=0)
            
            eps = 0.001
            self.center[(abs(self.center) < eps) & (self.center < 0)] = -eps
            self.center[(abs(self.center) < eps) & (self.center > 0)] = eps

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        '''
        Return anomaly score for all input nodes.
        By default, it returns the negative energy (or propagated energy).
        If args.use_mlp_fusion is enabled, it utilizes the MLP-based late-fusion 
        mechanism to combine L_NLL, L_energy, and L_one-class into a single scalar score.
        '''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        
        # =====================================================================
        # OPTIONAL FEATURE: MLP-based Score Fusion (Enabled via args)
        # =====================================================================
        if hasattr(args, 'use_mlp_fusion') and args.use_mlp_fusion:
            # 1. Forward pass to obtain both logits and latent embeddings
            if hasattr(args, 'use_occ') and args.use_occ:
                logits, out_features = self.encoder.feature_list(x, edge_index)
                embeddings = out_features[-1]
            else:
                logits = self.encoder(x, edge_index)
                embeddings = None
                
            # 2. Compute Negative Log-Likelihood (L_NLL)
            # Since test labels are unavailable, we use the negative maximum log-probability as a proxy
            log_probs = F.log_softmax(logits, dim=-1)
            l_nll_node = -torch.max(log_probs, dim=-1)[0]
            
            # 3. Compute Energy-based Density (L_energy)
            if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
                logits_tmp = torch.stack([logits, torch.zeros_like(logits)], dim=2)
                l_energy_node = - args.T * torch.logsumexp(logits_tmp / args.T, dim=-1).sum(dim=1)
            else: # for single-label multi-class classification
                l_energy_node = - args.T * torch.logsumexp(logits / args.T, dim=-1)
                
            if args.use_prop: # apply energy belief propagation
                l_energy_node = self.propagation(l_energy_node, edge_index, args.K, args.alpha)
                
            # 4. Compute Spatial Geometric Distance (L_one-class)
            # Calculated as the squared Euclidean distance to the hypersphere center
            if embeddings is not None and getattr(self, 'center', None) is not None:
                l_occ_node = torch.sum((embeddings - self.center)**2, dim=1)
            else:
                l_occ_node = torch.zeros_like(l_energy_node)
                
            # 5. Fuse the distinct scalar scores using the Multi-Layer Perceptron
            anomaly_scores = self.fusion(l_nll_node, l_energy_node, l_occ_node)
            return anomaly_scores[node_idx].squeeze()

        # =====================================================================
        # DEFAULT BEHAVIOR: Standard Energy/Propagation Score
        # =====================================================================
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
            
        if args.use_prop: # use energy belief propagation
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
            
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        # logits_in = self.encoder(x_in, edge_index_in)
        # logits_out = self.encoder(x_out, edge_index_out)
        # LẤY THÊM EMBEDDINGS TỪ FEATURE_LIST
        if hasattr(args, 'use_occ') and args.use_occ:
            logits_in, out_features_in = self.encoder.feature_list(x_in, edge_index_in)
            embeddings_in = out_features_in[-1]
            logits_out = self.encoder(x_out, edge_index_out)
        else:
            logits_in = self.encoder(x_in, edge_index_in)
            logits_out = self.encoder(x_out, edge_index_out)
        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        if args.use_reg: # if use energy regularization
            if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
            else: # for single-label multi-class classification
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop: # use energy belief propagation
                energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)[train_ood_idx]
            else:   
                energy_in = energy_in[train_in_idx]
                #energy_out = energy_out[train_in_idx]
                energy_out = energy_out[train_ood_idx]

            # truncate to have the same length
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            # compute regularization loss
            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)

            loss = sup_loss + args.lamda * reg_loss
        else:
            loss = sup_loss

        #One-Class Loss (L_one-class)
        if hasattr(args, 'use_occ') and args.use_occ:
            outputs_train = embeddings_in[train_in_idx]
            occ_loss, dist, scores = loss_function(args.nu, self.center, outputs_train, radius=self.radius)
            loss = loss + args.beta * occ_loss
        return loss

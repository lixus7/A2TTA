import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class MLP_Model(nn.Module):
    """Some Information about MLP"""
    def __init__(self, args):
        super(MLP_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).transpose(1, 2).unsqueeze(-1)
        
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).squeeze(-1).reshape(1, 2)
        x = prediction.reshape(-1, 12)
        return x



class LSTM_Model(nn.Module):
    """Some Information about LSTM"""
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).unsqueeze(-1).transpose(1, 2).transpose(1, 3)   # [bs, t, n, f]
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)  # (b, f, n, t) -> (b, n, f, t) -> (b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)  # (b * n, f, 1, t) -> (b * n, init_dim, 1, t) -> (b * n, init_dim, t) -> (b * n, t, init_dim)

        out, _ = self.lstm(x)  # (b * n, t, hidden_dim) -> (b * n, t, hidden_dim)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b*n, t)
        return x


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=10):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_a = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_dim, r)))
        self.lora_b = nn.Parameter(torch.zeros(r, out_dim))
        self.scaling = 1 / (r * in_dim)

    def forward(self, x):
        return x + self.scaling * torch.matmul(torch.matmul(x, self.lora_a.to(x.device)), self.lora_b.to(x.device))
    

class STLora_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(STLora_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        self.lora_layers = nn.ModuleList()  # 存放LoRA层的列表
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    
    def add_lora_layer(self):
        in_dim = self.args.gcn["hidden_channel"]
        out_dim = self.args.gcn["hidden_channel"]
        lora_layer = LoRALayer(in_dim, out_dim)
        self.lora_layers.append(lora_layer)
        self.freeze_lora_layers()  # 冻结现有的LoRA层
    
    def freeze_lora_layers(self):
        for lora_layer in self.lora_layers[:-1]:  # 冻结除了最后一个之外的所有LoRA层
            for param in lora_layer.parameters():
                param.requires_grad = False

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        x = self.tcn1(x)                                           # [bs * N, 1, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x


class EAC_Model(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], 
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Initialize subspace and adjust matrix
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
        # Compute adaptive parameters using low-rank matrices
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes




class TrafficStream_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(TrafficStream_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x




# =====================================================================
# Ported from ST-TTC (https://github.com/Onedean/ST-TTC, arxiv 2506.00635)
#   continual_learning_setting/ — retrain-type STGNN backbone trained
#   per-year, paired with a spectral-domain calibrator (FRPlusModule)
#   that runs at test time inside the trainer.
# The forward pass here is a plain STGNN base predictor (same shape as
# TrafficStream_Model). The TTC overlay is applied in test_model_with_ttc.
# =====================================================================
class STTTC_Model(nn.Module):
    def __init__(self, args):
        super(STTTC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))
        x = self.tcn1(x)
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.args.gcn["out_channel"]))
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))
        x = self.tcn1(x)
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.args.gcn["out_channel"]))
        x = x + data.x
        return x


class STKEC_Model(nn.Module):
    """Some Information about STKEC_Model"""
    def __init__(self, args):
        super(STKEC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()

        self.memory=nn.Parameter(torch.zeros(size=(args.cluster, args.gcn["out_channel"]), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
    def forward(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]
        scores = F.softmax(attention, dim=1)                       # [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, scores
    
    def feature(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        return x



class Universal_Model(nn.Module):
    def __init__(self, args):
        super(Universal_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.use_eac = args.use_eac
        
        # Initialize GCN layers based on spectral (sp) or spatial (st) options
        if args.gcn_type == 'st':
            self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
            self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["in_channel"], bias=True, gcn=False)
        elif args.gcn_type == 'sp':
            self.gcn1 = ChebGraphConv(args.gcn["in_channel"], args.gcn["hidden_channel"])
            self.gcn2 = ChebGraphConv(args.gcn["hidden_channel"], args.gcn["in_channel"])
        
        # Select TCN type based on args
        if args.tcn_type == 'conv':
            self.tcn = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], 
                                kernel_size=args.tcn["kernel_size"],
                                dilation=args.tcn["dilation"],
                                padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2))
        elif args.tcn_type == 'rec':
            self.tcn = nn.LSTM(input_size=args.gcn["hidden_channel"], hidden_size=args.gcn["hidden_channel"], batch_first=True)
        elif args.tcn_type == 'attn':
            self.tcn = nn.MultiheadAttention(embed_dim=args.gcn["hidden_channel"], num_heads=4)
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        if self.use_eac:
            self.rank = args.rank  # 设定低秩的值
            self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
            self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
            self.year = args.year
            self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
        
        B, N, T = x.shape
        
        if self.use_eac:
            adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
            x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        # Apply the selected GCN layers
        x = F.relu(self.gcn1(x, adj))  # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # [bs * N, 1, feature]
        
        # Apply the selected TCN method
        if self.args.tcn_type == 'conv':
            x = self.tcn(x)  # temporal convolution
        elif self.args.tcn_type == 'rec':
            # x = x.reshape((-1, self.args.gcn["hidden_channel"])).unsqueeze(dim=-1)
            # out, _ = self.tcn(x)
            # x = out.reshape((-1, 1, self.args.gcn["hidden_channel"]))
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x)
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])
        elif self.args.tcn_type == 'attn':
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x, x, x)  # Multihead attention
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])


        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # [bs, N, feature]
        x = self.gcn2(x, adj)  # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # [bs * N, feature]

        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:

            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))

            self.num_nodes = new_num_nodes


# =====================================================================
# Ported from STRAP-main (NeurIPS'25 STRAP) src/model/model.py
# Includes:
#   * Four pluggable backbones: STGNN / DCRNN / ASTGNN / TGCN
#   * Four retrain wrappers:    STGNN_Model / DCRNN_Model / ASTGNN_Model / TGCN_Model
#   * PECPM_Model  (KDD'23 baseline, re-implemented by STRAP authors)
#   * RAP_Model    (= STRAP, NeurIPS'25)
#   * Simplified STRAP retrieval module + helpers
# =====================================================================
import os
import pickle


# -----------------------------------------------
# Backbones
# -----------------------------------------------

class STGNN_Backbone(nn.Module):
    def __init__(self, args):
        super(STGNN_Backbone, self).__init__()
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        B, N, H = x.shape
        x = x.reshape(B * N, 1, H)
        x = self.tcn(x)
        x = x.reshape(B, N, H)
        x = self.gcn2(x, adj)
        return x


class DCRNN_Backbone(nn.Module):
    def __init__(self, args):
        super(DCRNN_Backbone, self).__init__()
        self.diffusion_conv_forward = BatchGCNConv(
            args.gcn["in_channel"], args.gcn["hidden_channel"] // 2, bias=True, gcn=False
        )
        self.diffusion_conv_backward = BatchGCNConv(
            args.gcn["in_channel"], args.gcn["hidden_channel"] // 2, bias=True, gcn=False
        )
        self.gru_cell = nn.GRUCell(args.gcn["hidden_channel"], args.gcn["hidden_channel"])
        self.diffusion_conv_out = BatchGCNConv(
            args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False
        )

    def forward(self, x, adj):
        B, N, _ = x.shape
        backward_adj = adj.transpose(0, 1)
        forward_diff = F.relu(self.diffusion_conv_forward(x, adj))
        backward_diff = F.relu(self.diffusion_conv_backward(x, backward_adj))
        diff_features = torch.cat([forward_diff, backward_diff], dim=-1)
        diff_features_flat = diff_features.reshape(B * N, -1)
        h = torch.zeros_like(diff_features_flat)
        h = self.gru_cell(diff_features_flat, h)
        h = h.reshape(B, N, -1)
        x = self.diffusion_conv_out(h, adj)
        return x


class ASTGNN_Backbone(nn.Module):
    def __init__(self, args):
        super(ASTGNN_Backbone, self).__init__()
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.attention_layer = nn.Sequential(
            nn.Linear(args.gcn["in_channel"], args.gcn["hidden_channel"]),
            nn.ReLU(),
            nn.Linear(args.gcn["hidden_channel"], 1),
        )
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )

    def _compute_adaptive_adj(self, x, adj):
        B, N, Fdim = x.shape
        x_flat = x.reshape(B * N, Fdim)
        attention_scores = self.attention_layer(x_flat).squeeze(-1).reshape(B, N)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_adj = adj.unsqueeze(0) * attention_weights.unsqueeze(-1)
        return weighted_adj.mean(dim=0)

    def forward(self, x, adj):
        B, N, _ = x.shape
        adaptive_adj = self._compute_adaptive_adj(x, adj)
        x = F.relu(self.gcn1(x, adaptive_adj))
        x = x.reshape(B * N, 1, -1)
        x = self.tcn(x)
        x = x.reshape(B, N, -1)
        x = self.gcn2(x, adaptive_adj)
        return x


class TGCN_Backbone(nn.Module):
    def __init__(self, args):
        super(TGCN_Backbone, self).__init__()
        self.args = args
        self.gcn = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.input_size = args.gcn["hidden_channel"]
        self.hidden_size = args.gcn["hidden_channel"]
        self.weight_xz = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hz = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_xr = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hr = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_xh = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, args.gcn["out_channel"])
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
        x_gcn = self.gcn(x, adj)
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h))
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))
        h = (1 - z) * h + z * h_tilde
        return self.output_layer(h)


# -----------------------------------------------
# Retrain wrappers: one `Model` per backbone
# -----------------------------------------------

def _count_params_log(self):
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    log = self.args.logger.info if hasattr(self.args, "logger") else print
    log(f"Total Parameters: {total_params}")
    log(f"Trainable Parameters: {trainable_params}")


class STGNN_Model(nn.Module):
    def __init__(self, args):
        super(STGNN_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = STGNN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class DCRNN_Model(nn.Module):
    def __init__(self, args):
        super(DCRNN_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = DCRNN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class ASTGNN_Model(nn.Module):
    def __init__(self, args):
        super(ASTGNN_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = ASTGNN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class TGCN_Model(nn.Module):
    def __init__(self, args):
        super(TGCN_Model, self).__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.gcn = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.input_size = args.gcn["hidden_channel"]
        self.hidden_size = args.gcn["hidden_channel"]
        self.weight_xz = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hz = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_xr = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hr = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_xh = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, args.y_len)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        batch_size, num_nodes, _ = x.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
        x_gcn = self.gcn(x, adj)
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h))
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))
        h = (1 - z) * h + z * h_tilde
        feat = h.reshape(-1, self.hidden_size)
        output = self.output_layer(feat)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        batch_size, num_nodes, _ = x.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
        x_gcn = self.gcn(x, adj)
        z = self.sigmoid(self.weight_xz(x_gcn) + self.weight_hz(h))
        r = self.sigmoid(self.weight_xr(x_gcn) + self.weight_hr(h))
        h_tilde = self.activation(self.weight_xh(x_gcn) + self.weight_hh(r * h))
        h = (1 - z) * h + z * h_tilde
        return h.reshape(-1, self.hidden_size)


def _select_backbone(args):
    """Factory used by PECPM / STRAP(RAP) to pick one of four backbones."""
    btype = getattr(args, "backbone_type", "stgnn")
    backbones = {
        "stgnn": STGNN_Backbone,
        "dcrnn": DCRNN_Backbone,
        "astgnn": ASTGNN_Backbone,
        "tgcn": TGCN_Backbone,
    }
    if btype not in backbones:
        raise ValueError(
            f"Unsupported backbone_type={btype!r}. "
            f"PECPM/RAP support only: {sorted(backbones)}. "
            "Use --method for retrain-only baselines such as GWN/STID/iTransformer/DLinear."
        )
    return backbones[btype](args), btype


# -----------------------------------------------
# PECPM (KDD'23, re-implemented by STRAP authors)
# -----------------------------------------------

class PECPM_Model(nn.Module):
    def __init__(self, args):
        super(PECPM_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout

        if not hasattr(args, "attention_weight"):
            self.top_k = 5
        elif isinstance(args.attention_weight, dict):
            self.top_k = 5
        else:
            self.top_k = args.attention_weight

        self.historical_patterns = None

        self.backbone, btype = _select_backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        logger = getattr(args, "logger", None)
        if logger:
            logger.info(f"PECPM initialized with backbone {btype}")

    count_parameters = _count_params_log

    def pattern_matching(self, current_features):
        # Cap history *before* the matmul. Otherwise the first batch of a year
        # stores all B*N rows, and batch 2 builds a (B*N) x (B*N) similarity
        # matrix on CPU (~331GB at PEMS04 year 2012) and OOMs.
        max_patterns = 1000
        device = current_features.device
        current_detached = current_features.detach()
        # Keep at most max_patterns rows of the new batch when growing history.
        if current_detached.size(0) > max_patterns:
            new_history_rows = current_detached[-max_patterns:].cpu()
        else:
            new_history_rows = current_detached.cpu()

        if self.historical_patterns is None:
            self.historical_patterns = new_history_rows
            return torch.ones(current_features.size(0), 1, device=device)

        if hasattr(self.args, "attention_weight") and isinstance(self.args.attention_weight, dict):
            year_offset = str(self.args.year - self.args.begin_year)
            self.top_k = self.args.attention_weight.get(year_offset, 5)

        # Run similarity on GPU: history is now bounded to <= max_patterns rows,
        # so moving it to the GPU per batch is cheap.
        history_gpu = self.historical_patterns.to(device, non_blocking=True)
        current_norm = F.normalize(current_detached, p=2, dim=1)
        history_norm = F.normalize(history_gpu, p=2, dim=1)
        similarity = torch.mm(current_norm, history_norm.t())
        topk_values, _ = similarity.topk(min(self.top_k, similarity.size(1)), dim=1)
        pattern_scores = topk_values.mean(dim=1).unsqueeze(1)

        self.historical_patterns = torch.cat([self.historical_patterns, new_history_rows], dim=0)
        if self.historical_patterns.size(0) > max_patterns:
            self.historical_patterns = self.historical_patterns[-max_patterns:]
        return pattern_scores

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        pattern_scores = self.pattern_matching(feat)
        enhanced = feat * pattern_scores
        x = enhanced + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        pattern_scores = self.pattern_matching(feat)
        enhanced = feat * pattern_scores
        return enhanced + data.x


# -----------------------------------------------
# Simplified STRAP retrieval module (+ helpers)
# -----------------------------------------------

class _PatternLibraryManager:
    """Year-keyed pattern store with optional on-disk cache under args.path/pattern_libraries/."""

    def __init__(self, args):
        self.base_dir = None
        if hasattr(args, "path") and args.path:
            self.base_dir = os.path.join(args.path, "pattern_libraries")
            os.makedirs(self.base_dir, exist_ok=True)
        self._cache = {}

    def _key(self, year, ptype):
        return f"{int(year)}::{ptype}"

    def _file(self, year, ptype):
        if self.base_dir is None:
            return None
        return os.path.join(self.base_dir, f"{int(year)}_{ptype}.pkl")

    def get(self, year, ptype="spatiotemporal"):
        k = self._key(year, ptype)
        if k in self._cache:
            return self._cache[k]
        fp = self._file(year, ptype)
        if fp and os.path.exists(fp):
            with open(fp, "rb") as f:
                data = pickle.load(f)
            self._cache[k] = data
            return data
        return None

    def update(self, year, data, meta=None, ptype="spatiotemporal"):
        k = self._key(year, ptype)
        payload = {"patterns": data.get("patterns", []), "values": data.get("values", []), "metadata": meta or {}}
        self._cache[k] = payload
        fp = self._file(year, ptype)
        if fp:
            with open(fp, "wb") as f:
                pickle.dump(payload, f)
        return True


class _RandomProjection(nn.Module):
    def __init__(self, input_dim, output_dim, seed=42):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        w = torch.randn(input_dim, output_dim, generator=g) / max(output_dim, 1) ** 0.5
        self.register_buffer("weight", w)

    def forward(self, x):
        return x @ self.weight


class STRAP(nn.Module):
    """Simplified STRAP retrieval module (ported from STRAP-main/src/model/model.py)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        gcn_cfg = getattr(args, "gcn", {})
        if isinstance(gcn_cfg, dict):
            self.feature_dim = gcn_cfg.get("hidden_channel", gcn_cfg.get("out_channel", 64))
        else:
            self.feature_dim = getattr(gcn_cfg, "hidden_channel", getattr(gcn_cfg, "out_channel", 64))

        self.k_neighbors = int(getattr(args, "k_neighbors", 16))
        self.max_patterns = int(getattr(args, "max_patterns", 2048))
        self.fusion_weight = float(getattr(args, "fusion_weight", 0.7))

        self.pattern_manager = _PatternLibraryManager(args)
        self.current_year = None
        self.projector = None

        self.patterns = {"spatiotemporal": None}
        self.values = {"spatiotemporal": None}

    def _ensure_projector(self, input_dim, device):
        if self.projector is None or self.projector.weight.shape[0] != input_dim:
            self.projector = _RandomProjection(input_dim, self.feature_dim).to(device)

    def _to_tensor(self, a, device):
        if isinstance(a, torch.Tensor):
            return a.to(device=device, dtype=torch.float32)
        return torch.tensor(a, device=device, dtype=torch.float32)

    def _normalize(self, x):
        return F.normalize(x, dim=-1, eps=1e-8)

    def _build_from_data(self, data):
        x = data.x if isinstance(data.x, torch.Tensor) else torch.tensor(data.x, dtype=torch.float32)
        x = x.detach().float()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        self._ensure_projector(x.shape[-1], x.device)
        feats = self._normalize(self.projector(x))
        if feats.shape[0] > self.max_patterns:
            idx = torch.randperm(feats.shape[0], device=feats.device)[: self.max_patterns]
            feats = feats[idx]
        values = feats.clone()
        return feats.cpu(), values.cpu()

    def switch_to_year(self, year):
        lib = self.pattern_manager.get(year, "spatiotemporal")
        if lib is None:
            return False
        self.current_year = int(year)
        self.patterns["spatiotemporal"] = lib["patterns"]
        self.values["spatiotemporal"] = lib["values"]
        return True

    def extract_patterns(self, data, adj=None, year=None):
        if year is None:
            year = getattr(self.args, "year", None)
        if year is None:
            return False
        patterns, values = self._build_from_data(data)
        meta = {
            "method": "simplified_strap",
            "num_patterns": int(patterns.shape[0]),
            "feature_dim": int(patterns.shape[1]),
        }
        self.pattern_manager.update(year, {"patterns": patterns, "values": values}, meta, "spatiotemporal")
        self.current_year = int(year)
        self.patterns["spatiotemporal"] = patterns
        self.values["spatiotemporal"] = values
        return True

    def _retrieve(self, query):
        patterns = self.patterns["spatiotemporal"]
        values = self.values["spatiotemporal"]
        if patterns is None or values is None:
            return query
        patterns = self._to_tensor(patterns, query.device)
        values = self._to_tensor(values, query.device)
        q_n = self._normalize(query)
        p_n = self._normalize(patterns)
        sim = q_n @ p_n.t()
        k = max(1, min(self.k_neighbors, sim.shape[1]))
        topk_val, topk_idx = torch.topk(sim, k=k, dim=1)
        neighbor_values = values[topk_idx]
        weights = F.softmax(topk_val, dim=1).unsqueeze(-1)
        return (neighbor_values * weights).sum(dim=1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.shape[-1] != self.feature_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.feature_dim).squeeze(1)
        retrieved = self._retrieve(x)
        out = self.fusion_weight * x + (1.0 - self.fusion_weight) * retrieved
        mode = getattr(self.args, "return_pattern_or_value", "value")
        if mode == "pattern":
            return self._normalize(out)
        return out


# -----------------------------------------------
# RAP_Model (= STRAP, NeurIPS'25)
# -----------------------------------------------

class RAP_Model(nn.Module):
    def __init__(self, args):
        super(RAP_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout

        self.backbone, btype = _select_backbone(args)

        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.use_strap = getattr(args, "use_strap", True)
        if self.use_strap:
            self.strap = STRAP(args)
            if hasattr(args, "path"):
                self.pattern_dir = os.path.join(args.path, "pattern_libraries")
                os.makedirs(self.pattern_dir, exist_ok=True)
            self.strap_adapter = nn.Linear(args.gcn["out_channel"], self.strap.feature_dim)
            setattr(args, "return_pattern_or_value", "value")

        self.current_year = getattr(args, "year", None)
        self.pattern_initialized = False

        logger = getattr(args, "logger", None)
        msg = f"RAP initialized with backbone {btype} and year {self.current_year}"
        (logger.info if logger else print)(msg)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log = self.args.logger.info if hasattr(self.args, "logger") else print
        log(f"Total Parameters: {total}")
        log(f"Trainable Parameters: {trainable}")
        if self.use_strap:
            sp = sum(p.numel() for p in self.strap.parameters() if p.requires_grad)
            log(f"strap Parameters: {sp}")

    def load_state_dict(self, state_dict, strict=True):
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)

        state_dict = dict(state_dict)
        has_strap_keys = any(
            k.startswith("strap_adapter") or k.startswith("strap.") for k in state_dict
        )

        # Pure-backbone checkpoint (e.g. retrain_stgnn first-year pkl as the year-N
        # starting point): strap.* / strap_adapter.* are RAP-only and keep their
        # freshly-initialized values.
        if not has_strap_keys:
            return super().load_state_dict(state_dict, strict=False)

        # strap.projector is built lazily on first forward, so its buffer may be
        # present-or-absent in either the source dict or the model's own state.
        # Reconcile both directions so strict=True still catches real typos elsewhere.
        own = self.state_dict()
        src_has = "strap.projector.weight" in state_dict
        own_has = "strap.projector.weight" in own
        if src_has and not own_has:
            state_dict.pop("strap.projector.weight")
        elif own_has and not src_has:
            state_dict["strap.projector.weight"] = own["strap.projector.weight"]

        return super().load_state_dict(state_dict, strict=strict)

    def initialize_patterns(self, data, adj, force=False):
        if not self.use_strap:
            return False
        year = self.current_year if self.current_year is not None else getattr(self.args, "year", None)
        if year is None:
            return False
        has_lib = self.strap.switch_to_year(year)
        if not has_lib or force:
            ok = self.strap.extract_patterns(data, adj, year)
            self.pattern_initialized = bool(ok)
            return self.pattern_initialized
        self.pattern_initialized = True
        return True

    def set_year(self, year):
        self.current_year = year
        if self.use_strap:
            has_lib = self.strap.switch_to_year(year)
            self.pattern_initialized = has_lib
        return self.pattern_initialized

    def _prepare_strap(self, data, adj):
        if self.use_strap and not self.pattern_initialized and self.training:
            adj_device = adj.device
            adj_cpu = adj.cpu()
            self.initialize_patterns(data, adj_cpu)
            adj = adj.to(adj_device)
        return adj

    def _apply_strap(self, feature_mid):
        if not (self.use_strap and self.pattern_initialized):
            return feature_mid
        try:
            if self.current_year is not None:
                self.strap.switch_to_year(self.current_year)
            B, N, C = feature_mid.shape
            flat = feature_mid.reshape(-1, C)
            adapted = self.strap_adapter(flat)
            self.args.return_pattern_or_value = "value"
            enhanced = self.strap(adapted)
            return enhanced.reshape(B, N, -1)
        except Exception as e:
            print(f"STRAP application error: {e}")
            return feature_mid

    def forward(self, data, adj):
        adj = self._prepare_strap(data, adj)
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat_mid = self.backbone(x, adj)
        enhanced = self._apply_strap(feat_mid)
        feat_out = enhanced.reshape(-1, enhanced.shape[-1])
        if feat_out.shape[-1] != self.args.gcn["out_channel"]:
            feat_out = F.adaptive_avg_pool1d(
                feat_out.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        x = feat_out + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat_mid = self.backbone(x, adj)
        enhanced = self._apply_strap(feat_mid)
        feat_out = enhanced.reshape(-1, enhanced.shape[-1])
        if feat_out.shape[-1] != self.args.gcn["out_channel"]:
            feat_out = F.adaptive_avg_pool1d(
                feat_out.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        return feat_out + data.x


# ============================================================================
# Conventional spatio-temporal forecasting baselines from the STBP paper
# (ICLR 2026, Liu & Zhang). Implementations are **lightweight** to match the
# existing pipeline contract (`forward(data, adj) -> [B*N, y_len]`) — same
# philosophy as the simplified STGNN/DCRNN/T-GCN backbones above. They are
# intended as fair-comparison continual-baselines, not faithful reproductions
# of the original papers' full architectures.
#
#   GWN_Model         -- Graph WaveNet (Wu et al. IJCAI 2019), simplified
#   STID_Model        -- STID (Shao et al. CIKM 2022), node-MLP variant
#   ITRANSFORMER_Model-- iTransformer (Liu et al. ICLR 2024), N-as-tokens
#   DLINEAR_Model     -- DLinear (Zeng et al. AAAI 2023), trend+seasonal
# ============================================================================


class GWN_Backbone(nn.Module):
    """Graph WaveNet (Wu et al. IJCAI 2019), retrain-mode reimplementation.

    Faithful to the GWN architecture:
      - 1x1 start conv embeds the raw time history into residual channels.
      - Stacked ST-blocks. Each block applies a *gated dilated TCN on the
        time axis* (filter = tanh, gate = sigmoid, multiplied), then a GCN
        with an adaptive adjacency `softmax(relu(E1 @ E2.T))` blended with
        the input static adjacency, then a residual + BN.
      - A skip path from each block accumulates into the final output, which
        goes through two 1x1 convs to produce the prediction tensor.

    Simplifications vs the original (kept within the existing parameter
    budget of ~10-30k for fair baseline comparison):
      - 2 ST-blocks with dilations [1, 2] instead of 4-8 blocks with
        dilations [1,2,1,2,1,2,1,2].
      - residual_channels = hidden_channel // 2.

    Streaming-mode caveat:
      `E1, E2 ∈ R^{args.graph_size × adp_dim}` and N may change year-to-year.
      This model is therefore only safe under `strategy: retrain` (fresh
      model each period). The existing trainer guarantees this when
      `args.init == False` (see default_trainer.py:117).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        T_out = args.gcn["out_channel"]
        residual_channels = max(args.gcn["hidden_channel"] // 2, 16)
        skip_channels = args.gcn["hidden_channel"]
        kernel_size = 2
        dilations = [1, 2]
        adp_dim = 10
        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "GWN_Backbone needs args.graph_size set at __init__ time "
                "(per-year node count). The main.py loop sets it before "
                "instantiating the model."
            )

        # Learnable adaptive-adjacency embeddings (the GWN signature feature).
        self.E1 = nn.Parameter(torch.randn(N, adp_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, adp_dim) * 0.01)

        # [B, 1, N, T] -> [B, C_r, N, T]
        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for d in dilations:
            self.filter_convs.append(
                nn.Conv2d(residual_channels, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.gate_convs.append(
                nn.Conv2d(residual_channels, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.skip_convs.append(
                nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
            )
            self.gcn_layers.append(
                BatchGCNConv(residual_channels, residual_channels, bias=True, gcn=False)
            )
            self.bn.append(nn.BatchNorm2d(residual_channels))
            receptive_field += (kernel_size - 1) * d

        self.end_conv1 = nn.Conv2d(skip_channels, skip_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(skip_channels, T_out, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def forward(self, x, adj):
        # x: [B, N, T]  T = args.gcn["in_channel"] = 12
        T = x.shape[-1]
        x = x.unsqueeze(1)  # [B, 1, N, T]
        if T < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T, 0))

        x = self.start_conv(x)  # [B, C_r, N, T_pad]

        # Adaptive adjacency, blended with the input static adj.
        adp = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        adj_eff = 0.5 * (adj + adp)

        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g  # [B, C_r, N, T_block]

            # Skip path: project to skip_channels, keep last time step.
            s = self.skip_convs[i](x)[..., -1:]  # [B, skip, N, 1]
            skip = s if skip is None else skip + s

            # GCN over space, batched across time steps:
            #   [B, C_r, N, T_block] -> [B*T_block, N, C_r] -> gcn -> reshape back
            B_, C_, N_, Tn = x.shape
            x_g = x.permute(0, 3, 2, 1).reshape(B_ * Tn, N_, C_)
            x_g = self.gcn_layers[i](x_g, adj_eff)
            x_g = x_g.reshape(B_, Tn, N_, C_).permute(0, 3, 2, 1)

            # Residual (trim earlier time steps so shapes match).
            x = x_g + residual[..., -Tn:]
            x = self.bn[i](x)

        # End convs: [B, skip, N, 1] -> [B, T_out, N, 1] -> [B, N, T_out]
        x = F.relu(skip)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        return x.squeeze(-1).permute(0, 2, 1)


class GWN_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = GWN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class STID_Backbone(nn.Module):
    """STID (Shao et al. CIKM 2022), retrain-mode reimplementation.

    Faithful to the STID core idea: combine a time-series embedding with a
    *node-identity embedding* (the paper's key contribution — `E_node ∈
    R^{N×D}`, looked up per node), then process the concatenation with a
    stack of residual MLP blocks.

    Adaptation (vs the original paper):
      - The original additionally concatenates a time-of-day embedding
        (`R^{288×D}`) and a day-of-week embedding (`R^{7×D}`), looked up
        from per-sample timestamp metadata. This pipeline's dataloader
        (`SpatioTemporalDataset`) does not surface timestamps, so we omit
        these two embeddings. Documenting this as a deliberate limitation
        — re-adding them would require dataloader changes.
      - 3 residual MLP blocks (matches the original default).

    Streaming-mode caveat:
      `nn.Embedding(N, D_node)` is sized with `args.graph_size`, so this
      model is only safe under `strategy: retrain` (fresh model each year).
      Incremental loading across years would break because the embedding
      row count would change.
    """

    def __init__(self, args):
        super().__init__()
        H = args.gcn["hidden_channel"]                                  # 64
        D_node = getattr(args, "stid_node_dim", max(H // 2, 16))        # 32
        D_total = H + D_node
        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "STID_Backbone needs args.graph_size set at __init__ time. "
                "Use strategy=retrain so the model is freshly instantiated "
                "for each year's N."
            )

        self.time_embed = nn.Linear(args.gcn["in_channel"], H)
        self.node_embed = nn.Embedding(N, D_node)
        # Fixed-std init instead of xavier-by-fan-in: xavier on shape [N, D_node]
        # gives scale sqrt(6/(N+D)), which collapses to ~0.002 once N>~10k
        # (PEMS03 from 2018 onward), making the node-identity signal
        # numerically invisible vs the H=64 time embedding.
        nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.02)
        # Buffer the node-id range so .to(device) follows the model.
        self.register_buffer("node_ids", torch.arange(N), persistent=False)

        num_blocks = getattr(args, "stid_blocks", 3)
        dropout = getattr(args, "dropout", 0.15)
        # Pre-norm residual MLP block. Without LayerNorm the residual stream
        # variance compounds across blocks and combined with lr=0.03 + MSE on
        # raw-scale traffic flow this stalls the optimizer (epoch-30 train
        # loss ~5k on some seed/year combos, never reaching the ~10 MAE basin
        # within the 100-epoch budget).
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(D_total),
                    nn.Linear(D_total, D_total),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(D_total, D_total),
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(D_total)
        self.out_proj = nn.Linear(D_total, args.gcn["out_channel"])

    def forward(self, x, adj):
        # x: [B, N, in_channel]
        B, N, _ = x.shape
        time_h = self.time_embed(x)                              # [B, N, H]
        ids = self.node_ids[:N]                                  # [N]
        node_h = self.node_embed(ids).unsqueeze(0).expand(B, -1, -1)  # [B, N, D_node]
        h = torch.cat([time_h, node_h], dim=-1)                  # [B, N, D_total]
        for block in self.blocks:
            h = h + block(h)
        h = self.final_norm(h)
        return self.out_proj(h)


class STID_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = STID_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class ITRANSFORMER_Backbone(nn.Module):
    """iTransformer-style: each node's 12-step history is one token, then
    self-attention across the N node-tokens, then project back to channel dim.

    The adjacency is ignored (matching the original iTransformer's premise of
    not requiring a graph). Uses one transformer encoder layer to keep param
    count modest.
    """

    def __init__(self, args):
        super().__init__()
        H = args.gcn["hidden_channel"]
        self.embed = nn.Linear(args.gcn["in_channel"], H)
        nhead = getattr(args, "itrans_nhead", 4)
        # batch_first=True so input is [B, N, H]
        self.encoder = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=nhead,
            dim_feedforward=H * 2,
            dropout=getattr(args, "dropout", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.out_proj = nn.Linear(H, args.gcn["out_channel"])

    def forward(self, x, adj):
        # x: [B, N, in_channel] — treat each of the N nodes as a token.
        h = self.embed(x)
        h = self.encoder(h)
        return self.out_proj(h)


class ITRANSFORMER_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = ITRANSFORMER_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


class DLINEAR_Backbone(nn.Module):
    """DLinear-style: trend (moving-avg) + seasonal (residual) linear projections.

    Decomposition uses a kernel-3 moving average with reflect padding. Both
    branches share a single linear head shape [in_channel -> out_channel].
    The adjacency is ignored. Parameter count is tiny (two Linear layers).
    """

    def __init__(self, args):
        super().__init__()
        self.kernel = getattr(args, "dlinear_kernel", 3)
        in_c = args.gcn["in_channel"]
        out_c = args.gcn["out_channel"]
        self.trend_linear = nn.Linear(in_c, out_c)
        self.seasonal_linear = nn.Linear(in_c, out_c)

    def _moving_avg(self, x):
        # x: [B, N, L]
        B, N, L = x.shape
        pad = self.kernel // 2
        z = x.reshape(B * N, 1, L)
        # `replicate` instead of `reflect`: the CUDA kernel for reflect-pad
        # raises "invalid configuration argument" once B*N exceeds the grid
        # block limit on large datasets (e.g. PEMS03 from year 2007 onwards).
        # For kernel=3 the boundary contribution is one sample on each side,
        # so replicate vs reflect changes the trend by < O(1/L).
        z = F.pad(z, (pad, pad), mode="replicate")
        z = F.avg_pool1d(z, kernel_size=self.kernel, stride=1, padding=0)
        return z.reshape(B, N, L)

    def forward(self, x, adj):
        trend = self._moving_avg(x)
        seasonal = x - trend
        return self.trend_linear(trend) + self.seasonal_linear(seasonal)


class DLINEAR_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = DLINEAR_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


# ============================================================================
# ST-Norm (Deng et al., KDD 2021, https://github.com/JLDeng/ST-Norm)
# Lightweight retrain-mode reimplementation. The contribution of ST-Norm is
# two plug-in normalization modules (SNorm + TNorm) layered into a Wavenet-
# style dilated-TCN base. No GCN / no adjacency: ST-Norm's original "Wavenet"
# variant in the official repo does not use the graph at all, so adj is
# ignored at forward time (matches iTransformer/DLinear in this file).
#
#   STNORM_Model -- TCN with SNorm (across N) + TNorm (across T)
# ============================================================================


class _SNorm(nn.Module):
    """Spatial Normalization (Deng et al., KDD 2021, Eq. 4).

    Standardizes each (batch, channel, time) coordinate across the N spatial
    nodes, then applies a per-channel learnable affine. Stateless — every
    forward pass recomputes mean/var, so this module is N-agnostic and works
    unchanged when the per-year node count changes.
    """

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, N, T]
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class _TNorm(nn.Module):
    """Temporal Normalization (Deng et al., KDD 2021, Eq. 5).

    BatchNorm-style normalization along the temporal axis (also reducing over
    the batch axis to stabilize per-node statistics). Tracks per-(channel,
    node) running mean/var so eval uses fixed stats. Per-(channel, node)
    learnable affine. Sized to args.graph_size at construction.
    """

    def __init__(self, num_nodes, channels, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_mean", torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_var", torch.ones(1, channels, num_nodes, 1))

    def forward(self, x):
        # x: [B, C, N, T]
        if self.training:
            mean = x.mean(dim=(0, 3), keepdim=True)
            var = x.var(dim=(0, 3), keepdim=True, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm * self.gamma + self.beta


class STNORM_Backbone(nn.Module):
    """ST-Norm (Deng et al., KDD 2021), retrain-mode reimplementation.

    Architecture (faithful to the ST-Norm/Wavenet variant in the official
    repo, https://github.com/JLDeng/ST-Norm):
      - 1x1 start conv embeds the raw time history into residual channels.
      - Stacked ST-blocks. Each block computes SNorm(x) and TNorm(x), then
        *concatenates* [x, SNorm(x), TNorm(x)] along the channel axis as
        input to a gated dilated TCN (filter = tanh, gate = sigmoid).
      - A skip path from each block accumulates into the final output, then
        two 1x1 convs project to the prediction tensor.

    Simplifications vs the original (kept within the same parameter budget as
    the existing GWN baseline for fair comparison):
      - 2 ST-blocks with dilations [1, 2] (original: 8 blocks).
      - residual_channels = hidden_channel // 2.

    Streaming-mode caveat:
      `_TNorm` holds per-node learnable affine + running-stat buffers of
      shape [1, C, N, 1], and N changes year-to-year on xxltrafficdata. This
      model is therefore only safe under `strategy: retrain` (fresh model per
      period), matching GWN / STID / iTransformer / DLinear.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        T_out = args.gcn["out_channel"]
        residual_channels = max(args.gcn["hidden_channel"] // 2, 16)
        skip_channels = args.gcn["hidden_channel"]
        kernel_size = 2
        dilations = [1, 2]
        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "STNORM_Backbone needs args.graph_size set at __init__ time "
                "(per-year node count). Use strategy=retrain so the model "
                "is freshly instantiated for each year's N."
            )

        # Filter/gate convs see [residual || SNorm || TNorm] concatenated.
        norm_in = residual_channels * 3

        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))

        self.snorms = nn.ModuleList()
        self.tnorms = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for d in dilations:
            self.snorms.append(_SNorm(residual_channels))
            self.tnorms.append(_TNorm(N, residual_channels))
            self.filter_convs.append(
                nn.Conv2d(norm_in, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.gate_convs.append(
                nn.Conv2d(norm_in, residual_channels,
                          kernel_size=(1, kernel_size), dilation=(1, d))
            )
            self.skip_convs.append(
                nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
            )
            self.bn.append(nn.BatchNorm2d(residual_channels))
            receptive_field += (kernel_size - 1) * d

        self.end_conv1 = nn.Conv2d(skip_channels, skip_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(skip_channels, T_out, kernel_size=(1, 1))
        self.receptive_field = receptive_field

    def forward(self, x, adj):
        # x: [B, N, T] — adj ignored (ST-Norm is non-graph TCN).
        T = x.shape[-1]
        x = x.unsqueeze(1)  # [B, 1, N, T]
        if T < self.receptive_field:
            x = F.pad(x, (self.receptive_field - T, 0))

        x = self.start_conv(x)  # [B, C_r, N, T_pad]

        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            x_cat = torch.cat([x, self.snorms[i](x), self.tnorms[i](x)], dim=1)
            f = torch.tanh(self.filter_convs[i](x_cat))
            g = torch.sigmoid(self.gate_convs[i](x_cat))
            x = f * g  # [B, C_r, N, T_block]

            s = self.skip_convs[i](x)[..., -1:]  # [B, skip, N, 1]
            skip = s if skip is None else skip + s

            # Residual (trim earlier time steps so shapes match).
            x = x + residual[..., -x.shape[-1]:]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        return x.squeeze(-1).permute(0, 2, 1)


class STNORM_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = STNORM_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x


# ============================================================================
# STAEformer (Liu et al., CIKM 2023, https://github.com/XDZhelheim/STAEformer)
# "Spatio-Temporal Adaptive Embedding makes Vanilla Transformer SOTA for
# Traffic Forecasting".
# Lightweight retrain-mode reimplementation, no GCN / no adjacency (the paper
# claims a pure-Transformer baseline). The contribution is the learnable
# spatio-temporal adaptive embedding `E ∈ R^{T × N × D}` concatenated with the
# input projection. We omit the optional time-of-day / day-of-week embeddings
# (same rationale as STID in this file): the dataloader does not surface
# timestamps.
#
#   STAEFORMER_Model -- input proj + STAE + alternating temporal/spatial attn
# ============================================================================


class _STAE_AttentionLayer(nn.Module):
    """Multi-head attention along the -2 dim of an n-D tensor.

    Mathematically equivalent to STAEformer/model/STAEformer.py::AttentionLayer
    but the QK^T attention matrix is never explicitly materialized — we route
    through `F.scaled_dot_product_attention` (PyTorch 2.0+), which uses
    FlashAttention on CUDA and brings memory from O(L^2) down to O(L).

    Why: spatial attention on xxltrafficdata has L = N (per-year node count,
    up to 3000+). The naive QK^T tensor of shape (heads, B, T, N, N) is
    O(N^2) and OOMs hard at N≥1000 even on 24GB GPUs — for N=3000 the
    attention matrix alone is ~200 GB. SDPA tiles the computation and keeps
    peak memory linear in N.
    """

    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        if not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError(
                "STAEformer port requires torch.nn.functional.scaled_dot_product_attention "
                "(PyTorch >= 2.0). Upgrade torch or remove STAEFORMER from --method."
            )
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Inputs: (..., L, model_dim). SDPA expects (..., heads, L, head_dim);
        # broadcasting over any leading dims is fine.
        q = self.FC_Q(query)
        k = self.FC_K(key)
        v = self.FC_V(value)

        L_q = q.shape[-2]
        L_k = k.shape[-2]
        leading = q.shape[:-2]

        # Reshape last dim into (heads, head_dim), then move heads to dim -3
        # so SDPA attends over L_q/L_k (which sits at dim -2 after the move).
        q = q.reshape(*leading, L_q, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*leading, L_k, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*leading, L_k, self.num_heads, self.head_dim).transpose(-3, -2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # (..., heads, L_q, head_dim) -> (..., L_q, heads*head_dim)
        out = out.transpose(-3, -2).reshape(*leading, L_q, self.model_dim)
        return self.out_proj(out)


class _STAE_SelfAttentionLayer(nn.Module):
    """Pre-LN-style transformer encoder block (LN-after-add variant, matching
    upstream). `dim` argument selects which axis to attend across (1=time,
    2=node).
    """

    def __init__(self, model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = _STAE_AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out


class STAEFORMER_Backbone(nn.Module):
    """STAEformer (Liu et al., CIKM 2023), retrain-mode reimplementation.

    Architecture (faithful to upstream `model/STAEformer.py`):
      - Input projection: Linear(1 → input_embedding_dim).
      - Spatio-temporal adaptive embedding `E ∈ R^{T × N × adp_dim}` (the
        paper's key contribution), xavier-initialized.
      - Concatenate [input_emb, adp_emb] along feature axis →
        model_dim = input_embedding_dim + adp_embedding_dim.
      - `num_layers` stacked temporal-attention blocks (attend across T),
        then `num_layers` spatial-attention blocks (attend across N).
      - Mixed projection: reshape (B, N, T*model_dim) → Linear → (B, N, T_out).

    Differences vs upstream defaults (kept reasonable for the existing
    parameter budget across baselines; tunable via args.staeformer_*):
      - tod_embedding_dim=0, dow_embedding_dim=0 — the dataloader does not
        surface timestamps in this pipeline (same omission as STID).
      - spatial_embedding_dim=0 (matches upstream default).
      - input_embedding_dim=24, adaptive_embedding_dim=80 → model_dim=104.
      - num_heads=4, num_layers=3, feed_forward_dim=256, dropout=0.1.

    Streaming-mode caveat:
      `adaptive_embedding` and the mixed-projection output head are both
      sized to args.graph_size at __init__. N changes year-to-year on
      xxltrafficdata, so this model is retrain-only (strategy: retrain),
      matching GWN / STID / iTransformer / DLinear / STNORM.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        N = int(getattr(args, "graph_size", 0))
        if N <= 0:
            raise ValueError(
                "STAEFORMER_Backbone needs args.graph_size set at __init__ time. "
                "Use strategy=retrain so the model is freshly instantiated "
                "for each year's N."
            )
        in_steps = args.gcn["in_channel"]   # 12
        out_steps = args.gcn["out_channel"]  # 12
        input_dim = 1
        input_embedding_dim = int(getattr(args, "staeformer_input_emb", 24))
        adaptive_embedding_dim = int(getattr(args, "staeformer_adp_emb", 80))
        feed_forward_dim = int(getattr(args, "staeformer_ff_dim", 256))
        num_heads = int(getattr(args, "staeformer_num_heads", 4))
        num_layers = int(getattr(args, "staeformer_num_layers", 3))
        dropout = float(getattr(args, "dropout", 0.1))

        model_dim = input_embedding_dim + adaptive_embedding_dim
        if model_dim % num_heads != 0:
            raise ValueError(
                f"STAEFORMER: model_dim={model_dim} not divisible by "
                f"num_heads={num_heads}. Tune staeformer_input_emb / "
                f"staeformer_adp_emb / staeformer_num_heads."
            )

        self.in_steps = in_steps
        self.out_steps = out_steps
        self.model_dim = model_dim
        self.num_nodes = N

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(in_steps, N, adaptive_embedding_dim)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        self.attn_layers_t = nn.ModuleList(
            [
                _STAE_SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                _STAE_SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Mixed projection (the upstream `use_mixed_proj=True` head).
        self.output_proj = nn.Linear(in_steps * model_dim, out_steps)

    def forward(self, x, adj):
        # x: [B, N, T_in] — adj ignored (STAEformer is non-graph).
        B, N, T = x.shape
        # Reshape to upstream format: (B, T_in, N, input_dim=1)
        z = x.transpose(1, 2).unsqueeze(-1)
        z = self.input_proj(z)  # (B, T_in, N, input_emb)
        adp = self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1)
        z = torch.cat([z, adp], dim=-1)  # (B, T_in, N, model_dim)

        for attn in self.attn_layers_t:
            z = attn(z, dim=1)  # attend across time
        for attn in self.attn_layers_s:
            z = attn(z, dim=2)  # attend across nodes

        # Mixed projection: (B, T_in, N, D) -> (B, N, T_in*D) -> Linear -> (B, N, T_out)
        out = z.transpose(1, 2).reshape(B, N, self.in_steps * self.model_dim)
        out = self.output_proj(out)
        return out  # (B, N, T_out)


class STAEFORMER_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, "dropout", 0.1)
        self.backbone = STAEFORMER_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

    count_parameters = _count_params_log

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        x = feat + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))
        feat = self.backbone(x, adj).reshape(-1, self.args.gcn["out_channel"])
        return feat + data.x

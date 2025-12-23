import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Local Feature Encoder
class LocalEncoder(nn.Module):
    def __init__(self):
        super(LocalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16)
            # nn.LeakyReLU()
        )

    def forward(self, xy):
        return self.encoder(xy)
    
class GlobalGraphEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=16, out_dim=128, num_layers=4, max_nodes=200):
        super(GlobalGraphEncoder, self).__init__()
        self.max_nodes = max_nodes

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.embedding_layer = nn.Linear(hidden_dim * max_nodes, 3200)  # Optional if you want an explicit 3200-dim embedding
        self.final_proj = nn.Linear(3200, out_dim)

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = nn.functional.leaky_relu(x)

        if batch is not None and batch > 1:
            # Process each graph in batch individually and stack results
            from torch_geometric.utils import to_dense_batch
            x_padded, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            x_padded = x_padded[:, :self.max_nodes, :]  # [B, max_nodes, hidden_dim]
            x_flat = x_padded.reshape(x_padded.size(0), -1)  # [B, max_nodes * hidden_dim]
        else:
            # Single graph case
            n = x.size(0)
            if n < self.max_nodes:
                pad = torch.zeros(self.max_nodes - n, x.size(1), device=x.device)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:self.max_nodes]
            x_flat = x.reshape(1, -1)  # [1, max_nodes * hidden_dim]

        embed = self.embedding_layer(x_flat)  # [B, 3200]
        return self.final_proj(embed)         # [B, 128]

    

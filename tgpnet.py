import torch.nn as nn
from .encoders import LocalEncoder, GlobalGraphEncoder
from .inference import DepthInferenceModule

# Model wrapper
class TGPNet(nn.Module):
    def __init__(self):
        super(TGPNet, self).__init__()
        self.local_encoder = LocalEncoder()
        self.global_encoder = GlobalGraphEncoder(in_dim=4)  # xyzw
        self.depth_infer = DepthInferenceModule()

    def forward(self, data, current_idx, local_input, batch=None):
        # data is a torch_geometric.data.Data object
        x = data.x           # shape [N, 4] (xyzw)
        edge_index = data.edge_index  # shape [2, num_edges]

        local_feat = self.local_encoder(local_input.unsqueeze(0))  # shape [1, D]
        global_feat = self.global_encoder(x, edge_index, batch=batch)  # shape [1, D]
        return self.depth_infer(local_feat, global_feat)

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from modules.qcnet_agent_encoder import QCNetAgentEncoder
from modules.qcnet_map_encoder import QCNetMapEncoder


class QCNetEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pt2pl_denoise_inference: bool,
                 pl2pl_denoise_inference: bool,
                 cart_denoise_inference: bool
                 ) -> None:
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pl_denoise_inference=pt2pl_denoise_inference,
            pl2pl_denoise_inference=pl2pl_denoise_inference,
            cart_denoise_inference=cart_denoise_inference
        )
        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: HeteroData, cart_noise_pred_net: nn.Module, pt2pl_noise_pred_net: nn.Module, pl2pl_noise_pred_net: nn.Module) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data, cart_noise_pred_net, pt2pl_noise_pred_net, pl2pl_noise_pred_net)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

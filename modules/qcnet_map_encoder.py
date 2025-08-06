from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from modules.uncertainty_conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils import angle_between_2d_vectors
from utils import merge_edges
from utils import weight_init
from utils import wrap_angle
from utils import pyramid_noise_like
from utils import get_s_from_o


class QCNetMapEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pt2pl_denoise_inference: bool,
                 pl2pl_denoise_inference: bool,
                 cart_denoise_inference: bool) -> None:
        super(QCNetMapEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        # denoise step for diffusion
        self.pt2pl_denoise_inference = pt2pl_denoise_inference
        self.pl2pl_denoise_inference = pl2pl_denoise_inference
        self.cart_denoise_inference = cart_denoise_inference

        self.diffusion_inferece_scheduler = DDPMScheduler(
            num_train_timesteps=20,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            trained_betas=None,
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1.0,
            sample_max_value=1.0,
            timestep_spacing="leading",
            steps_offset=0,
            rescale_betas_zero_snr=False,
        )
        
        if dataset == 'argoverse_v2':
            if input_dim == 2:
                input_dim_x_pt = 1
                input_dim_x_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError('{} is not a valid dimension'.format(input_dim))
            
        elif dataset == 'nuscenes':
                # dim onlt have 2 in nuscenes
                input_dim_x_pt = 1
                input_dim_x_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
        elif dataset == 'nuscenes':
            # TODO: check the embedding size
            self.type_pt_emb = nn.Embedding(17, hidden_dim)
            self.side_pt_emb = nn.Embedding(3, hidden_dim)
            self.type_pl_emb = nn.Embedding(4, hidden_dim)
            self.int_pl_emb = nn.Embedding(3, hidden_dim)
        else:    
            raise ValueError('{} is not a valid dataset'.format(dataset))
        
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)


        # temporal usage
        # self.pt2pl_noise_pred_net = ConditionalUnet1D(
        #     input_dim=3,
        #     local_cond_dim=3,
        #     # global_cond_dim=256,
        #     down_dims=[128,256],
        #     cond_predict_scale=False,
        #     )
        
        # self.pl2pl_noise_pred_net = ConditionalUnet1D(
        #     input_dim=3,
        #     local_cond_dim=3,
        #     # global_cond_dim=256,
        #     down_dims=[128,256],
        #     cond_predict_scale=False,
        #     )
        
        # self.cart_noise_pred_net = ConditionalUnet1D(
        #     input_dim=2,
        #     local_cond_dim=2,
        #     # global_cond_dim=256,
        #     down_dims=[128,256],
        #     cond_predict_scale=False,
        #     )


    def forward(self, data: HeteroData, cart_noise_pred_net: nn.Module, pt2pl_noise_pred_net: nn.Module, pl2pl_noise_pred_net: nn.Module) -> Dict[str, torch.Tensor]:
    # def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:

        if self.dataset == 'argoverse_v2':
            pos_pt = data['map_point']['position'][:, :self.input_dim].contiguous()
            orient_pt = data['map_point']['orientation'].contiguous()
            pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
            orient_pl = data['map_polygon']['orientation'].contiguous()
            orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

        elif self.dataset == 'nuscenes':

            # convert to cartesian coordinate system
            ## online map
            cart_online_map_point_list = []
            for i in range(data['online_map_polygon']['position'].shape[0]):
                cart_online_map_point_list.append(data['online_map_point']['position']\
                                                [get_s_from_o(data[('online_map_point', 'to', 'online_map_polygon')]['edge_index'], i)])
            cart_online_map_point = torch.stack(cart_online_map_point_list)

            ## gt map
            cart_gt_map_point_list = []
            for i in range(data['maptr_gt_map_polygon']['position'].shape[0]):
                cart_gt_map_point_list.append(data['maptr_gt_map_point']['position']\
                                                [get_s_from_o(data[('maptr_gt_map_point', 'to', 'maptr_gt_map_polygon')]['edge_index'], i)])
            cart_gt_map_point = torch.stack(cart_gt_map_point_list)

            # send to diffusion
            cart_diffusion_original_samples = cart_gt_map_point
            cart_diffusion_local_cond = cart_online_map_point

            # whether using diffusion module for cartesian coordinates
            if self.cart_denoise_inference:

                # diffusion denoise      
                if cart_online_map_point.shape[1] % 2 != 0:
                    # padding = torch.randn(cart_online_map_point.shape[0], 1, cart_online_map_point.shape[2])
                    padding = cart_online_map_point[:, -1:, :] 
                    padding = padding.to(cart_online_map_point.device)
                    cart_online_map_point = torch.cat((cart_online_map_point, padding), dim=1)
                    padding_flag = True

                cart_noise = pyramid_noise_like(cart_online_map_point)
                cart_diffusion_output = cart_noise

                # not use grad
                for param in cart_noise_pred_net.parameters():
                    param.requires_grad = False                

                for k in self.diffusion_inferece_scheduler.timesteps[:]:

                    cart_diffusion_output = self.diffusion_inferece_scheduler.scale_model_input(cart_diffusion_output)
                    timesteps = k.unsqueeze(-1).repeat(cart_diffusion_output.shape[0]).to(cart_online_map_point.device)

                    # diffusion freeze the parameters
                    cart_noise_pred = cart_noise_pred_net(cart_noise, timestep=timesteps, local_cond=cart_online_map_point, global_cond=None)

                    cart_diffusion_output = self.diffusion_inferece_scheduler.step(
                                model_output=cart_noise_pred,
                                timestep=k,
                                sample=cart_diffusion_output
                        ).prev_sample
            
                if padding_flag:
                    cart_diffusion_output = cart_diffusion_output[:, :-1, :] 

                # TODO: visualization
                # calculate the orentation
                cart_diffusion_orientation_list = []
                for i in range(cart_diffusion_output.shape[0]):
                    cart_online_map_vectors = cart_diffusion_output[i][1:] - cart_diffusion_output[i][:-1]
                    cart_diffusion_orientation = torch.cat([torch.atan2(cart_online_map_vectors[:, 1], cart_online_map_vectors[:, 0])], dim=0)
                    cart_diffusion_orientation_list.append(cart_diffusion_orientation.unsqueeze(0))
                
                ## orientation needs two points and needs to padding original orientation
                cart_diffusion_orientation = torch.cat(cart_diffusion_orientation_list)
                cart_diffusion_orientation_padding = data['online_map_point']['orientation'].reshape(cart_diffusion_orientation.shape[0], -1)[:,-1].reshape(-1,1)
                cart_diffusion_orientation = torch.cat([cart_diffusion_orientation, cart_diffusion_orientation_padding], dim=1)

                # convert to qcnet input
                pos_pt = cart_diffusion_output.reshape(-1, 2)[:, :self.input_dim].contiguous()
                orient_pt = cart_diffusion_orientation.reshape(-1).contiguous()
                pos_pl = cart_diffusion_output[:, 0, :][:, :self.input_dim].contiguous()
                orient_pl = cart_diffusion_orientation[:,0].contiguous()
                orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)
                  
            else:
                pos_pt = data['online_map_point']['position'][:, :self.input_dim].contiguous()
                orient_pt = data['online_map_point']['orientation'].contiguous()
                pos_pl = data['online_map_polygon']['position'][:, :self.input_dim].contiguous()
                orient_pl = data['online_map_polygon']['orientation'].contiguous()
                orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)
 
            pos_pt_gt = data['maptr_gt_map_point']['position'][:, :self.input_dim].contiguous()
            orient_pt_gt = data['maptr_gt_map_point']['orientation'].contiguous()
            pos_pl_gt = data['maptr_gt_map_polygon']['position'][:, :self.input_dim].contiguous()
            orient_pl_gt = data['maptr_gt_map_polygon']['orientation'].contiguous()

            orient_vector_pl_gt = torch.stack([orient_pl_gt.cos(), orient_pl_gt.sin()], dim=-1)      
            
        if self.dataset == 'argoverse_v2':
            if self.input_dim == 2:
                # magnitude means length of map element
                x_pt = data['map_point']['magnitude'].unsqueeze(-1)
                x_pl = None
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
                x_pl = data['map_polygon']['height'].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'].long()),
                                     self.side_pt_emb(data['map_point']['side'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'].long()),
                                     self.int_pl_emb(data['map_polygon']['is_intersection'].long())]
        
        elif self.dataset == 'nuscenes':
            # magnitude means length of map element and input element is 2
            x_pt = data['online_map_point']['magnitude'].unsqueeze(-1)
            x_pl = None

            x_pt_categorical_embs = [self.type_pt_emb(data['online_map_point']['type'].long()),
                                     self.side_pt_emb(data['online_map_point']['side'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['online_map_polygon']['type'].long()),
                                     self.int_pl_emb(data['online_map_polygon']['is_intersection'].long())]

        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)

        if self.dataset == 'argoverse_v2':
            edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index']
        elif self.dataset == 'nuscenes':
            edge_index_pt2pl = data['online_map_point', 'to', 'online_map_polygon']['edge_index']
            edge_index_pt2pl_gt = data['maptr_gt_map_point', 'to', 'maptr_gt_map_polygon']['edge_index']
        else:
            ValueError('not a valid dataset')

        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
        rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])

        if self.dataset == 'nuscenes':
            rel_pos_pt2pl_gt = pos_pt_gt[edge_index_pt2pl_gt[0]] - pos_pl_gt[edge_index_pt2pl_gt[1]]
            rel_orient_pt2pl_gt = wrap_angle(orient_pt_gt[edge_index_pt2pl_gt[0]] - orient_pl_gt[edge_index_pt2pl_gt[1]])

        # realtional 4-d descriptor
        if self.input_dim == 2:

            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                        nbr_vector=rel_pos_pt2pl[:, :2]),
                rel_orient_pt2pl], dim=-1)

            # generating samples for training r_pt2pl diffusion module
            if self.dataset == 'argoverse_v2':
                r_pt2pl_diffusion_original_samples = r_pt2pl
                r_pt2pl_diffusion_local_cond = r_pt2pl
            elif self.dataset == 'nuscenes':
                r_pt2pl_gt = torch.stack(
                    [torch.norm(rel_pos_pt2pl_gt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(ctr_vector=orient_vector_pl_gt[edge_index_pt2pl_gt[1]],
                                            nbr_vector=rel_pos_pt2pl_gt[:, :2]),
                    rel_orient_pt2pl_gt], dim=-1)
                r_pt2pl_diffusion_original_samples = r_pt2pl_gt
                r_pt2pl_diffusion_local_cond = r_pt2pl

            # whether using diffusion module
            if self.pt2pl_denoise_inference:
                r_pt2pl = r_pt2pl.unsqueeze(0)
                padding_flag = False
                if r_pt2pl.shape[1] % 2 != 0:
                    padding = r_pt2pl[:, -1:, :] 
                    padding = padding.to(r_pt2pl.device)
                    r_pt2pl = torch.cat((r_pt2pl, padding), dim=1)
                    padding_flag = True
                                    
                r_pt2pl_noise = pyramid_noise_like(r_pt2pl)
                r_pt2pl_diffusion_output = r_pt2pl_noise

                # not use grad
                for param in pt2pl_noise_pred_net.parameters():
                    param.requires_grad = False   

                for k in self.diffusion_inferece_scheduler.timesteps[:]:
                    r_pt2pl_diffusion_output = self.diffusion_inferece_scheduler.scale_model_input(r_pt2pl_diffusion_output)
                    timesteps = k.unsqueeze(-1).repeat(r_pt2pl_diffusion_output.shape[0]).to(r_pt2pl.device)

                    # diffusion freeze the parameters
                    pt2pl_noise_pred = pt2pl_noise_pred_net(r_pt2pl_noise, timestep=timesteps, local_cond=r_pt2pl, global_cond=None)

                    r_pt2pl_diffusion_output = self.diffusion_inferece_scheduler.step(
                                model_output=pt2pl_noise_pred,
                                timestep=k,
                                sample=r_pt2pl_diffusion_output
                        ).prev_sample
                
                if padding_flag:
                    r_pt2pl_diffusion_output = r_pt2pl_diffusion_output[:, :-1, :] 
                
                r_pt2pl = r_pt2pl_diffusion_output.squeeze(0)


        elif self.input_dim == 3:
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_pos_pt2pl[:, -1],
                 rel_orient_pt2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)

        if self.dataset == 'argoverse_v2':
            edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index']
            edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                                batch=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
                                                loop=False, max_num_neighbors=300)
            type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type']
        elif self.dataset == 'nuscenes':
            edge_index_pl2pl = data['online_map_polygon', 'to', 'online_map_polygon']['edge_index']
            edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                                batch=data['online_map_polygon']['batch'] if isinstance(data, Batch) else None,
                                                loop=False, max_num_neighbors=300)
            type_pl2pl = data['online_map_polygon', 'to', 'online_map_polygon']['type']



            edge_index_pl2pl_gt = data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['edge_index']
            edge_index_pl2pl_radius_gt = radius_graph(x=pos_pl_gt[:, :2], r=self.pl2pl_radius,
                                                batch=data['maptr_gt_map_polygon']['batch'] if isinstance(data, Batch) else None,
                                                loop=False, max_num_neighbors=300)
            type_pl2pl_gt = data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['type']

        
        type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
        edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
                                                   edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])

        if self.dataset == 'nuscenes':
            type_pl2pl_radius_gt = type_pl2pl_gt.new_zeros(edge_index_pl2pl_radius_gt.size(1), dtype=torch.uint8)
            edge_index_pl2pl_gt, type_pl2pl_gt = merge_edges(edge_indices=[edge_index_pl2pl_radius_gt, edge_index_pl2pl_gt],
                                                    edge_attrs=[type_pl2pl_radius_gt, type_pl2pl_gt], reduce='max')
            rel_pos_pl2pl_gt = pos_pl_gt[edge_index_pl2pl_gt[0]] - pos_pl_gt[edge_index_pl2pl_gt[1]]
            rel_orient_pl2pl_gt = wrap_angle(orient_pl_gt[edge_index_pl2pl_gt[0]] - orient_pl_gt[edge_index_pl2pl_gt[1]])


        if self.input_dim == 2:
            r_pl2pl = torch.stack(
            [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                        nbr_vector=rel_pos_pl2pl[:, :2]),
                rel_orient_pl2pl], dim=-1)
            
            if self.dataset == 'nuscenes':
                r_pl2pl_gt = torch.stack(
                    [torch.norm(rel_pos_pl2pl_gt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(ctr_vector=orient_vector_pl_gt[edge_index_pl2pl_gt[1]],
                                            nbr_vector=rel_pos_pl2pl_gt[:, :2]),
                    rel_orient_pl2pl_gt], dim=-1)
                r_pl2pl_diffusion_original_samples = r_pl2pl_gt
                r_pl2pl_diffusion_local_cond = r_pl2pl

            # whether using diffusion module
            if self.pl2pl_denoise_inference:
                r_pl2pl = r_pl2pl.unsqueeze(0)
                padding_flag = False
                if r_pl2pl.shape[1] % 2 != 0:
                    padding = r_pl2pl[:, -1:, :] 
                    padding = padding.to(r_pl2pl.device)
                    r_pl2pl = torch.cat((r_pl2pl, padding), dim=1)
                    padding_flag = True
                                    
                r_pl2pl_noise = pyramid_noise_like(r_pl2pl)
                r_pl2pl_diffusion_output = r_pl2pl_noise

                # not use grad
                for param in pl2pl_noise_pred_net.parameters():
                    param.requires_grad = False   

                for k in self.diffusion_inferece_scheduler.timesteps[:]:
                    r_pl2pl_diffusion_output = self.diffusion_inferece_scheduler.scale_model_input(r_pl2pl_diffusion_output)
                    timesteps = k.unsqueeze(-1).repeat(r_pl2pl_diffusion_output.shape[0]).to(r_pl2pl.device)

                    # diffusion freeze the parameters
                    pl2pl_noise_pred = pl2pl_noise_pred_net(r_pl2pl_noise, timestep=timesteps, local_cond=r_pl2pl, global_cond=None)

                    r_pl2pl_diffusion_output = self.diffusion_inferece_scheduler.step(
                                model_output=pl2pl_noise_pred,
                                timestep=k,
                                sample=r_pl2pl_diffusion_output
                        ).prev_sample
                
                if padding_flag:
                    r_pl2pl_diffusion_output = r_pl2pl_diffusion_output[:, :-1, :] 
                
                r_pl2pl = r_pl2pl_diffusion_output.squeeze(0)

        elif self.input_dim == 3:
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_pos_pl2pl[:, -1],
                 rel_orient_pl2pl], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                      dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim)

        return {'x_pt': x_pt, 'x_pl': x_pl, 
                'r_pt2pl_diffusion_original_samples': r_pt2pl_diffusion_original_samples,
                'r_pt2pl_diffusion_local_cond': r_pt2pl_diffusion_local_cond,
                'r_pl2pl_diffusion_original_samples': r_pl2pl_diffusion_original_samples,
                'r_pl2pl_diffusion_local_cond': r_pl2pl_diffusion_local_cond,
                'cart_diffusion_original_samples': cart_diffusion_original_samples,
                'cart_diffusion_local_cond': cart_diffusion_local_cond,
                }
    

    


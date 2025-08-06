import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from datasets.nuscenes_knowledge_graph_dataset import NuScenesKGDataset

import pickle

class NuscenesDataset(Dataset):

    def __init__(self,
                root: str,
                split: str,
                resources_dir: str = '/home/szu3sgh/xc_slash/QCNet/resources',
                nuscenes_knowledge_graph_path: str = '/home/szu3sgh/Sgh_CR_RIX/szu3sgh/nuscenes_knowledge_graph/',
                transform: Optional[Callable] = None,
                local_radius: float = 50,
                centerline=False) -> None:

        self._split = split
        self._local_radius = local_radius
        self.centerline_used = centerline

        if split == 'train':
            self._directory = 'train'
        elif split == 'train_val':
            self._directory = 'train_val'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        elif split == 'mini_val':
            self._directory = 'mini_val'
        elif split == 'mini_train':
            self._directory = 'mini_train'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0].split('-')[1] + '.pkl' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]

        # get predicted map types
        self._predicted_map_polygon_types = ['divider', 'boundary', 'ped_crossing']
        self._predicted_map_point_types = ['divider', 'boundary', 'ped_crossing']

        # get prior knowledge graph
        self._resources_dir = resources_dir
        self._nuscenes_knowledge_graph_metadata_path = os.path.join(resources_dir, 'graph_meta_data.pkl')
        self._nuscenes_knowledge_graph_path = nuscenes_knowledge_graph_path

        # set dim and betas_dim
        self.dim = 2
        self.betas_dim = 2

        super(NuscenesDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            with open(raw_path, 'rb') as handle:
                raw_data = pickle.load(handle)
            
            processed_data = dict()
            processed_data['scenario_id'] = self.get_scenario_id(raw_data)
            processed_data['city'] = self.get_city(raw_data)
            
            processed_data['agent'] = self.get_agent_features(raw_data, self.dim)

            origin, rotate_mat = processed_data['agent']['origin'], processed_data['agent']['rotate_mat']
            
            processed_data.update(self.get_online_map_features(raw_data, self.dim, self.betas_dim, origin, rotate_mat))
            processed_data.update(self.get_maptr_gt_map_features(raw_data, self.dim, origin, rotate_mat))

            # get prior knowledge graph
            processed_data['prior_knowledge_graph'] = self.get_prior_knowledge_graph(raw_data)


            seq_id = os.path.splitext(os.path.basename(raw_path))[0].split('-')[1]
            # torch.save(processed_data, os.path.join(self.processed_dir, str(seq_id) + '.pt'))

            with open(os.path.join(self.processed_dir, f'{seq_id}.pkl'), 'wb') as processed_handle:
                pickle.dump(processed_data, processed_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    @staticmethod
    def get_scenario_id(data: Dict) -> str:
        return data['scene_name']

    @staticmethod
    def get_city(data: Dict) -> str:
        return data['map_name']
    
    def get_agent_features(self, data, dim) -> Dict[str, Any]:
        
        num_historical_steps = data['ego_hist'].shape[0] # 20
        num_future_steps = data['ego_fut'].shape[0] # 30
        num_steps = num_historical_steps + num_future_steps

        agent_indices = np.where(data['agent_type'] == 1)[0]
        num_nodes = len(agent_indices) + 1

        av_index = 0

        # ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        # https://argoverse.github.io/user-guide/tasks/motion_forecasting.html
        nusc_agent_category = np.concatenate(([3], np.full(len(agent_indices), 2)))
        nusc_agent_type = np.concatenate((data['ego_type'], data['agent_type'][agent_indices]))
        nusc_data_id = ['ego'] + [data['agent_name'][i] for i in agent_indices]

        assert nusc_agent_type.shape[0] == num_nodes
        assert nusc_agent_category.shape[0] == num_nodes
        assert len(nusc_data_id) == num_nodes

        num_agents = num_nodes

        # initialization
        valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
        position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
        heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
        acceleration = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

        # XYXdYdXddYddSC
        # position(X Y), velocity(Xd Yd), Acceleration(Xdd Ydd), Heading(Sin Cos)
        origin = torch.tensor(data['ego_hist'][-1][:2], dtype=torch.float)
        av_heading_vector = origin - torch.tensor(data['ego_hist'][-2][:2], dtype=torch.float)
        theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])

        x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)

        padding_mask[0, 0:50] = False
        xy = np.vstack([data['ego_hist'][:, 0:2], data['ego_fut'][:, 0:2]])
        # ego position
        x[0, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
        # ego velocity
        velocity[0, :] = torch.tensor(np.vstack([data['ego_hist'][:, 2:4], data['ego_fut'][:, 2:4]]))
        # ego acceleration
        acceleration[0, :] = torch.tensor(np.vstack([data['ego_hist'][:, 4:6], data['ego_fut'][:, 4:6]]))
        # ego heading
        heading_vector = torch.tensor(np.vstack([data['ego_hist'][:, 6:8], data['ego_fut'][:, 6:8]]))

        for heading_idx, (sin_value, cos_value) in enumerate(heading_vector):
            heading_angle = torch.atan2(sin_value, cos_value)
            # print(heading_angle)
            heading[0, heading_idx] = heading_angle 
            # print(heading_angle.sin())
            # print(heading_angle.cos())

        # predict and valid mask
        # generate predict_mask and valid_mask
        ego_agent_idx = 0
        ego_agent_steps = torch.arange(num_steps, dtype=torch.int)

        valid_mask[ego_agent_idx, ego_agent_steps] = True
        current_valid_mask[ego_agent_idx] = valid_mask[ego_agent_idx, num_historical_steps - 1]
        predict_mask[ego_agent_idx, ego_agent_steps] = True
        # a time step t is valid only when both t and t-1 are valid
        valid_mask[ego_agent_idx, 1: num_historical_steps] = (
                valid_mask[ego_agent_idx, :num_historical_steps - 1] &
                valid_mask[ego_agent_idx, 1: num_historical_steps])
        valid_mask[ego_agent_idx, 0] = False

        predict_mask[ego_agent_idx, :num_historical_steps] = False
        if not current_valid_mask[ego_agent_idx]:
            predict_mask[ego_agent_idx, num_historical_steps:] = False


        for i, index in enumerate(agent_indices):
            i += 1

            padding_mask[i, 0:50] = False
            xy = np.vstack([data['agent_hist'][index][:,:2], data['agent_fut'][index][:,:2]])
            # agent position
            x[i, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
            # agent velocity
            velocity[i, :] = torch.tensor(np.vstack([data['agent_hist'][index][:, 2:4], data['agent_fut'][index][:, 2:4]]))
            # agent acceleration
            acceleration[i, :] = torch.tensor(np.vstack([data['agent_hist'][index][:, 4:6], data['agent_fut'][index][:, 4:6]]))
            # agent heading
            current_heading_vector = torch.tensor(np.vstack([data['agent_hist'][index][:, 6:8], data['agent_hist'][index][:, 6:8]]))
            for heading_idx, (sin_value, cos_value) in enumerate(current_heading_vector):
                heading_angle = torch.atan2(sin_value, cos_value)
                heading[i, heading_idx] = heading_angle 
            
            # generate predict_mask and valid_mask
            agent_idx = i
            agent_steps = torch.arange(num_steps, dtype=torch.int)

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: num_historical_steps] = (
                    valid_mask[agent_idx, :num_historical_steps - 1] &
                    valid_mask[agent_idx, 1: num_historical_steps])
            valid_mask[agent_idx, 0] = False

            predict_mask[agent_idx, :num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, num_historical_steps:] = False

            # heading_vector = x[i, 19] - x[i, 18]
            # rotate_angles[i] = torch.atan2(heading_vector[1], heading_vector[0])

            positions = x.clone()
            position = positions

        return {
        # agent
        'num_nodes': num_nodes,
        'av_index': av_index,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'id': nusc_data_id,
        'type': nusc_agent_type,
        'category': nusc_agent_category,
        'position': position,
        'heading': heading,
        'velocity': velocity,
        
        'origin': origin,
        'rotate_mat': rotate_mat  
        }

    def get_online_map_features(self, data, dim, betas_dim, origin, rotate_mat):

        if 'predicted_map' not in data.keys():
            map_data = {
                'online_map_polygon': {},
                'online_map_point': {},
                ('online_map_point', 'to', 'online_map_polygon'): {},
                ('online_map_polygon', 'to', 'online_map_polygon'): {},
            }

            num_polygons = 0
            num_points = 0

            polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
            polygon_position_betas = torch.zeros(num_polygons, betas_dim, dtype=torch.float)
            polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
            polygon_height = torch.zeros(num_polygons, dtype=torch.float)
            polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
            point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_position_betas: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

            map_data['online_map_polygon']['num_nodes'] = num_polygons
            map_data['online_map_polygon']['position'] = polygon_position
            map_data['online_map_polygon']['position_betas'] = polygon_position_betas
            map_data['online_map_polygon']['orientation'] = polygon_orientation
            map_data['online_map_polygon']['height'] = polygon_height
            map_data['online_map_polygon']['type'] = polygon_type
            map_data['online_map_polygon']['is_intersection'] = polygon_is_intersection

            map_data['online_map_point']['num_nodes'] = 0
            map_data['online_map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['position_betas'] = torch.cat([], dtype=torch.float)
            map_data['online_map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['online_map_point']['side'] = torch.tensor([], dtype=torch.uint8)

            point_to_polygon_edge_index = torch.stack(
                [torch.arange(num_points.sum(), dtype=torch.long),
                torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
            
            # polygon to polygon relationship
            # online mapping doesn't have polygon relationships like left/right etc.
            row, col = torch.combinations(torch.arange(num_polygons), r=2).T
            polygon_to_polygon_edge_index = torch.cat([torch.stack([row, col], dim=0), 
                                    torch.stack([col, row], dim=0)], dim=1)

            # online mapping doesn't have polygon relationship types
            polygon_to_polygon_type = torch.zeros(polygon_to_polygon_edge_index.shape[1])

            map_data['online_map_point', 'to', 'online_map_polygon']['edge_index'] = point_to_polygon_edge_index
            map_data['online_map_polygon', 'to', 'online_map_polygon']['edge_index'] = polygon_to_polygon_edge_index
            map_data['online_map_polygon', 'to', 'online_map_polygon']['type'] = polygon_to_polygon_type

            print('no predicted map !')
            return map_data
        
        divider_len      = len(data['predicted_map']['divider'])
        boundary_len     = len(data['predicted_map']['boundary'])
        ped_crossing_len = len(data['predicted_map']['ped_crossing'])

        num_polygons = divider_len  + boundary_len + ped_crossing_len

        # initialization and online mapping only has 2 dimensions
        polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
        polygon_position_betas = torch.zeros(num_polygons, betas_dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_position_betas: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        # generate ids
        divider_ids = list(range(0, divider_len))
        boundary_ids = list(range(divider_len, divider_len + boundary_len))
        ped_crossing_ids = list(range(divider_len + boundary_len, divider_len + boundary_len + ped_crossing_len))

        polygon_ids = divider_ids + boundary_ids + ped_crossing_ids

        # initilize idx
        idx = 0

        divider = data['predicted_map']['divider']
        divider_betas = data['predicted_map']['divider_betas']

        # extract divider information
        for divider_index in range(divider_len):

            divider_idx = polygon_ids.index(idx)

            # extract current divider and divider_betas
            current_global_divider = torch.from_numpy(divider[divider_index]).to(torch.float32)
            current_divider_betas = torch.from_numpy(divider_betas[divider_index])

            # global coordinates -> local coordinates
            current_local_divider = torch.zeros(current_global_divider.shape)

            for current_global_divider_idx, current_global_divider_sample in enumerate(current_global_divider):
                current_local_divider[current_global_divider_idx] = torch.matmul(current_global_divider_sample - origin, rotate_mat)
            
            current_divider = current_local_divider

            # get polygon info
            polygon_position[divider_idx] = current_divider[0, :dim]
            polygon_position_betas[divider_idx] = current_divider_betas[0, :betas_dim]

            polygon_orientation[divider_idx] = torch.atan2(current_divider[1, 1] - current_divider[0, 1],
                                                            current_divider[1, 0] - current_divider[0, 0])
            ## don't have height information for online mapping
            polygon_height[divider_idx] = 0
            polygon_type[divider_idx] = self._predicted_map_polygon_types.index('divider')
            ## don't have intersection information for online mapping
            polygon_is_intersection[divider_idx] = 0

            # get position info
            point_position[divider_idx] = torch.cat([current_divider[:-1, :dim]], dim=0)
            point_position_betas[divider_idx] = torch.cat([current_divider_betas[:-1, :dim]], dim=0)

            divider_vectors = current_divider[1:] - current_divider[:-1]

            point_orientation[divider_idx] = torch.cat([torch.atan2(divider_vectors[:, 1], divider_vectors[:, 0])], dim=0)
            point_magnitude[divider_idx] = torch.norm(torch.cat([divider_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[divider_idx] = torch.cat([torch.zeros(divider_vectors[:, 1].shape)], dim=0)
            point_type[divider_idx] = torch.cat(
                [torch.full((len(divider_vectors),), self._predicted_map_point_types.index('divider'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[divider_idx] = torch.cat(
                [torch.full((len(divider_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_divider_idx = idx - 1

        # extract boundary information
        boundary = data['predicted_map']['boundary']
        boundary_betas = data['predicted_map']['boundary_betas']

        for boundary_index in range(boundary_len):

            boundary_idx = polygon_ids.index(idx)

            # extract current boundary and boundary_betas
            current_global_boundary = torch.from_numpy(boundary[boundary_index]).to(torch.float32)
            current_boundary_betas = torch.from_numpy(boundary_betas[boundary_index])

            # global coordinates -> local coordinates
            current_local_boundary = torch.zeros(current_global_boundary.shape)

            for current_global_boundary_idx, current_global_boundary_sample in enumerate(current_global_boundary):
                current_local_boundary[current_global_boundary_idx] = torch.matmul(current_global_boundary_sample - origin, rotate_mat)
            
            current_boundary = current_local_boundary

            # get polygon info
            polygon_position[boundary_idx] = current_boundary[0, :dim]
            polygon_position_betas[boundary_idx] = current_boundary_betas[0, :betas_dim]

            polygon_orientation[boundary_idx] = torch.atan2(current_boundary[1, 1] - current_boundary[0, 1],
                                                            current_boundary[1, 0] - current_boundary[0, 0])
            ## don't have height information for online mapping
            polygon_height[boundary_idx] = 0
            polygon_type[boundary_idx] = self._predicted_map_polygon_types.index('boundary')
            ## don't have intersection information for online mapping
            polygon_is_intersection[boundary_idx] = 0

            # get position info
            point_position[boundary_idx] = torch.cat([current_boundary[:-1, :dim]], dim=0)
            point_position_betas[boundary_idx] = torch.cat([current_boundary_betas[:-1, :dim]], dim=0)

            boundary_vectors = current_boundary[1:] - current_boundary[:-1]

            point_orientation[boundary_idx] = torch.cat([torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0])], dim=0)
            point_magnitude[boundary_idx] = torch.norm(torch.cat([boundary_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[boundary_idx] = torch.cat([torch.zeros(boundary_vectors[:, 1].shape)], dim=0)
            point_type[boundary_idx] = torch.cat(
                [torch.full((len(boundary_vectors),), self._predicted_map_point_types.index('boundary'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[boundary_idx] = torch.cat(
                [torch.full((len(boundary_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_boundary_idx = idx - 1

        # extract ped_crossing information
        ped_crossing = data['predicted_map']['ped_crossing']
        ped_crossing_betas = data['predicted_map']['ped_crossing_betas']

        for ped_crossing_index in range(ped_crossing_len):

            ped_crossing_idx = polygon_ids.index(idx)

            # extract current boundary and boundary_betas
            current_global_ped_crossing = torch.from_numpy(ped_crossing[ped_crossing_index]).to(torch.float32)
            current_ped_crossing_betas = torch.from_numpy(ped_crossing_betas[ped_crossing_index])

            # global coordinates -> local coordinates
            current_local_ped_crossing = torch.zeros(current_global_ped_crossing.shape)

            for current_global_ped_crossing_idx, current_global_ped_crossing_sample in enumerate(current_global_ped_crossing):
                current_local_ped_crossing[current_global_ped_crossing_idx] = torch.matmul(current_global_ped_crossing_sample - origin, rotate_mat)
            
            current_ped_crossing = current_local_ped_crossing

            # get polygon info
            polygon_position[ped_crossing_idx] = current_ped_crossing[0, :dim]
            polygon_position_betas[ped_crossing_idx] = current_ped_crossing_betas[0, :betas_dim]

            polygon_orientation[ped_crossing_idx] = torch.atan2(current_ped_crossing[1, 1] - current_ped_crossing[0, 1],
                                                            current_ped_crossing[1, 0] - current_ped_crossing[0, 0])
            ## don't have height information for online mapping
            polygon_height[ped_crossing_idx] = 0
            polygon_type[ped_crossing_idx] = self._predicted_map_polygon_types.index('ped_crossing')
            ## don't have intersection information for online mapping
            polygon_is_intersection[ped_crossing_idx] = 0

            # get position info
            point_position[ped_crossing_idx] = torch.cat([current_ped_crossing[:-1, :dim]], dim=0)
            point_position_betas[ped_crossing_idx] = torch.cat([current_ped_crossing_betas[:-1, :dim]], dim=0)

            ped_crossing_vectors = current_ped_crossing[1:] - current_ped_crossing[:-1]

            point_orientation[ped_crossing_idx] = torch.cat([torch.atan2(ped_crossing_vectors[:, 1], ped_crossing_vectors[:, 0])], dim=0)
            point_magnitude[ped_crossing_idx] = torch.norm(torch.cat([ped_crossing_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[ped_crossing_idx] = torch.cat([torch.zeros(ped_crossing_vectors[:, 1].shape)], dim=0)  
            point_type[ped_crossing_idx] = torch.cat(
                [torch.full((len(ped_crossing_vectors),), self._predicted_map_point_types.index('ped_crossing'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[ped_crossing_idx] = torch.cat(
                [torch.full((len(ped_crossing_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_ped_crossing_idx = idx - 1

        # check extracted lane features quality
        if divider_len:
            assert polygon_position[last_divider_idx][-1] == torch.matmul(torch.from_numpy(divider[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1]
            assert polygon_position_betas[divider_idx][-1] == torch.from_numpy(divider_betas[-1]).to(dtype=torch.float)[0][-1]

        if boundary_len:
            assert polygon_position[last_boundary_idx][-1] == torch.matmul(torch.from_numpy(boundary[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1]      
            assert polygon_position_betas[last_boundary_idx][-1] == torch.from_numpy(boundary_betas[-1]).to(dtype=torch.float)[0][-1]

        if ped_crossing_len:
            assert polygon_position[last_ped_crossing_idx][-1] == torch.matmul(torch.from_numpy(ped_crossing[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1]          
            assert polygon_position_betas[last_ped_crossing_idx][-1] == torch.from_numpy(ped_crossing_betas[-1]).to(dtype=torch.float)[0][-1]

        # point to polygon relationship
        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        
        # polygon to polygon relationship
        # online mapping doesn't have polygon relationships like left/right etc.
        row, col = torch.combinations(torch.arange(num_polygons), r=2).T
        polygon_to_polygon_edge_index = torch.cat([torch.stack([row, col], dim=0), 
                                torch.stack([col, row], dim=0)], dim=1)

        # online mapping doesn't have polygon relationship types
        polygon_to_polygon_type = torch.zeros(polygon_to_polygon_edge_index.shape[1])

        # organize data
        map_data = {
            'online_map_polygon': {},
            'online_map_point': {},
            ('online_map_point', 'to', 'online_map_polygon'): {},
            ('online_map_polygon', 'to', 'online_map_polygon'): {},
        }
        map_data['online_map_polygon']['num_nodes'] = num_polygons
        map_data['online_map_polygon']['position'] = polygon_position
        map_data['online_map_polygon']['position_betas'] = polygon_position_betas
        map_data['online_map_polygon']['orientation'] = polygon_orientation
        map_data['online_map_polygon']['height'] = polygon_height
        map_data['online_map_polygon']['type'] = polygon_type
        map_data['online_map_polygon']['is_intersection'] = polygon_is_intersection

        if len(num_points) == 0:
            map_data['online_map_point']['num_nodes'] = 0
            map_data['online_map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['position_betas'] = torch.cat([], dtype=torch.float)
            map_data['online_map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['online_map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['online_map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['online_map_point']['num_nodes'] = num_points.sum().item()
            map_data['online_map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['online_map_point']['position_betas'] = torch.cat(point_position_betas, dim=0)
            map_data['online_map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['online_map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            map_data['online_map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['online_map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['online_map_point']['side'] = torch.cat(point_side, dim=0)

        map_data['online_map_point', 'to', 'online_map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['online_map_polygon', 'to', 'online_map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['online_map_polygon', 'to', 'online_map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def get_maptr_gt_map_features(self, data, dim, origin, rotate_mat):

        if 'maptr_gt_map' not in data.keys():
            map_data = {
                'maptr_gt_map_polygon': {},
                'maptr_gt_map_point': {},
                ('maptr_gt_map_point', 'to', 'maptr_gt_map_polygon'): {},
                ('maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon'): {},
            }

            num_polygons = 0
            num_points = 0

            polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
            polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
            polygon_height = torch.zeros(num_polygons, dtype=torch.float)
            polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
            point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
            point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

            map_data['maptr_gt_map_polygon']['num_nodes'] = num_polygons
            map_data['maptr_gt_map_polygon']['position'] = polygon_position
            map_data['maptr_gt_map_polygon']['orientation'] = polygon_orientation
            map_data['maptr_gt_map_polygon']['height'] = polygon_height
            map_data['maptr_gt_map_polygon']['type'] = polygon_type
            map_data['maptr_gt_map_polygon']['is_intersection'] = polygon_is_intersection

            map_data['maptr_gt_map_point']['num_nodes'] = 0
            map_data['maptr_gt_map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['maptr_gt_map_point']['side'] = torch.tensor([], dtype=torch.uint8)

            point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        
            # polygon to polygon relationship
            # online mapping doesn't have polygon relationships like left/right etc.
            row, col = torch.combinations(torch.arange(num_polygons), r=2).T
            polygon_to_polygon_edge_index = torch.cat([torch.stack([row, col], dim=0), 
                                    torch.stack([col, row], dim=0)], dim=1)

            # online mapping doesn't have polygon relationship types
            polygon_to_polygon_type = torch.zeros(polygon_to_polygon_edge_index.shape[1])

            map_data['maptr_gt_map_point', 'to', 'maptr_gt_map_polygon']['edge_index'] = point_to_polygon_edge_index
            map_data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['edge_index'] = polygon_to_polygon_edge_index
            map_data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['type'] = polygon_to_polygon_type

            print('no maptr gt map !')
            return map_data
        
        divider_len      = len(data['maptr_gt_map']['divider'])
        boundary_len     = len(data['maptr_gt_map']['boundary'])
        ped_crossing_len = len(data['maptr_gt_map']['ped_crossing'])

        num_polygons = divider_len  + boundary_len + ped_crossing_len

        # initialization and online mapping only has 2 dimensions
        polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        # generate ids
        divider_ids = list(range(0, divider_len))
        boundary_ids = list(range(divider_len, divider_len + boundary_len))
        ped_crossing_ids = list(range(divider_len + boundary_len, divider_len + boundary_len + ped_crossing_len))
        polygon_ids = divider_ids + boundary_ids + ped_crossing_ids

        # initilize idx
        idx = 0
        divider = data['maptr_gt_map']['divider']

        # extract divider information
        for divider_index in range(divider_len):
            divider_idx = polygon_ids.index(idx)

            # extract current divider and divider_betas
            current_global_divider = torch.from_numpy(divider[divider_index]).to(torch.float32)

            # global coordinates -> local coordinates
            current_local_divider = torch.zeros(current_global_divider.shape)

            for current_global_divider_idx, current_global_divider_sample in enumerate(current_global_divider):
                current_local_divider[current_global_divider_idx] = torch.matmul(current_global_divider_sample - origin, rotate_mat)
            
            current_divider = current_local_divider

            # get polygon info
            polygon_position[divider_idx] = current_divider[0, :dim]
            polygon_orientation[divider_idx] = torch.atan2(current_divider[1, 1] - current_divider[0, 1],
                                                            current_divider[1, 0] - current_divider[0, 0])
            ## don't have height information for online mapping
            polygon_height[divider_idx] = 0
            polygon_type[divider_idx] = self._predicted_map_polygon_types.index('divider')
            ## don't have intersection information for online mapping
            polygon_is_intersection[divider_idx] = 0

            # get position info
            point_position[divider_idx] = torch.cat([current_divider[:-1, :dim]], dim=0)
            divider_vectors = current_divider[1:] - current_divider[:-1]

            point_orientation[divider_idx] = torch.cat([torch.atan2(divider_vectors[:, 1], divider_vectors[:, 0])], dim=0)
            point_magnitude[divider_idx] = torch.norm(torch.cat([divider_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[divider_idx] = torch.cat([torch.zeros(divider_vectors[:, 1].shape)], dim=0)
            point_type[divider_idx] = torch.cat(
                [torch.full((len(divider_vectors),), self._predicted_map_point_types.index('divider'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[divider_idx] = torch.cat(
                [torch.full((len(divider_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_divider_idx = idx - 1

        # extract boundary information
        boundary = data['maptr_gt_map']['boundary']

        for boundary_index in range(boundary_len):
            boundary_idx = polygon_ids.index(idx)

            # extract current boundary and boundary_betas
            current_global_boundary = torch.from_numpy(boundary[boundary_index]).to(torch.float32)

            # global coordinates -> local coordinates
            current_local_boundary = torch.zeros(current_global_boundary.shape)

            for current_global_boundary_idx, current_global_boundary_sample in enumerate(current_global_boundary):
                current_local_boundary[current_global_boundary_idx] = torch.matmul(current_global_boundary_sample - origin, rotate_mat)
            
            current_boundary = current_local_boundary

            # get polygon info
            polygon_position[boundary_idx] = current_boundary[0, :dim]
            polygon_orientation[boundary_idx] = torch.atan2(current_boundary[1, 1] - current_boundary[0, 1],
                                                            current_boundary[1, 0] - current_boundary[0, 0])
            ## don't have height information for online mapping
            polygon_height[boundary_idx] = 0
            polygon_type[boundary_idx] = self._predicted_map_polygon_types.index('boundary')
            ## don't have intersection information for online mapping
            polygon_is_intersection[boundary_idx] = 0

            # get position info
            point_position[boundary_idx] = torch.cat([current_boundary[:-1, :dim]], dim=0)

            boundary_vectors = current_boundary[1:] - current_boundary[:-1]

            point_orientation[boundary_idx] = torch.cat([torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0])], dim=0)
            point_magnitude[boundary_idx] = torch.norm(torch.cat([boundary_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[boundary_idx] = torch.cat([torch.zeros(boundary_vectors[:, 1].shape)], dim=0)
            point_type[boundary_idx] = torch.cat(
                [torch.full((len(boundary_vectors),), self._predicted_map_point_types.index('boundary'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[boundary_idx] = torch.cat(
                [torch.full((len(boundary_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_boundary_idx = idx - 1

        # extract ped_crossing information
        ped_crossing = data['maptr_gt_map']['ped_crossing']

        for ped_crossing_index in range(ped_crossing_len):
            ped_crossing_idx = polygon_ids.index(idx)

            # extract current boundary and boundary_betas
            current_global_ped_crossing = torch.from_numpy(ped_crossing[ped_crossing_index]).to(torch.float32)

            # global coordinates -> local coordinates
            current_local_ped_crossing = torch.zeros(current_global_ped_crossing.shape)

            for current_global_ped_crossing_idx, current_global_ped_crossing_sample in enumerate(current_global_ped_crossing):
                current_local_ped_crossing[current_global_ped_crossing_idx] = torch.matmul(current_global_ped_crossing_sample - origin, rotate_mat)
            
            current_ped_crossing = current_local_ped_crossing

            # get polygon info
            polygon_position[ped_crossing_idx] = current_ped_crossing[0, :dim]

            polygon_orientation[ped_crossing_idx] = torch.atan2(current_ped_crossing[1, 1] - current_ped_crossing[0, 1],
                                                            current_ped_crossing[1, 0] - current_ped_crossing[0, 0])
            ## don't have height information for online mapping
            polygon_height[ped_crossing_idx] = 0
            polygon_type[ped_crossing_idx] = self._predicted_map_polygon_types.index('ped_crossing')
            ## don't have intersection information for online mapping
            polygon_is_intersection[ped_crossing_idx] = 0

            # get position info
            point_position[ped_crossing_idx] = torch.cat([current_ped_crossing[:-1, :dim]], dim=0)

            ped_crossing_vectors = current_ped_crossing[1:] - current_ped_crossing[:-1]

            point_orientation[ped_crossing_idx] = torch.cat([torch.atan2(ped_crossing_vectors[:, 1], ped_crossing_vectors[:, 0])], dim=0)
            point_magnitude[ped_crossing_idx] = torch.norm(torch.cat([ped_crossing_vectors[:, :2]], dim=0), p=2, dim=-1)
            
            ## point position doesn't have height
            point_height[ped_crossing_idx] = torch.cat([torch.zeros(ped_crossing_vectors[:, 1].shape)], dim=0)  
            point_type[ped_crossing_idx] = torch.cat(
                [torch.full((len(ped_crossing_vectors),), self._predicted_map_point_types.index('ped_crossing'), dtype=torch.uint8)], dim=0)

            ## online mapping doesn't have boundary    
            point_side[ped_crossing_idx] = torch.cat(
                [torch.full((len(ped_crossing_vectors),), 0, dtype=torch.uint8)], dim=0)

            idx = idx + 1

        last_ped_crossing_idx = idx - 1

        # maptr_gt_map maybe doesn't have divider or boundary or ped_crossing
        # check extracted lane features quality
        if divider_len:
            assert polygon_position[last_divider_idx][-1] == torch.matmul(torch.from_numpy(divider[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1]
        if boundary_len:
            assert polygon_position[last_boundary_idx][-1] == torch.matmul(torch.from_numpy(boundary[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1]  
        if ped_crossing_len:
            assert polygon_position[last_ped_crossing_idx][-1] == torch.matmul(torch.from_numpy(ped_crossing[-1]).to(dtype=torch.float)[0] - origin, rotate_mat)[-1] 
            
        # point to polygon relationship
        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        
        # polygon to polygon relationship
        # online mapping doesn't have polygon relationships like left/right etc.
        row, col = torch.combinations(torch.arange(num_polygons), r=2).T
        polygon_to_polygon_edge_index = torch.cat([torch.stack([row, col], dim=0), 
                                torch.stack([col, row], dim=0)], dim=1)

        # online mapping doesn't have polygon relationship types
        polygon_to_polygon_type = torch.zeros(polygon_to_polygon_edge_index.shape[1])

        # organize data
        map_data = {
            'maptr_gt_map_polygon': {},
            'maptr_gt_map_point': {},
            ('maptr_gt_map_point', 'to', 'maptr_gt_map_polygon'): {},
            ('maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon'): {},
        }
        map_data['maptr_gt_map_polygon']['num_nodes'] = num_polygons
        map_data['maptr_gt_map_polygon']['position'] = polygon_position
        map_data['maptr_gt_map_polygon']['orientation'] = polygon_orientation
        map_data['maptr_gt_map_polygon']['height'] = polygon_height
        map_data['maptr_gt_map_polygon']['type'] = polygon_type
        map_data['maptr_gt_map_polygon']['is_intersection'] = polygon_is_intersection

        if len(num_points) == 0:
            map_data['maptr_gt_map_point']['num_nodes'] = 0
            map_data['maptr_gt_map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['maptr_gt_map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['maptr_gt_map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['maptr_gt_map_point']['num_nodes'] = num_points.sum().item()
            map_data['maptr_gt_map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['maptr_gt_map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['maptr_gt_map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            map_data['maptr_gt_map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['maptr_gt_map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['maptr_gt_map_point']['side'] = torch.cat(point_side, dim=0)

        map_data['maptr_gt_map_point', 'to', 'maptr_gt_map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['maptr_gt_map_polygon', 'to', 'maptr_gt_map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    def get_prior_knowledge_graph(self, data):

        # prior knowledge graph only generate train and val dataset
        if self._split in ['train_val', 'test', 'mini_val', 'mini_train']:
            return []

        with open(self._nuscenes_knowledge_graph_metadata_path, 'rb') as f:
            graph_meta_data = pickle.load(f)
        
        split = self._split
        file_path = os.path.join(self._resources_dir, f'{split}_nuscenes_kg_scene_sample_list.pkl')

        # load split dataset
        dataset = NuScenesKGDataset(root = os.path.join(self._nuscenes_knowledge_graph_path, split), data_metadata = graph_meta_data)

        # generate nuscenes kg file which contains kg_scene or sample token
        # get nuscenes_kg_scene_sample_list
        if os.path.isfile(file_path):
            file_path = os.path.join(self._resources_dir, f'{split}_nuscenes_kg_scene_sample_list.pkl')
            with open(file_path, 'rb') as f:
                nuscenes_kg_scene_sample_list = pickle.load(f)

        else:
            print('Generate nuscenes_kg_scene_sample_list ....')

            nuscenes_kg_scene_sample_list = []
            for graph_data in tqdm(dataset):

                hasSceneParticipant_edge = np.array(graph_data[('Scene', 'hasSceneParticipant', 'SceneParticipant')]['edge_index'])
                target_scene_participant = graph_data['target_index']
                scene_dict = graph_data['indexes_to_uris']['Scene']

                if target_scene_participant in hasSceneParticipant_edge[1]:
                    indices = np.where(target_scene_participant == hasSceneParticipant_edge[1])[0]
                    
                    assert len(indices) == 1
                    kg_scene_sample_id = hasSceneParticipant_edge[0][indices[0]]
                    scene_id = scene_dict[kg_scene_sample_id].replace('http://www.nuscenes.org/nuScenes/Scene_', '')
                    
                    nuscenes_kg_scene_sample_list.append(scene_id)

            with open(os.path.join(self._resources_dir, f'{split}_nuscenes_kg_scene_sample_list.pkl'), "wb") as f:
                pickle.dump(nuscenes_kg_scene_sample_list, f)

        # add knowledge graph for dataset
        prior_knowledge_indices = np.where(np.array(nuscenes_kg_scene_sample_list) == data['sample_token'])[0]
        prior_knowledge_graph = []

        if prior_knowledge_indices.size != 0:  
            prior_knowledge_graph.append(dataset[prior_knowledge_indices[0]])  

            # assert prior-knowledge is correct        
            hasSceneParticipant_edge = np.array(prior_knowledge_graph[0][('Scene', 'hasSceneParticipant', 'SceneParticipant')]['edge_index'])
            target_scene_participant = prior_knowledge_graph[0]['target_index']
            scene_dict = prior_knowledge_graph[0]['indexes_to_uris']['Scene']

            if target_scene_participant in hasSceneParticipant_edge[1]:
                indices = np.where(target_scene_participant == hasSceneParticipant_edge[1])[0]
                
                assert len(indices) == 1
                kg_scene_sample_id = hasSceneParticipant_edge[0][indices[0]]
                scene_id = scene_dict[kg_scene_sample_id].replace('http://www.nuscenes.org/nuScenes/Scene_', '')

            assert scene_id == data['sample_token']

        else:
            pass
            # print('no prior knowledge graph')

        return prior_knowledge_graph


if __name__ == '__main__':

    # print('generate val dataset!')
    # NuscenesDataset('/home/szu3sgh/Sgh_CR_RIX/SGH_CR_RIX/rix2_shared/nuscenes_omg_dataset/stream', 'val')

    print('generate train dataset!')
    NuscenesDataset('/home/szu3sgh/Sgh_CR_RIX/SGH_CR_RIX/rix2_shared/nuscenes_omg_dataset/stream', 'train')



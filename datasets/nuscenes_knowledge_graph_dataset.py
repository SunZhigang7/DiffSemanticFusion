import os
import os.path as osp
from tqdm import tqdm
import torch
import pickle
from torch_geometric.data import Dataset

class NuScenesKGDataset(Dataset):
    def __init__(self, root, data_metadata, transform=None, pre_transform=None, pre_filter=None):
        self.data_metadata = data_metadata
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        list_raw_files = os.listdir(self.raw_dir)
        list_raw_files = [
            file for file in list_raw_files if file.endswith('pt')]
        # list_raw_files = ['0.pt', '1.pt', '2.pt']
        return list_raw_files

    @property
    def processed_file_names(self):
        list_processed_files = os.listdir(self.processed_dir)
        list_processed_files = [file for file in list_processed_files if file.startswith('data')]
        return list_processed_files

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        for fn in tqdm(self.raw_file_names):

            data = torch.load(os.path.join(self.raw_dir, fn))

            # initilize dataset metadata
            for rel in self.data_metadata[1]:
                if rel not in data.edge_index_dict.keys():
                    data[rel]['edge_index'] = torch.tensor(
                        [[], []], dtype=torch.int64)

            for node in self.data_metadata[0]:
                if node not in data.x_dict:
                    if node == 'Sequence':
                        data[node].x = torch.zeros(5, 10)
                    if node == 'SceneParticipant':
                        data[node].x = torch.zeros(5, 24)
                    if node == 'Participant':
                        data[node].x = torch.zeros(5, 26)
                    if node == 'LaneSnippet':
                        data[node].x = torch.zeros(5, 1)
                    if node == 'LaneSlice':
                        data[node].x = torch.zeros(5, 4)
                    if node == 'OrderedPose':
                        data[node].x = torch.zeros(5, 3)
                    else:
                        data[node].x = torch.zeros(5, 1)

            for node in ['Scene', 'Lane', 'CarparkArea', 'Walkway', 'Intersection', 'LaneConnector',
                         'PedCrossingStopArea', 'StopSignArea', 'TrafficLightStopArea', 'TurnStopArea', 
                         'YieldStopArea', 'PedCrossing']:
                data[node].x = torch.zeros(data[node].x.size()[0], 5)
            
            torch.save(data, osp.join(self.processed_dir, f'data_{fn}'))
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                self.processed_file_names[idx]))

        return data

if __name__ == '__main__':

    with open('/home/szu3sgh/xc_slash/QCNet/resources/graph_meta_data.pkl', 'rb') as f:
        graph_meta_data = pickle.load(f)

    dataset = NuScenesKGDataset(root = os.path.join('/home/szu3sgh/Sgh_CR_RIX/szu3sgh/nuscenes_knowledge_graph/', 'val'), data_metadata = graph_meta_data)
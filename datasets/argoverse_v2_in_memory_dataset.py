import os
import torch
import pickle
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset, download_url

class ArgoverseV2DatasetInMemory(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None, combined_file_name='data.pt'):
        
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        
        self.split = split
        self.root_dir = os.path.join(root, split)
        self.combined_file_name = combined_file_name

        super().__init__(self.root_dir, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        list_processed_files = os.listdir(self.processed_dir)
        list_processed_files = [file for file in list_processed_files if file.endswith('pkl')]
        return list_processed_files

    @property
    def processed_file_names(self):
        return [self.combined_file_name]

    def download(self):
        pass

    def process(self):
        if not os.path.exists(self.processed_paths[0]):
            # load whole dataset
            data_list = []
            for fn in tqdm(self.raw_file_names):
                with open(os.path.join(self.processed_dir, fn), 'rb') as handle:
                    data = HeteroData(pickle.load(handle))

                data_list.append(data)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    ArgoverseV2DatasetInMemory('/home/szu3sgh/Sgh_CR_RIX/SGH_CR_RIX/rix2_shared/av2_motion_forecasting', 'val')
    ArgoverseV2DatasetInMemory('/home/szu3sgh/Sgh_CR_RIX/SGH_CR_RIX/rix2_shared/av2_motion_forecasting', 'test')
    ArgoverseV2DatasetInMemory('/home/szu3sgh/Sgh_CR_RIX/SGH_CR_RIX/rix2_shared/av2_motion_forecasting', 'train')
    


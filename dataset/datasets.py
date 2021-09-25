from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import torch
import numpy as np
# from pathlib import Path
# change this to Path asap, don't want to break it now
# download if dataset doesn't exist 
import os

class LandCoverNet(Dataset):
    def __init__(self, dataset_path, bands=['B02', 'B03', 'B04'], transform=None, target_transform=None, download=True, train=True):
        self.dataset_path = dataset_path
        self.pattern = 'ref_landcovernet_v1_labels_'
        self.bands = bands
        self.labels = self.get_labels()
        self.transform = transform
       
        
    def get_labels(self):
        
        labels = pd.DataFrame(columns=['tile_name','date'])

        for tile in os.listdir(self.dataset_path):

            tile_name = tile.replace(self.pattern, '')
            label = pd.read_csv(f'{self.dataset_path}/{tile}/labels/{tile_name}_labeling_dates.csv')
            label = label.sample(7)
            label = label[[tile_name[:-3]]].rename(columns={tile_name[:-3]: 'date'})
            label['tile_name'] = tile_name
            labels = labels.append(label, ignore_index=True)
        return labels
    
    def __len__(self):
        return len(self.labels)
    
    def open_image(self, img_path):
        imgs = []
        for band in self.bands:
            imgs.append(io.imread(f'{img_path}{band}_10m.tif'))
        return np.stack(imgs, axis=2)
    
    def __getitem__(self, idx):
       
        tile_name = self.labels.iloc[idx, 0]
        tile_date = str(self.labels.iloc[idx, 1])
#         print(tile_date)
        tile_year = tile_date[:4]
        ref_dir = f'{self.dataset_path}/{self.pattern}{tile_name}'
        img_path = f'{ref_dir}/source/{tile_name}_{tile_date}_'
        label_path = f'{ref_dir}/labels/{tile_name}_{tile_year}_LC_10m.tif'
        image = torch.from_numpy(self.open_image(img_path).transpose((2, 0, 1)).astype(np.int32)).float()
        label = torch.from_numpy(io.imread(label_path)[:,:,0].astype(np.int32)).long()
        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)
#         if self.target_transform:
#             label = self.target_transform(label)
        return image, label
    
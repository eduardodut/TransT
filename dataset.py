from turtle import pos
import torch
from torch.utils.data import Dataset, ConcatDataset
# import torch
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
from glob import glob
import os
import pandas as pd
from PIL import Image
import json
import numpy as np
import supervisely_lib as  sly
from typing import Any, Callable, Dict, Optional, Tuple, List, TypeVar
from PIL import Image
import multiprocessing as mp
import copy
T_co = TypeVar('T_co', covariant=True)

def default_ann_loader(ann_path):
    f = open(ann_path,'r')
    ann = json.load(f)
    f.close()
    return ann

def default_img_loader(img_path):
    return Image.open(img_path).convert("RGB")
    
class Tset_Supervisely(Dataset):
    def __init__(self, 
                 project_path,
                 pre_load_ann: bool = False,
                 load_img: Optional[Callable] = None,
                 load_ann:  Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        
        self.project = sly.Project(project_path,sly.OpenMode.READ)
        self.pre_load_ann = pre_load_ann
        
        self.load_img = load_img
        if not load_img:
            self.load_img = default_img_loader
        self.transform = transform
        self.target_transform = target_transform
        
        self.load_ann = load_ann
        if not load_ann:
            self.load_ann = default_ann_loader
        
        self.meta = self.project.meta
        self.project_name = self.project.name
        self.samples = []
        for dataset in self.project.datasets:
            for imagem in dataset:
                img_path = dataset.get_img_path(imagem)
                ann_path = dataset.get_ann_path(imagem)
                ann_loaded = False
                if pre_load_ann:
                    ann_path = self.load_ann(ann_path)
                    ann_loaded = True
                self.samples.append([dataset.name, img_path, ann_path, ann_loaded])
        self.samples = pd.DataFrame(self.samples,columns=['seq_label', 'img', 'ann', 'ann_loaded'])
        self.samples['project_label'] = project_path.split(os.sep)[-1]
        seq_idx = pd.DataFrame(self.samples['seq_label'].unique()).reset_index()
        seq_idx.columns = ['seq_idx','seq_label']
        self.samples = self.samples.merge(seq_idx,on=['seq_label'],how='left')
        self.samples = self.samples[['project_label','seq_idx','seq_label', 'img', 'ann', 'ann_loaded']]
               
    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> Any:
        project_label, seq_idx, seq_label,img_path, ann, ann_loaded = self.samples.iloc[index]
        img = self.load_img(img_path)
        if self.transform:
            img = self.transform(img)
        if not ann_loaded:
            ann = self.load_ann(ann)   
            self.samples.loc[index,['ann','ann_loaded']] = [ann,True]
        if self.target_transform:
            ann = self.target_transform(ann)            
        return project_label, seq_idx, seq_label, img, ann
    
    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        out = copy.deepcopy(self)
        out.samples = pd.concat([self.samples,other.samples],ignore_index=True)
        
        return out
    def __radd__(self,other):
        if other == 0 or (other != other): # nan
            return self
        else:
            return self.__add__(other)
    def load_anns(self,num_workers: int = 1):
        to_load = self.samples.loc[self.samples['ann_loaded'] == False]
        if to_load.shape[0] > 0:
            if num_workers < 0:
                num_workers = mp.cpu_count()
            if num_workers == 1:
                to_load['ann'] = to_load['ann'].apply(self.load_ann) 
            elif num_workers > 1:    
                pool = mp.Pool(processes=num_workers)
                to_load['ann'] = pool.map(self.load_ann, to_load['ann'].values)
                pool.close()
                pool.join()
            to_load['ann_loaded'] = True
            self.samples.loc[self.samples['ann_loaded'] == False] = to_load
    def img_by_seq(self):
        return self.samples.groupby(['seq_idx','seq_label']).agg({'img':'count'}) 
    def filter_by_seq(self, seq_label):
        out = copy.deepcopy(self)
        out.samples = out.samples.loc[out.samples.seq_label == seq_label]
        return out

   
if __name__ == "__main__":
    import os
    
    print

        
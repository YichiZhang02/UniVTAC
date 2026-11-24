"""Reproducible, without-replacement anchor sampling from the full contact pool."""
import math

import torch
from torch.utils.data import Sampler


class EpochAnchorSampler(Sampler):
    def __init__(self,dataset,samples=80000,*,num_replicas=1,rank=0,seed=42):
        if samples<1 or num_replicas<1 or not 0<=rank<num_replicas:
            raise ValueError('Invalid anchor sampling count or distributed rank')
        self.population=len(dataset)
        if self.population<1:raise ValueError('Empty anchor population')
        self.selected_count=min(samples,self.population)
        self.num_replicas,self.rank,self.seed=num_replicas,rank,seed
        self.num_samples=math.ceil(self.selected_count/num_replicas)
        self.total_size=self.num_samples*num_replicas
        self.epoch=0

    def set_epoch(self,epoch):self.epoch=epoch

    def __len__(self):return self.num_samples

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        selected=torch.randperm(self.population,generator=generator)[:self.selected_count].tolist()
        if self.total_size>len(selected):
            selected+=(selected*math.ceil((self.total_size-len(selected))/len(selected)))[:self.total_size-len(selected)]
        return iter(selected[self.rank:self.total_size:self.num_replicas])

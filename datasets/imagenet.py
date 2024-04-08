import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from .sampler import ClassAwareSampler
import random

class LT_Dataset(Dataset):
    num_classes = 1000

    def __init__(self, config, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.config = config
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]
        

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
    

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        '''
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 
        '''
        meta = dict()
        if self.config.sampler.dual_sample.enable: #balance sampler
            assert self.config.sampler.weighted_sampler.type in ["balance", "reverse", "long-tailed"]
            if self.config.sampler.dual_sample.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_data[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "balance":
                sample_class = random.randint(0, self.num_classes-1)
                sample_indexes = self.class_data[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.config.sampler.dual_sample.type == "long-tailed":
                sample_index = index #random.randint(0, self.__len__() - 1)
            
            sample_path, sample_target = self.img_path[sample_index], self.targets[sample_index]
            with open(sample_path, 'rb') as f:
                sample_image = Image.open(f).convert('RGB')
            if self.transform is not None:
                sample_image = self.transform(sample_image)
            meta['sample_image'] = sample_image
            meta['sample_label'] = sample_target

        if self.config.sampler.type == "weighted sampler":
            assert self.config.sampler.weighted_sampler.type in ["balance", "reverse"]
            if  self.config.sampler.weighted_sampler.type == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.config.sampler.weighted_sampler.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_data[sample_class]
            index = random.choice(sample_indexes)
        path, target = self.img_path[index], self.targets[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, target, meta        


class LT_Dataset_Eval(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 


class ImageNet_LT(object):
    def __init__(self, config, distributed, root="", batch_size=60, num_works=40):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        
        train_txt = "./datasets/data_txt/ImageNet_LT_train_part.txt"
        eval_txt = "./datasets/data_txt/ImageNet_LT_test_part.txt"
        
        train_dataset = LT_Dataset(config, root, train_txt, transform=transform_train)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
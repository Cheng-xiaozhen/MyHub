import json
import os
import random
import torch
import numpy as np
from PIL import Image
from ...core.base_dataset import BaseDataset
from ...core.registry import register_dataset

@register_dataset('CrossDataset')
class CrossDataset(BaseDataset):
    def __init__(self,path,image_size,**kwargs):
        """
        跨数据集加载器, 用于在多个数据集上进行训练和评估
        Args:
            path (str): 数据集路径
            image_size (int): 图像大小
            **kwargs: 每个数据集特有的其他参数
        """
        self.image_size = image_size
        super(CrossDataset, self).__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:
        """
        重写基类的方法以初始化跨数据集路径
        从json文件中读取所有数据集的图像路径和标签
        """
        if isinstance(self.path, str):
            dataset_paths = [self.path]
        elif isinstance(self.path, list):
            dataset_paths = self.path
        else:
            raise TypeError(f"path参数必须是字符串或字符串列表,但收到{type(self.path)}")

        self.samples = []
        for json_path in dataset_paths:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"数据集路径不存在: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.samples.extend(data) # 期待数据格式为[ {"path":"image1.jpg","label":0} ...]
        self.entry_path = ','.join(dataset_paths)
    
    def _len__(self) -> int:
        """
        返回数据集中的样本数量
        """
        return len(self.samples)

    def __getitem__(self,idx):
        """
        获取数据集中的一个样本
        Args:
            idx (int): 样本索引
        Returns:
            dict: 包含图像和标签的字典
        """
        sample = self.samples[idx]
        image_path = sample.get('path', None)
        label = sample.get('label', None)
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image)
        except Exception as e:
            print(f"[❌]无法加载图像 {image_path}: {e}")
            # 跳过当前样本, 递归获取下一个样本
            return self.__getitem__((idx+1)%len(self))
        
        # 如果有transform, 则应用transform
        if self.common_transform:
            out = self.common_transform(image=image)
            image = out['image']
        if self.post_transform:
            out = self.post_transform(image=image)
            image = out['image']
        
        # 构建输出字典
        output = {
            "image": image,
            "label": torch.tensor(label,dtype=torch.float)
        }

        # 应用任何后处理函数
        if self.post_funcs:
            self.post_funcs(output)
        
        return output

    def __str__(self):
        """
        返回数据集的字符串表示, 包含样本数量和标签分布
        """
        label_counts = {0: 0, 1: 0}
        for sample in self.samples:
            label = sample["label"]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1  # 如果出现了非0/1的标签，也记录

        return (f"CrossDataset from {self.entry_path}\n"
                f"Total samples: {len(self.samples)}\n"
                f"Label 0 samples (real): {label_counts.get(0, 0)}\n"
                f"Label 1 samples (fake): {label_counts.get(1, 0)}")
    
    
    

        


        


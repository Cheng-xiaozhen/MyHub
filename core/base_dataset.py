from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    所有datasets的baseclass
    该类定义了所有dataset类必须实现的接口
    它继承自torch.utils.data.Dataset，abc.ABC
    """

    def __init__(self,
                 path: Union[str, List[str]],
                 common_transform: Optional[Any] = None,
                 post_transform: Optional[Any] = None,
                 img_loader: Any = None,
                 post_funcs: Any = None,
                 **kwargs):
        """
        初始化dataset
        Args:
            path (str): 数据集路径
            common_transform (Optional[Any]): 作用于输入数据的变换
            post_transform (Optional[Any]): 作用于输入数据的后处理变换
            img_loader (Any): 图像加载函数
            post_funcs (Optional[List[callable]]): 后处理函数列表
            **kwargs: 每个数据集特有的其他参数
        """
        super().__init__()
        self.path = path
        self.common_transform = common_transform
        self.post_transform = post_transform
        self.img_loader = img_loader
        self.post_funcs = post_funcs

        # Initialize dataset paths
        self._init_dataset_path()

    @abstractmethod
    def _init_dataset_path(self) -> None:
        """
        初始化数据集路径
        该方法需要被每个数据集类实现，用于设置图像和标签
        """

        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        返回数据集中样本的数量
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据集中的一个样本
        Args:
            idx (int): 样本索引
        Returns:
            Dict[str, torch.Tensor]: 包含样本数据的字典
            字典至少包含:
            - 'image': torch.Tensor, 输入图像
            可选字段:
            - 'mask': torch.Tensor, 目标掩码
            - 'label': torch.Tensor, 目标标签
            - 'origin_shape': torch.Tensor, 原始图像形状
            - 'edge_mask': torch.Tensor, 图像的边缘掩码
            - ......
        """
        pass

    def __str__(self) -> str:
        """
        返回数据集的字符串表示
        """
        cls_name = self.__class__.__name__
        cls_path = self.path
        cls_len = len(self)
        return f"{cls_name}在{cls_path}，样本数量为{cls_len:,}"

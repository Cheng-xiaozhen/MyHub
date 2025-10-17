from abc import ABC, abstractmethod
from typing import Dict, Any
import albumentations as albu


class BaseTransform(ABC):

    """
    所有transforms的baseclass
    该类定义了所有transform类必须实现的接口
    子类应实现get_train_transform和get_test_transform方法
    以返回albumentations的Compose对象
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_train_transform(self) -> albu.Compose:
        """
        得到训练时的transform
        Returns:
            albu.Compose: 包含 train_transform的albumentations Compose对象
        """
        raise NotImplementedError("子类必须实现get_train_transform方法")

    @abstractmethod
    def get_test_transform(self) -> albu.Compose:
        """
        得到测试时的transform
        Returns:
            albu.Compose: 包含 test_transform的albumentations Compose对象
        """
        raise NotImplementedError("子类必须实现get_test_transform方法")

    @abstractmethod
    def get_post_transform(self) -> albu.Compose:
        """
        得到post_transform
        Returns:
            albu.Compose: 包含 post_transform的albumentations Compose对象
        """
        raise NotImplementedError("子类必须实现get_post_transform方法")

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        对data dictionary应用transform
        Args:
            data_dict (Dict[str, Any]): 包含输入数据的字典
                必须至少包含:
                    - 'image': 输入图像张量
                可能包含:
                    - 'mask': 真实掩码
                    - 'label': 真实标签
                    - 其他transform特定的输入
        Returns:
            Dict[str, Any]: 包含变换后数据的字典
                必须至少包含:
                    - 'image': 变换后的图像张量
                可能包含:
                    - 'mask': 变换后的掩码
                    - 'label': 变换后的标签
                    - 其他transform特定的输出
        """
        # 默认，使用training transforms
        transform = self.get_train_transform()

        # 对data_dict应用transform
        transformed = transform(**data_dict)

        return transformed

    def __str__(self) -> str:
        """
        返回transform的字符串表示
        """
        return self.__class__.__name__

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ...core.base_transform import BaseTransform
from ...core.registry import register_transform


@register_transform("CrossTransform")
class CrossTransform(BaseTransform):
    """
    跨数据集transform类, 提供通用的训练和测试transform
    """

    def __init__(self, output_size: tuple = (224, 224), norm_type='image_net'):
        """
        初始化transform类, 定义输出尺寸和归一化类型
        Args:
            output_size (tuple): 输出图像尺寸, 默认为(224, 224)
            norm_type (str): 归一化类型
        """
        super().__init__()
        self.output_size = output_size
        self.norm_type = norm_type

    def get_post_transform(self) -> albu.Compose:
        """
        post-processing transform, 包括归一化和转换为tensor
        Returns:
            albu.Compose: post-processing transform
        """
        if self.norm_type == 'image_net':
            # 使用ImageNet的均值和标准差进行归一化
            return albu.Compose([
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'clip':
            # 使用CLIP的均值和标准差进行归一化
            return albu.Compose([
                albu.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'standard':
            # 使用标准归一化, 将像素值映射到 [0, 1]
            return albu.Compose([
                albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'none':
            # 不进行归一化, 直接转换为tensor
            return albu.Compose([
                albu.ToFloat(max_value=255.0),  # 确保 uint8 转 float32，并映射到 [0, 1]
                ToTensorV2(transpose_mask=True)
            ])
        else:
            raise NotImplementedError("请使用 'image_net', 'clip', 'standard' or 'none' 作为 norm_type")

    def get_train_transform(self) -> albu.Compose:
        """
        训练时的transform
        ⚠ ⚠ ⚠ 需要根据具体数据集进行调整
        目前实现了一些常见的数据增强操作
        """
        return albu.Compose([
            # 水平和垂直翻转
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # 亮度和对比度调整
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower=70,
                quality_upper=100,
                p=0.2
            ),
            # 旋转90度
            albu.RandomRotate90(p=0.5),
            # 高斯模糊
            albu.GaussianBlur(
                blur_limit=(3, 7),
                p=0.2
            )
        ])

    def get_test_transform(self) -> albu.Compose:
        """
        测试时的transform, 待完善
        """
        return albu.Compose([
        ])

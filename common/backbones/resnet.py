import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractors.fph import FPH
from ...core.base_model import BaseModel
from ...core.registry import register_model
from .extractors.high_frequency_feature_extraction import FFTExtractor,DCTExtractor
from .extractors.sobel import SobelFilter
from .extractors.bayar_conv import BayerConv


class Resnet(BaseModel):
    def __init__(self,
                 input_head=None,
                 output_type='label',  
                 backbone='resnet101',  
                 pretrained=True,  
                 image_size=256,  # to set mask size when output_type is mask
                 num_channels=3):
        """
        Resnet backbone model
        Args:
            input_head: 允许在resnet的主干网络前插入一个额外的预处理器模块, 例如sobel算子, FFT提取器等
            output_type: str, 'label' or 'mask',输出类型
            backbone: str, 只支持'resnet50', 'resnet101' or 'resnet152'
            pretrained: bool, if true, 使用ImageNet-1k预训练权重
            image_size: int, 当output_type为'mask'时, 设置输出掩码的大小
            num_channels: int, 与input_head相关, 额外输入头的通道数
        """
        super(Resnet, self).__init__()
        assert backbone in ['resnet50', 'resnet101', 'resnet152'], "只支持 resnet50, resnet101 or resnet152"
        self.model = timm.create_model(backbone, pretrained=pretrained) # 创建ResNet模型
        self.backbone = self.model.forward_features # 获取ResNet模型的特征提取层
        out_channels = self.model.num_features # 获取ResNet模型的输出特征通道数
        self.head = None # 输出头
        self.output_type = output_type

        if output_type == 'label': # 用于分类的输出头
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )
        elif output_type == 'mask': # 用于生成掩码的输出头
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
            )
        else:
            raise ValueError(f"不支持的output_type: {output_type}")

        original_first_layer = list(self.model.children())[0]
        if input_head is not None: # 如果提供了额外的输入头, 修改ResNet的第一层卷积以适应额外的输入通道
            self.input_head = input_head
            new_first_layer = nn.Conv2d(num_channels + 3, original_first_layer.out_channels,
                                        kernel_size=original_first_layer.kernel_size,
                                        stride=original_first_layer.stride,
                                        padding=original_first_layer.padding, bias=False)
            new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
            if num_channels > 0:
                new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(
                    new_first_layer.weight[:, 3:, :, :])
            self.model.conv1 = new_first_layer
        else:
            self.input_head = None

    def forward(self, image, *args, **kwargs):
        # 1.如果有额外的输入头, 先通过该头处理输入图像
        if self.input_head is not None: 
            # 2.将处理后的特征与原始图像拼接
            feature = self.input_head(image)
            x = torch.cat([image, feature], dim=1)
        else:
            x = image
        
        # 3. 将拼接后的特征送入主干网络和输出头
        out = self.head(self.backbone(x))
        
        # 4. 根据输出类型计算损失和预测结果
        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()
        # 5. 构建输出字典
        out_dict = {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {
                "combined_loss": loss
            }
        }

        return out_dict


@register_model("Resnet50")
class Resnet50(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet50', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("Resnet101")
class Resnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("Resnet152")
class Resnet152(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type, backbone='resnet152', pretrained=pretrained,
                         image_size=image_size, num_channels=0)


@register_model("SobelResnet101")
class SobelResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=SobelFilter(), output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=1)


@register_model("BayerResnet101")
class BayerResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=BayerConv(), output_type=output_type, backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("FFTResnet101")
class FFTResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=FFTExtractor(), output_type=output_type, backbone='resnet101',
                         pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("DCTResnet101")
class DCTResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=DCTExtractor(), output_type=output_type, backbone='resnet101',
                         pretrained=pretrained,
                         image_size=image_size, num_channels=3)


@register_model("QtResnet101")
class QtResnet101(Resnet):
    def __init__(self, output_type='label', pretrained=True, image_size=256):
        super().__init__(input_head=None, output_type=output_type,
                         backbone='resnet101', pretrained=pretrained,
                         image_size=image_size, num_channels=0)
        self.fph = FPH()

        # 定位主干中间模块
        self.stem = nn.Sequential(
            self.model.conv1,  # conv1
            self.model.bn1,
            self.model.act1,
            self.model.maxpool
        )
        # 主干后续部分
        self.blocks = nn.Sequential(
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )

        # 通道融合（拼接后通道不一致）
        in_channels = self.model.layer1[0].conv1.in_channels + 256  # 通常是 64 + 256
        self.fusion_conv = nn.Conv2d(in_channels, self.model.layer1[0].conv1.in_channels, kernel_size=1)

    def forward(self, image, dct, qt, *args, **kwargs):
        dct = dct.squeeze(1).long()  # [B,1,H,W] -> [B,H,W]
        # FPH 特征（B, 256, H/8, W/8）
        x_aux = self.fph(dct, qt)

        # 主干前部分
        x = self.stem(image)

        # 若尺寸不一致，则插值对齐
        if x.shape[-2:] != x_aux.shape[-2:]:
            x_aux = F.interpolate(x_aux, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 拼接并降维融合
        x = torch.cat([x, x_aux], dim=1)
        x = self.fusion_conv(x)

        # 继续主干
        x = self.blocks(x)
        out = self.head(x)

        if self.output_type == 'label':
            if len(out.shape) == 2:
                out = out.squeeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(out, kwargs['label'].float())
            pred = out.sigmoid()
        else:
            loss = F.binary_cross_entropy_with_logits(out, kwargs['mask'].float())
            pred = out.sigmoid()

        return {
            "backward_loss": loss,
            f"pred_{self.output_type}": pred,
            "visual_loss": {"combined_loss": loss}
        }

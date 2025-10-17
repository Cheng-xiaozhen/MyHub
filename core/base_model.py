import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseModel(nn.Module):
    """
    所有models的baseclass
    该类定义了所有model类必须实现的接口
    该模型至少应以图像作为输入，并返回包含必要输出的字典
    """

    def __init__(self):
        super().__init__()

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            **kwargs (Dict[str, Any]): Dictionary containing input data.
                May contain (Consistent with the input from the dataset):
                    - 'image': Input image tensor
                    - 'mask': Ground truth mask
                    - 'label': Ground truth label
                    - Other model-specific inputs
                    
        Returns:
            Dict[str, Any]: Dictionary containing model outputs.
                Must contain at least:
                    - 'backward_loss': Backward loss value
                May contain:
                    - 'pred_mask': Predicted mask
                    - 'pred_label': Predicted label
                Visualize (Optional):
                    - 'visual_loss': Automatically visualize with the key-value pairs 
                    eg. {
                        "predict_loss": predict_loss,
                        "edge_loss": edge_loss,
                        "combined_loss": combined_loss
                    }
                    - 'visual_image': Automatically visualize with the key-value pairs
                    eg. {
                        "mask": mask,
                        "pred_mask": pred_mask,
                        "pred_label": pred_label
                    }
                    - Other model-specific outputs

                return eg.
                {
                    "backward_loss": combined_loss,

                    # optional below
                    "pred_mask": mask_pred,
                    "visual_loss": {
                        "combined_loss": combined_loss
                    },
                    "visual_image": {
                        "pred_mask": mask_pred,
                    }
                }
        """
        """
        模型的前向传播
        Args:
            **kwargs (Dict[str, Any]): 包含输入数据的字典
                可能包含（与数据集的输入一致）：
                    - 'image': 输入图像张量
                    - 'mask': 真实掩码
                    - 'label': 真实标签
                    - 其他模型特定的输入
        Returns:
            Dict[str, Any]: 包含模型输出的字典
                至少包含：
                    - 'backward_loss': 反向损失值
                可能包含：
                    - 'pred_mask': 预测掩码
                    - 'pred_label': 预测标签
                可视化（可选）：
                    - 'visual_loss': 自动可视化损失值，键值对形式
                    eg. {
                        "predict_loss": predict_loss,
                        "edge_loss": edge_loss,
                        "combined_loss": combined_loss
                    }
                    - 'visual_image': 自动可视化图像，键值对形式
                    eg. {
                        "mask": mask,
                        "pred_mask": pred_mask,
                        "pred_label": pred_label
                    }
                    - 其他模型特定的输出
                返回示例：
                {
                    "backward_loss": combined_loss,

                    # optional below
                    "pred_mask": mask_pred,
                    "visual_loss": {
                        "combined_loss": combined_loss
                    },
                    "visual_image": {
                        "mask": mask,
                        "pred_mask": pred_mask,
                        "pred_label": pred_label
                    }
                }
        """
        raise NotImplementedError("子类必须实现forward方法")

    def get_prediction(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        用于inference的方法
        在不计算损失的情况下获取模型预测
        Args:
            data_dict (Dict[str, Any]): 包含输入数据的字典
                必须至少包含：
                    - 'image': 输入图像张量
        Returns:
            Dict[str, Any]: 包含模型输出的字典
                必须至少包含：
                    - 'pred': 模型预测
        """
        with torch.no_grad():
            return self.forward(**data_dict)

    def compute_loss(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]) -> torch.Tensor:
        """
        用于training时计算loss的方法
        Args:
            data_dict (Dict[str, Any]): 包含输入数据的字典
            output_dict (Dict[str, Any]): 包含模型输出的字典
        Returns:
            torch.Tensor: 损失值
        """
        raise NotImplementedError("子类必须实现compute_loss方法")

    def get_metrics(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        用于evaluation时计算指标的方法
        Args:
            data_dict (Dict[str, Any]): 包含输入数据的字典
            output_dict (Dict[str, Any]): 包含模型输出的字典
        Returns:
            Dict[str, float]: 包含指标名称和值的字典
        """
        raise NotImplementedError("子类必须实现get_metrics方法")

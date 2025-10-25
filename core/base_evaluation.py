import torch.nn as nn
"""
下面这个类很重要，可以自动管理在多卡之间的一个具体数值变量的reduce（显卡之间归并数据）
可以用于实现算法的参考。
"""

"""
改变这个接口的主要目的是image-level的指标和pixel-level的指标的计算方式不同
"""
class BaseEvaluator(object): # 想了想没必要用nn.module 反而可能会引起一些其他的问题，比如梯度追踪，或者Parameter的追踪等等问题？？？（我有点困不是很确定）
    def __init__(self) -> None:
        self.name = None
        self.desc = None
        self.threshold = None
    def _check_pixel_level_params(self, predict, mask):
        if predict is None:
            raise ValueError(f"Detect none mask predict from the model, cannot calculate {self.name}. Please remove Pixel-level metrics from the script, or check the model output.")
        if mask is None:
            raise ValueError(f"Detect none mask label from the dataset, cannot calculate {self.name}. Please remove Pixel-level metrics from the script, or check the dataset output.")
    def _chekc_image_level_params(self, predict_label, label):
        if predict_label is None:
            raise ValueError(f"Detect none image-level predict label from the model, cannot calculate {self.name}. Please remove Image-level metrics from the script, or check the model output.")
        if label is None:
            raise ValueError(f"Detect none image-level binary label from the dataset, cannot calculate {self.name}. Please remove Image-level metrics from the script, or check the dataset output.")
    def batch_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """
        在每个batch结束时update。

        """
        raise NotImplementedError
    def remain_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """
        在每个batch结束时update。
        主要用于处理在最后一个batch之后的剩余数据。
        """
        raise NotImplementedError
    def epoch_update(self):
        """
        在遍历完整个数据集(包括所有批次和剩余数据)后, 调用该方法以进行最终的指标计算和更新。
        利用batch_update和remain_update中累积的数据，计算最终的评估指标。
        """
        raise NotImplementedError
    def recovery(self):
        """
        在完成一次完整的评估后,调用此方法将评估器内部的统计量清零
        """
        raise NotImplementedError




    

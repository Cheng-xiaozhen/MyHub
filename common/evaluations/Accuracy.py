import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...core.base_evaluation import BaseEvaluator
import torch.distributed as dist
import os
from ...training_scripts.utils import misc
from ...core.registry import register_evaluator

@register_evaluator("ImageAccuracyNoRemain")
class ImageAccuracyNoRemain(BaseEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__()
        self.name = "image-level Accuracy"
        self.desc = "image-level Accuracy"
        self.threshold = threshold
        self.true_cnt = torch.tensor(0.0, dtype=torch.float64, device='cuda')
        self.cnt = torch.tensor(0.0, dtype=torch.float64, device='cuda')

    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        predict = (predict_label > self.threshold).float().cuda()
        self.true_cnt += torch.tensor(torch.sum(predict * label).item() + torch.sum((1 - predict) * (1 - label)).item(), dtype=torch.float64, device='cuda')
        self.cnt += torch.tensor(len(label), dtype=torch.float64, device='cuda')
        return None

    def epoch_update(self):
        t = torch.tensor([self.true_cnt, self.cnt], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        true_cnt = t[0].item()
        cnt = t[1].item()
        acc = true_cnt / cnt
        return acc
    
    def recovery(self):
        self.true_cnt = torch.tensor(0.0, dtype=torch.float64, device='cuda')
        self.cnt = torch.tensor(0.0, dtype=torch.float64, device='cuda')

@register_evaluator("ImageAccuracy")
class ImageAccuracy(BaseEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__() 
        self.name = "image-level Accuracy"
        self.desc = "image-level Accuracy"
        self.threshold = threshold
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        self.world_size = misc.get_world_size()
        self.local_rank = misc.get_rank()

    def batch_update(self, predict_label, label, *args, **kwargs):
        """
        数据收集 每个gpu保存自己的数据，等epoch结束后再汇总计算
        """
        self._chekc_image_level_params(predict_label, label)
        self.predict.append(predict_label)
        self.label.append(label)
        return None
        
    def remain_update(self, predict_label, label, *args, **kwargs):
        """
        收集尾部数据 每个gpu保存自己的数据，等epoch结束后再汇总计算
        """
        self._chekc_image_level_params(predict_label, label)
        self.remain_predict.append(predict_label)
        self.remain_label.append(label)
        return None

    def epoch_update(self):
        """
        分布式计算环境中,无法直接在每个进程中计算Accuracy, 因为每个进程只包含了部分数据
        先收集汇总数据, 然后在主进程中计算Accuracy
        """
        if len(self.predict) != 0:
            # 1. 本地聚合 每个gpu将自己的数据拼接起来
            predict = torch.cat(self.predict, dim=0) # 将list中的tensor拼接成一个大的tensor
            label = torch.cat(self.label, dim=0) # 将list中的tensor拼接成一个大的tensor

            # 2. 跨gpu聚合 使用all_gather将每个gpu的数据收集到一起
            gather_predict_list = [torch.zeros_like(predict) for _ in range(self.world_size)] # 创建一个list,长度等于gpt数, 元素为和predict形状相同的全0 tensor
            gather_label_list = [torch.zeros_like(label) for _ in range(self.world_size)]

            dist.all_gather(gather_predict_list, predict)  # 广播,所有gpu广播并收集predict张量, 按rank存放在gather_predict_list中
            dist.all_gather(gather_label_list, label)

            gather_predict = torch.cat(gather_predict_list, dim=0) # 将收集到的数据片段合并
            gather_label = torch.cat(gather_label_list, dim=0) 
            # 3. 合并尾部数据
            if len(self.remain_predict) != 0:
                self.remain_predict = torch.cat(self.remain_predict, dim=0)
                self.remain_label = torch.cat(self.remain_label, dim=0)
                gather_predict = torch.cat([gather_predict, self.remain_predict], dim=0)
                gather_label = torch.cat([gather_label, self.remain_label], dim=0)
        else:
            # 处理drop_last 被丢弃的尾部数据
            if len(self.remain_predict) == 0:
                raise RuntimeError(f"没有数据可用于计算 {self.name}, 请检查.")
            gather_predict = torch.cat(self.remain_predict, dim=0) 
            gather_label = torch.cat(self.remain_label, dim=0)
        # 4. 最终计算
        binary_predict = (gather_predict > self.threshold).float()
        correct = (torch.sum(binary_predict == gather_label)).sum().item()
        total = gather_predict.shape[0]
        acc = correct / total
        return acc
    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        return None

@register_evaluator("PixelAccuracy")
class PixelAccuracy(BaseEvaluator):
    def __init__(self,threshold = 0.5, mode="origin") -> None:
        super().__init__()
        self.name = "pixel-level Accuracy"
        self.desc = "pixel-level Accuracy"
        self.threshold = threshold
        #  mode : "origin, reverse, double"
        self.mode = mode

    def Cal_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum(predict * mask, dim=(1, 2, 3))  
            TN = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3)) 
            FP = torch.sum(predict * (1-mask), dim=(1, 2, 3)) 
            FN = torch.sum((1-predict) * mask, dim=(1, 2, 3))         
        return TP, TN, FP, FN

    def Cal_Reverse_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum((1-predict) * mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask), dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3))
            FN = torch.sum(predict * mask, dim=(1, 2, 3))
        return TP, TN, FP, FN
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)
        if self.mode == "origin":
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            ACC = (TP + TN)/(TP + TN + FP + FN)
        elif self.mode == "reverse":
            TP, TN, FP, FN = self.Cal_Reverse_Confusion_Matrix(predict, mask, shape_mask)
            ACC = (TP + TN)/(TP + TN + FP + FN)
        elif self.mode == "double":
            # TODO
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            ACC = torch.max((TP + TN)/(TP + TN + FP + FN), (FP + FN)/(TP + TN + FP + FN))
        else:
            raise RuntimeError(f"Cal_ACC no mode name {self.mode}")
        # print("ACCCCCCCCCC",ACC)
        return ACC
    
    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask=None, *args, **kwargs)
    
    def epoch_update(self):

        return None
    
    def recovery(self):
        return None


def test_origin_image_ACC():
    import os
    import torch
    import torch.distributed as dist
    
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    if dist.get_rank() == 0:
        print("number of GPUs", num_gpus)
    
    DATA_LEN = 200
    float_tensor = torch.rand(DATA_LEN * num_gpus).cuda(local_rank)  # 生成一个长度为 200*num_gpus 的浮点数 tensor
    int_tensor = torch.randint(0, 2, (DATA_LEN * num_gpus,)).cuda(local_rank)  # 生成一个包含 0 或 1 的整数 tensor
    
    evaluator = ImageAccuracy(threshold=0.5)
    dist.barrier()
    dist.broadcast(float_tensor, src=0)
    dist.broadcast(int_tensor, src=0)
    
    all_predicts = []
    all_labels = []
    
    idx = dist.get_rank() * DATA_LEN
    if dist.get_rank() != num_gpus - 1:
        predict_labels = float_tensor[idx: idx + DATA_LEN].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN].cuda(local_rank)
    else:
        predict_labels = float_tensor[idx: idx + DATA_LEN - 50].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN - 50].cuda(local_rank)
    
    if dist.get_rank() == 0:  # 只在 rank 0 进程中收集数据
        all_predicts = float_tensor[:-50].cpu().numpy()
        all_labels = int_tensor[:-50].cpu().numpy()
        all_pred = (all_predicts > 0.5).astype(float)
        acc = (sum(all_pred * all_labels) + sum((1 - all_pred) * (1 - all_labels))) / len(all_pred)
        print(f"Calculated Accuracy: {acc}")
    
    # 运行 batch_update 更新统计数据
    evaluator.batch_update(predict_labels, true_labels)
    
    # 模拟一个 epoch 结束，调用 epoch_update 来计算 Accuracy
    gpu_acc = evaluator.epoch_update()
    
    if dist.get_rank() == 0:
        print(f"ACC Score: {gpu_acc}")
    
    # 清理分布式环境
    dist.destroy_process_group()


def test_pixal_ACC():
    batch_size, channels, height, width = 4, 1, 10, 10
    predict = torch.rand(batch_size, channels, height, width)
    mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # 生成一个 shape_mask
    shape_mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    # shape_mask = None
    acc = PixelAccuracy(mode="origin")
    acc_value_pytorch = acc.batch_update(predict, mask, shape_mask)
    print(acc_value_pytorch)

if __name__ == "__main__":
    # from sklearn.metrics import accuracy_score
    # print("test Image F1")
    # image_F1 = Image_Accuracy(threshold=0.5)
    # predict = torch.tensor([[0.9], [0.3], [0.4]])
    # label = torch.tensor([[1],[0],[1]])
    # print("my Acc:",image_F1(predict, label))

    # predict_binary = predict >= 0.5
    # # 将torch张量转换为numpy数组，因为sklearn的函数期望输入是numpy数组
    # predict_binary_np = predict_binary.numpy()
    # labels_np = label.numpy()

    # # 使用sklearn计算F1分数
    # acc = accuracy_score(labels_np, predict_binary_np)

    # print("Test Image Acc:", acc)
    # test_origin_image_ACC()
    test_pixal_ACC()
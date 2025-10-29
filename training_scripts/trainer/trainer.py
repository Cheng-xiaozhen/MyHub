import torch
from typing import Iterable
from contextlib import nullcontext


from ..utils import misc
from ..utils.cos_lr_schedular import adjust_learning_rate


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将归一化后的图像恢复到原始范围
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer=None,
                    log_per_epoch_count=20,
                    args=None):
    """
    训练模型一个epoch
    Args:
        model: 待训练的模型, 可能是DDP模型, 也可能是普通模型
        data_loader: 训练数据加载器
        optimizer: 优化器
        device: 训练设备
        epoch: 当前epoch数
        loss_scaler: 损失缩放器, 用于混合精度训练
        log_writer: 日志记录器, 用于记录训练过程中的指标
        log_per_epoch_count: 每个epoch中记录日志的次数
        args: 其他训练参数
    Returns:
        dict: 包含训练过程中各指标的平均值
    """
    
    model.train(True)# 切换模型到训练模式, 启用Dropout和BatchNorm等训练特性
    metric_logger = misc.MetricLogger(delimiter="  ") # 创建metric logger记录器, 用于记录和计算训练过程中的各种指标
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if args.if_amp: # 是否启用混合精度训练
        amp_placeholder = torch.cuda.amp.autocast() # 启用自动混合精度上下文管理器
        print("Auto Mixed Precision (AMP) is enabled.")
    else:
        amp_placeholder = nullcontext() # 空上下文管理器, 不进行任何操作
        print("Auto Mixed Precision (AMP) is disabled.")

    accum_iter = args.accum_iter # 梯度累积的迭代次数
    optimizer.zero_grad() # 清空梯度
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir)) # 打印日志目录
        
    total_step = len(data_loader) # 计算总的训练步数
    log_period = total_step / log_per_epoch_count # 每个epoch中记录日志的时间间隔

    # Start training
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 将数据移动到指定设备 
        for key in data_dict.keys(): 
            # Dataset的item格式应该是{'image':tensor,'label':tensor}
            # Dataloader会从Dataset中取出batch size个样本, 使用collate_fn合并成一个batch
            # 此时data_dict的格式应该是 {'image': <Tensor of shape [B, 3, 256, 256]>,'label': <Tensor of shape [B]>}
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        # 调整学习率
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize() # 同步CUDA设备, 确保所有前面的CUDA操作都完成
        
        with amp_placeholder:# 混合精度训练上下文管理器, 自动管理混合精度训练
            output_dict = model(
                **data_dict, # 解包为关键字参数 image, label
                if_predcit_label = args.if_predict_label
                )
            
            loss = output_dict['backward_loss']
            if output_dict.get('pred_mask') is not None:
                mask_pred = output_dict['pred_mask']

            visual_loss = output_dict['visual_loss']
            visual_loss_item = {}
            for k, v in visual_loss.items():
                visual_loss_item[k] = v.item()

            if output_dict.get('visual_image') is not None:
                visual_image = output_dict['visual_image']

            predict_loss = loss / accum_iter # 梯度累积, 多次计算梯度再进行一次参数更新, 模拟一个更大的batch
        
        # 1.损失缩放 2. 反向传播 3.更新参数
        loss_scaler(
            predict_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
            )
                
        if (data_iter_step + 1) % accum_iter == 0: # 一个累计周期结束后,清空梯度
            optimizer.zero_grad() 

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]
        # 更新metric logger
        # 当循环满足打印条件时, 在控制台打印
        # Epoch: [0]  [  20/1000]  combined_loss: [local: 0.6850 | reduced: 0.6901]  lr: 0.000100
        metric_logger.update(lr=lr)
        metric_logger.update(**visual_loss_item)
        
        # 记录tensorboard日志
        visual_loss_reduced = {}
        for k, v in visual_loss_item.items():
            visual_loss_reduced[k] = misc.all_reduce_mean(v) # 收集所有进程的指标并计算平均值

        if log_writer is not None and (data_iter_step + 1) % max(int(log_period), 1) == 0:
            """ 
            在tensorboard 中使用 epoch_1000x 作为 x 轴。
            细粒度化x轴, 便于观察每个epoch中的训练变化趋势
            例如 epoch_1000x = 500 表示 epoch 0.5,第0个epoch进行了50%的进度
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            for k, v in visual_loss_reduced.items():
                log_writer.add_scalar(f"train_loss/{k}", v, epoch_1000x)

    if data_dict.get('image') is not None:
        samples = data_dict['image']
    if data_dict.get('mask') is not None:
        mask = data_dict['mask']

    if log_writer is not None:
        if data_dict.get('image') is not None:
            log_writer.add_images('train/image', denormalize(samples), epoch)
        if output_dict.get('pred_mask') is not None:
            log_writer.add_images('train/predict', mask_pred, epoch)
            log_writer.add_images('train/predict_thresh_0.5', (mask_pred > 0.5) * 1.0, epoch)
        if data_dict.get('mask') is not None:
            log_writer.add_images('train/gt_mask', mask, epoch)

        if output_dict.get('visual_image') is not None:
            for k, v in visual_image.items():
                log_writer.add_images(f'train/{k}', v, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes() # 同步所有进程的指标, 确保在每个进程中都有相同的指标值
    print("Averaged stats:", metric_logger) # 打印每个进程的指标平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



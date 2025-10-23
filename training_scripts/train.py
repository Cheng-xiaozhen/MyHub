import os
import json
import time
import argparse
import datetime
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard.writer import SummaryWriter
from ..core.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, EVALUATORS, build_from_registry

from .utils import misc
from .utils.yaml import load_yaml_config,split_config, add_attr
from .trainer import train_one_epoch
from .tester import test_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('ForensicHub benchmark training launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = split_config(config)
    add_attr(args, output_dir=args.log_dir)
    add_attr(args, if_not_amp=not args.use_amp)
    return args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args


def main(args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args):
    # 初始化分布式训练环境
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system') # 使用file_system策略, 缓解dataloader多num_workers加载数据时的共享内存不足问题
    print('当前脚本所在目录: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    # 初始化随机种子
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 初始化transform, 用于数据预处理和增强
    transform = build_from_registry(TRANSFORMS, transform_args)
    train_transform = transform.get_train_transform()
    test_transform = transform.get_test_transform()
    post_transform = transform.get_post_transform()

    print("Train transform: ", train_transform)
    print("Test transform: ", test_transform)
    print("Post transform: ", post_transform)

    # 初始化post function, 用于后处理模型输出
    post_function_name = f"{model_args['name']}_post_func".lower()
    if model_args.get('post_func_name') is not None:
        post_function_name = f"{model_args['post_func_name']}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    print(post_function)

    # 初始化数据集
    # 多个数据集混合在一起组成训练集
    train_dataset_args["init_config"].update({
        "post_funcs": post_function,
        "common_transform": train_transform,
        "post_transform": post_transform
    })
    train_dataset = build_from_registry(DATASETS, train_dataset_args)

    # 测试集是单独分开的多个数据集
    test_dataset_list = {}
    for test_args in test_dataset_args:
        test_args["init_config"].update({
            "post_funcs": post_function,
            "common_transform": test_transform,
            "post_transform": post_transform
        })
        test_dataset_list[test_args["dataset_name"]] = build_from_registry(DATASETS, test_args)

    print(f"Train dataset: {train_dataset_args['dataset_name']}.")
    print(len(train_dataset))
    print(str(train_dataset))
    print(f"Test dataset: {[args['dataset_name'] for args in test_dataset_args]}.")
    print([len(dataset) for dataset in test_dataset_list.values()])

    # 初始化数据加载器和采样器
    test_sampler = {}
    if args.distributed:
        num_tasks = misc.get_world_size() # 总进程数
        global_rank = misc.get_rank() # 当前进程编号

        # 初始化采样器, 用于分布式训练时划分数据集, 每个进程只处理自己负责的那一份数据
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, 
            num_replicas=num_tasks, # 告诉DistributedSampler要将数据集划分为多少份
            rank=global_rank,# 根据rank参数, 选择当前进程负责处理哪一份数据
            shuffle=True
        )
        # 初始化测试集采样器, 测试集没有混合,所以是多个独立的采样器
        for test_dataset_name, dataset_test in test_dataset_list.items():
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=True
            )
            test_sampler[test_dataset_name] = sampler_test
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        for test_dataset_name, dataset_test in test_dataset_list.items():
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
            test_sampler[test_dataset_name] = sampler_test
        global_rank = 0

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)  # 初始化TensorBoard日志记录器
    else:
        log_writer = None

    # 初始化数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # 初始化测试集数据加载器
    test_dataloaders = {}
    for test_dataset_name in test_sampler.keys():
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset_list[test_dataset_name], sampler=test_sampler[test_dataset_name],
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        test_dataloaders[test_dataset_name] = test_dataloader

    # 初始化模型, 从注册表中构建模型实例
    model = build_from_registry(MODELS, model_args)

    # 初始化evaluator, 用于评估模型性能
    evaluator_list = []
    for eva_args in evaluator_args:
        evaluator_list.append(build_from_registry(EVALUATORS, eva_args))
    print(f"Evaluators: {evaluator_list}")

    # 将 普通BatchNorm 转换为 同步BatchNorm 
    # BatchNorm层会计算当前batch的均值和方差, 在分布式训练中, 每个GPU上的BatchNorm层只看到自己那一小部分数据, 可能导致统计量不准确
    # SyncBatchNorm会在所有GPU之间同步这些统计量, 使得每个GPU上的BatchNorm层都使用整个batch的数据来计算均值和方差
    # 类似梯度的同步
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device) # 将模型移动到指定的设备(如GPU)上

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    # 计算实际有效批量大小, 有效批次大小 = 每个GPU的批次大小 * 梯度累积步数 * GPU数量
    # 用于调整学习率等超参数,
    # 理论基础是学习率的线性缩放规则 (Linear Scaling Rule), 当batch size增加时, 学习率也应按比例线性增加
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # 如果没有指定学习率, 则根据有效批量大小自动计算学习率
    if args.lr is None:  
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    
    # 包装分布式数据并行模型
    if args.distributed:
        print(f"master port: {os.environ['MASTER_PORT']}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module # 从DDP模型中提取原始模型, 以便后续保存和加载模型权重等, 同时和非分布式变量保持一致

    # 初始化优化器, 这里使用AdamW优化器
    # TODO 后续将优化器设置放入yaml中
    args.opt = 'AdamW'
    args.betas = (0.9, 0.999)
    args.momentum = 0.9
    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount() # 初始化损失缩放器, 用于自动混合精度(AMP)训练, 防止梯度下溢

    # 从检查点恢复，如果提供了arg.resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 开始训练和测试循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0
    for epoch in range(args.start_epoch, args.epochs):
        
        optimizer.zero_grad()

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch) # 以epoch作为随机种子，打乱数据顺序
        
        # train for one epoch
        train_stats = train_one_epoch(
            model, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            log_writer=log_writer,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args
        )

        # saving checkpoint
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, 
                model=model, 
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler, 
                epoch=epoch
                )
        
        # test for one epoch
        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            values = {}  # dict of dict (dataset_name: {metric_name: metric_value})
            # test across all datasets in the `test_data_loaders' dict
            for test_dataset_name, test_dataloader in test_dataloaders.items():
                print(f'!!!Start Test: {test_dataset_name}', len(test_dataloader))
                test_stats = test_one_epoch(
                    model,
                    data_loader=test_dataloader,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=epoch,
                    name=test_dataset_name,
                    log_writer=log_writer,
                    args=args,
                    is_test=False,
                )
                one_metric_value = {}
                # Read the metric value from the test_stats dict
                for evaluator in evaluator_list:
                    evaluate_metric_value = test_stats[evaluator.name]
                    one_metric_value[evaluator.name] = evaluate_metric_value
                values[test_dataset_name] = one_metric_value

            metrics_dict = {metric: {dataset: values[dataset][metric] for dataset in values} for metric in
                            {m for d in values.values() for m in d}}
            # Calculate the mean of each metric across all datasets
            metric_means = {metric: np.mean(list(datasets.values())) for metric, datasets in metrics_dict.items()}
            # Calculate the mean of all metrics
            evaluate_metric_value = np.mean(list(metric_means.values()))

            # Store the best metric value
            if evaluate_metric_value > best_evaluate_metric_value:
                best_evaluate_metric_value = evaluate_metric_value
                print(
                    f"Best {' '.join([evaluator.name for evaluator in evaluator_list])} = {best_evaluate_metric_value}")
                # Save the best only after record epoch.
                if epoch >= args.record_epoch:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)
            else:
                print(f"Average {' '.join([evaluator.name for evaluator in evaluator_list])} = {evaluate_metric_value}")
            # Log the metrics to Tensorboard
            if log_writer is not None:
                for metric, datasets in metrics_dict.items():
                    log_writer.add_scalars(f'{metric}_Metric', datasets, epoch)
                log_writer.add_scalar('Average', evaluate_metric_value, epoch)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, }
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = get_args_parser()
    parser = argparse.ArgumentParser('ForensicHub benchmark training launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = split_config(config)

    add_attr(args, output_dir=args.log_dir)
    add_attr(args, if_not_amp=not args.use_amp)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args)

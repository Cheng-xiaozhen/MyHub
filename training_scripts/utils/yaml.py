import yaml
from argparse import Namespace


def try_parse_value(v):
    """
    尝试把字符串解析成 float,如将 '1e-4' 这种字符串转为 float
    如果失败就返回原值,保持不变
    Args:
        v (Any): 待解析的值
    Returns:
        Any: 解析后的值
    """
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    elif isinstance(v, dict):
        return {k: try_parse_value(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [try_parse_value(item) for item in v]
    else:
        return v


def load_yaml_config(path):
    """
    加载 YAML 配置文件为 Python 字典
    Args:
        path (str): YAML 文件路径
    Returns:
        dict: 解析后的配置字典
    """
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return try_parse_value(raw)


def dict_to_namespace(d):
    """
    递归地将 dict 转换成 argparse.Namespace
    Args:
        d (dict): 待转换的字典
    Returns:
        Namespace: 转换后的 Namespace 对象
    """
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_yaml_as_namespace(path):
    """
    直接从 YAML 路径加载为 Namespace 对象
    Args:
        path (str): YAML 文件路径
    Returns:
        Namespace: 转换后的 Namespace 对象
    """
    config_dict = load_yaml_config(path)
    return dict_to_namespace(config_dict)


def split_config(config):
    """
    将配置字典划分为多个组件参数
    Args:
        config (dict): 包含所有组件参数的字典
    Returns:
        tuple: 包含多个 Namespace 对象的元组，每个对象对应一个组件参数
    """
    model_args = config.pop("model", {})
    train_dataset_args = config.pop("train_dataset", {})
    test_dataset_args = config.pop("test_dataset", [])
    transform_args = config.pop("transform", {})
    evaluator_args = config.pop("evaluator", [])
    args = config  # 剩下的就是全局 args

    if "init_config" not in model_args:
        model_args["init_config"] = {}

    if "init_config" not in train_dataset_args:
        train_dataset_args["init_config"] = {}

    for x in test_dataset_args:
        if "init_config" not in x:
            x["init_config"] = {}

    if "init_config" not in transform_args:
        transform_args["init_config"] = {}

    for x in evaluator_args:
        if "init_config" not in x:
            x["init_config"] = {}

    return (
        dict_to_namespace(args),
        model_args,
        train_dataset_args,
        test_dataset_args,
        transform_args,
        evaluator_args
    )


def add_attr(ns, **kwargs):
    """
    向 Namespace 对象添加属性
    Args:
        ns (Namespace): 目标 Namespace 对象
        **kwargs: 要添加的属性键值对
    Returns:
        Namespace: 更新后的 Namespace 对象
    """
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns

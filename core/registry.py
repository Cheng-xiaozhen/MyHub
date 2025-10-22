import importlib
from collections.abc import Callable
from collections import abc
from typing import Dict, List, Optional, Type, Union, Any

import difflib
from rich.console import Console
from rich.table import Table

from .base_dataset import BaseDataset
from .base_model import BaseModel
from .base_transform import BaseTransform
from .base_evaluation import BaseEvaluator


class Registry:    
    """
    一个注册器类，用于将字符串映射到类或函数
    该类允许注册和检索模块（类或函数）
    """

    def __init__(self, name: str):
        """
        初始化注册器
        Args:
            name (str): 注册器的名称
        """
        self._name = name
        # 存储模块的字典, {类名:类}
        self._module_dict: Dict[str, Type] = dict()

    def __len__(self):
        """
        返回注册的模块数量
        """
        return len(self._module_dict)

    def __contains__(self, key):
        """
        检查注册器是否包含指定的键
        让你可以使用 `in` 关键字检查键是否存在
        Args:
            key (str): 要检查的键
        Returns:
            bool: 如果键存在于注册器中，则为True，否则为False
        """
        return self.get(key) is not None

    def __repr__(self):
        """
        返回注册器的字符串表示,包含注册的模块名称和对象
        当你print注册器实例时,它会用rich库格式化输出,显示所有已注册的模块
        """
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @property
    def name(self):
        """
        返回注册器的名称
        """
        return self._name

    def has(self, name: str) -> bool:
        """
        检查注册器是否包含指定的名称
        Args:
            name (str): 要检查的名称
        Returns:
            bool: 如果名称存在于注册器中，则为True，否则为False
        """
        return name in self._module_dict

    @property
    def module_dict(self):
        """
        返回注册的模块字典
        """
        return self._module_dict

    def _suggest_correction(self, input_string: str) -> Optional[str]:
        """Suggest the most similar string from the registered modules."""
        """
        查找与输入字符串最相似的注册模块名称
        Args:
            input_string (str): 要匹配的输入字符串
        Returns:
            Optional[str]: 如果找到相似名称，则返回该名称，否则返回None
        """
        suggestions = difflib.get_close_matches(input_string, self._module_dict.keys(), n=1, cutoff=0.6)
        if suggestions:
            return suggestions[0]
        return None

    def get(self, name):
        """
        根据名称获取注册的模块（类或函数）
        Args:
            name (str): 要获取的模块名称
        Returns:
            type: 注册的模块（类或函数）
        """
        if name in self._module_dict:
            return self._module_dict[name]
        suggestion = self._suggest_correction(name)
        if suggestion:
            raise KeyError(f'"{name}" 未注册在 {self.name}. 你是否指的是 "{suggestion}"?')
        else:
            raise KeyError(f'"{name}" 未注册在 {self.name} 且没有相似的名称.')

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        """
        注册一个模块（类或函数）

        Args:
            module (type): 要注册的模块。通常是一个类或函数，但一般所有`Callable`都是可以接受的。
            module_name (str or list of str, optional): 要注册的模块名称。如果未指定，则使用类名。默认值为None。
            force (bool): 是否覆盖具有相同名称的现有类。默认值为False。
        """
        if not callable(module):
            raise TypeError(f'module 必须是可Callable的, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__ # 没有名字,就是用类名
        if isinstance(module_name, str):
            module_name = [module_name] # 统一处理为列表,方便一个类注册多个名字
        for name in module_name:
            if not force and name in self._module_dict:
                # 如果不强制覆盖,且名字已存在,报错
                existed_module = self.module_dict[name]
                raise KeyError(f'{name}已经注册在{self.name} '
                               f'在 {existed_module.__module__}')
            self._module_dict[name] = module # 把 名字-类 映射存入字典

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            module: Optional[Type] = None,
            force: bool = False) -> Union[type, Callable]:
        """
        注册一个模块, _register_module的封装
        'self._module_dict'中添加一条记录,键是类名或指定的名称,值是类本身, 它可以用作装饰器或普通函数

        Args:
            name (str or list of str, optional): 要注册的模块名称。如果未指定，则使用类名。
            module (type, optional): 要注册的模块。默认为N
            force (bool): 是否覆盖具有相同名称的现有类。默认值为False。
        Returns:
            Union[type, Callable]: 如果作为装饰器使用,返回被注册的类或函数本身
            如果作为普通函数使用,返回一个装饰器函数,该函数接受一个类或函数作为参数并注册它

        """

        if not isinstance(force, bool):
            raise TypeError(f'force 必须是boolen类型, but got {type(force)}')

        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(f'name 必须是None, 字符串, 或字符串序列, but got {type(name)}')
        
        # 作为普通函数使用
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module
        
        # 作为装饰器使用
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def build(self, name: dict, *args, **kwargs) -> Any:
        """
        根据名称构建模块实例
        Args:
            name (str): 要构建的模块名称
            *args: 传递给模块构造函数的位置参数
            **kwargs: 传递给模块构造函数的关键字参数
        Returns:
            Any: 构建的模块实例
        """
        return self.get(name)(*args, **kwargs) # 拿到类后,立刻传入参数实例化


# 创建注册器实例 类似于全局变量, 可以在不同文件中导入使用
MODELS = Registry(name='MODELS')
POSTFUNCS = Registry(name='POSTFUNCS')
DATASETS = Registry(name='DATASETS')
TRANSFORMS = Registry(name='TRANSFORMS')
EVALUATORS = Registry(name='EVALUATORS')

# 向注册器中注册基础类
MODELS.register_module(module=BaseModel, name='BaseModel')
DATASETS.register_module(module=BaseDataset, name='BaseDataset')
TRANSFORMS.register_module(module=BaseTransform, name='BaseTransform')
EVALUATORS.register_module(module=BaseEvaluator, name='BaseEvaluator')


# 添加便捷的装饰器函数
def register_model(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """
    注册一个模型类
    """
    return MODELS.register_module(name=name, force=force)

def register_dataset(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """
    注册一个数据集类
    """
    return DATASETS.register_module(name=name, force=force)


def register_transform(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """
    注册一个transform类
    """
    return TRANSFORMS.register_module(name=name, force=force)


def register_postfunc(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """
    注册一个post-function函数
    """
    return POSTFUNCS.register_module(name=name, force=force)


def register_evaluator(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """
    注册一个评估器类
    """
    return EVALUATORS.register_module(name=name, force=force)


def build_from_registry(registry, config_args):
    """"
    从注册器中构建实例
    Args:
        registry (Registry): 要使用的注册器实例
        config_args (dict): 包含类名和初始化参数的字典
    Returns:
        Any: 构建的实例
    """
    # 获取类名
    name = config_args["name"]
    if name in registry.module_dict.keys():
        cls = registry.get(name)
    else:
        cls = None

    # ========== 懒加载逻辑 ==========
    if cls is None:
        from .lazy_maps import get_all_lazy_map
        lazy_map = get_all_lazy_map()
        module_path = lazy_map.get(name,None)

        if module_path is None:
            raise ValueError(f"{name}没有找到源文件")
        print(f"[Lazy import] 从{module_path} 加载 {name}")
        importlib.import_module(module_path) # 动态加载
        cls = registry.get(name)
    
    if cls is None:
        raise ImportError(f"无法从{module_path} 加载 {name}")

    # 获取 config 字典中的参数
    if "init_config" in config_args:
        config = config_args.get("init_config", {})  # 从字典中获取 config 部分
    else:
        config = {}

    # 处理额外的参数：比如如果参数是字符串 "true"，可以转化为布尔值
    for k, v in config.items():
        if isinstance(v, str):
            if v.lower() == "true":
                config[k] = True
            elif v.lower() == "false":
                config[k] = False

    print(f"[build_from_registry] 创建模型 '{name}' 参数: {config}")
    return cls(**config)


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Optional[Type] = None) -> bool:
    """
    检查是否是某种类型的序列
    Args:
        seq (Any): 要检查的序列
        expected_type (type or tuple): 期望的序列item类型
        seq_type (type, optional): 期望的序列类型。默认值为None
    Returns:
        bool: 如果序列有效则返回True，否则返回False
    
    示例:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

if __name__ == "__main__":
    pass
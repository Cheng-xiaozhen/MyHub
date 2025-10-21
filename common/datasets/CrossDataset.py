import random
from ...core.base_dataset import BaseDataset
from ...core.registry import register_dataset, build_from_registry, DATASETS


@register_dataset("CrossDataset")
class CrossDataset(BaseDataset):
    """
    跨数据集包装器
    """
    def __init__(self, dataset_config=[], **kwargs):
        """
        Args:
            dataset_config (list): 数据集配置列表，包含多个数据集的配置信息
            **kwargs (dict): 其他初始化参数，会传递给每个子数据集
        """
        super().__init__(path='', **kwargs)
        self.datasets = []
        self.pic_nums = []
        self.dataset_names = []
        self.return_mask = True
        for config in dataset_config:
            config['init_config']['common_transform'] = self.common_transform
            config['init_config']['post_transform'] = self.post_transform
            config['init_config']['post_funcs'] = self.post_funcs
            dataset = build_from_registry(DATASETS, config)
            self.dataset_names.append(config['name'])
            self.datasets.append(dataset)
            self.pic_nums.append(config['pic_nums'])

        # 获取所有数据集样本中 key 的最小公共集合
        self.common_keys = self.get_common_keys(self.datasets)
        print(self.common_keys)

    def __len__(self):
        """
        样本数量总和
        """
        total_samples = sum(self.pic_nums)  
        return total_samples

    def _init_dataset_path(self) -> None:
        pass

    def get_common_keys(self, datasets):
        """
        获取所有数据集样本中 key 的最小公共集合
        Args:
            datasets (list): 数据集列表，每个元素是一个数据集实例

        Returns:
            set: 所有数据集样本中 key 的最小公共集合
        """
        try:
            common_keys = set(datasets[0][0].keys()) # 初始化为第一个数据集的第一个样本的 key 集合
            for dataset in datasets[1:]:
                sample = dataset[0]
                common_keys &= set(sample.keys())
            return common_keys
        except Exception as e:
            return set()  # 如果出错，返回空集合

    def __getitem__(self, index):
        """
        获取数据集中的样本
        Args:
            index (int): 样本索引

        Returns:
            dict: 包含样本数据的字典，键为公共键，值为样本值
        """
        try:
            cumulative_samples = 0
            for i, pic_num in enumerate(self.pic_nums):
                cumulative_samples += pic_num # 累积的 pic_num 可以帮助我们确定从哪个数据集抽取
                if index < cumulative_samples:  # 如果 index 在这个数据集的范围内
                    selected_dataset = self.datasets[i] # 选择对应的数据集
                    # 在当前数据集中随机选择一个样本
                    selected_item = random.randint(0, len(selected_dataset) - 1)
                    origin_out_dict = selected_dataset[selected_item]
                    origin_out_dict = {key: origin_out_dict[key] for key in self.common_keys if key in origin_out_dict}
                    origin_out_dict['label'] = origin_out_dict['label'].long()
                    return origin_out_dict
        except Exception as e:
            raise IndexError("索引超出数据集范围")

    def __str__(self):
        """
        返回数据集的字符串表示
        """
        info = f"<=== CrossDataset with {len(self.datasets)} datasets: {self.dataset_names} ===>\n"
        for i, ds in enumerate(self.datasets):
            info += f"\n[Dataset {i} - {self.dataset_names[i]}]\n"
            info += str(ds) + "\n"
        info += f"\nTotal samples per epoch: {self.__len__():,}\n"
        info += f"<================================================>\n"
        return info

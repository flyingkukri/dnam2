from datasets import load_from_disk
import transformers

from torch.utils.data import DataLoader
from omegaconf import DictConfig


def build_dna_dataloader(
    cfg: DictConfig,
    device_batch_size: int,
):
    assert cfg.name == 'dna', f'Tried to build dna dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset.py')

    assert cfg.dataset.local is not None, "No local dataset provided"
    dataset = load_from_disk(cfg.dataset.local)
    dataset = dataset[cfg.dataset.split]

    print("===ds===")
    print(dataset)
    print(device_batch_size)

    return DataLoader(
        dataset,
        batch_size=device_batch_size,
    )
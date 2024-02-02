from datasets import load_dataset
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

    # build streams
    streams_dict = cfg.dataset.get('streams', None)
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            streams.append(
                Stream(
                    remote=stream.get('remote', None) or
                    cfg.dataset.get('remote', None),
                    local=stream.get('local', None) or
                    cfg.dataset.get('local', None),
                    split=stream.get('split', None) or
                    cfg.dataset.get('split', None),
                    proportion=stream.get('proportion', None),
                    repeat=stream.get('repeat', None),
                    samples=stream.get('samples', None),
                    download_retry=stream.get('download_retry', None) or
                    cfg.dataset.get('download_retry', 2),
                    download_timeout=stream.get('download_timeout', None) or
                    cfg.dataset.get('download_timeout', 60),
                    validate_hash=stream.get('validate_hash', None) or
                    cfg.dataset.get('validate_hash', None),
                    keep_zip=stream.get('keep_zip', None) or
                    cfg.dataset.get('keep_zip', False),
                    keep_raw=stream.get('keep_raw', None) or
                    cfg.dataset.get('keep_raw', True),
                ))

    assert cfg.dataset.local is not None, "No local dataset provided"
    dataset = load_dataset(cfg.dataset.local)
    dataset = dataset[cfg.dataset.split]

    return DataLoader(
        dataset,
        batch_size=device_batch_size,
    )
import os
import re
import warnings

import torch

from transformer_modules import MilliTransNet


LATEST_CHECKPOINT_NAME = 'milli_transnet_latest.pth'
FINAL_CHECKPOINT_NAME = 'milli_transnet_final.pth'
SNAPSHOT_PATTERN = re.compile(r'milli_transnet_ep(\d+)\.pth$')


def _to_float_list(values):
    scalar_values = []
    for value in values:
        if isinstance(value, torch.Tensor):
            scalar_values.append(float(value.detach().cpu().item()))
        else:
            scalar_values.append(float(value))
    return scalar_values


def get_checkpoint_directory(path):
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def build_milli_transnet(config):
    model = MilliTransNet.MilliTransNet(
        config['NRANGES'],
        config['NDOPPLER'],
        config['LONG_CH_SIZE'],
        config['SHORT_CH_SIZE'],
        config['SEQ_LEN'],
        config['JOINTS_IDX'],
        config['SHAPE_CODE_SIZE'],
        config['N_LAYERS'],
        config['N_HEADS'],
        config['N_DROPOUT'],
        config['FWD_EXPANSION'],
        output_process=config['OUTPUT_PROCESSING'],
        initializer=config['MODEL_INIT'],
        optimizer=config['OPTIMIZER']
    )
    model.set_device(config['DEVICE'])
    return model


def resolve_checkpoint_path(path, prefer_final=False):
    if os.path.isfile(path):
        return path

    checkpoint_dir = get_checkpoint_directory(path)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f'Checkpoint path does not exist: {path}')

    preferred_names = (
        [FINAL_CHECKPOINT_NAME, LATEST_CHECKPOINT_NAME]
        if prefer_final
        else [LATEST_CHECKPOINT_NAME, FINAL_CHECKPOINT_NAME]
    )
    for checkpoint_name in preferred_names:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path

    epoch_snapshots = []
    for item in os.listdir(checkpoint_dir):
        match = SNAPSHOT_PATTERN.fullmatch(item)
        if match:
            epoch_snapshots.append((int(match.group(1)), os.path.join(checkpoint_dir, item)))

    if epoch_snapshots:
        return max(epoch_snapshots, key=lambda item: item[0])[1]

    raise FileNotFoundError(f'No MilliTransNet checkpoint found in {checkpoint_dir}')


def save_training_checkpoint(
    model,
    checkpoint_path,
    epoch,
    next_lr,
    next_wd,
    all_lr,
    all_wd,
    training_loss,
    validation_loss
):
    optimizer = model.get_optimizer()
    checkpoint = {
        'checkpoint_format_version': 1,
        'epoch': int(epoch),
        'next_lr': float(next_lr),
        'next_wd': float(next_wd),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'all_lr': _to_float_list(all_lr),
        'all_wd': _to_float_list(all_wd),
        'training_loss': _to_float_list(training_loss),
        'validation_loss': _to_float_list(validation_loss),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def load_milli_transnet_checkpoint(path, config, load_optimizer):
    checkpoint = torch.load(path, map_location=config['DEVICE'])

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = build_milli_transnet(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_state = checkpoint.get('optimizer_state_dict')
        if load_optimizer and optimizer_state is not None and model.get_optimizer() is not None:
            model.get_optimizer().load_state_dict(optimizer_state)

        model.set_device(config['DEVICE'])
        return model, checkpoint

    if isinstance(checkpoint, MilliTransNet.MilliTransNet):
        checkpoint.set_device(config['DEVICE'])
        warnings.warn(
            (
                f'Loaded legacy full-model checkpoint from {path}. '
                'Model weights are restored, but resumable training state may be incomplete.'
            ),
            stacklevel=2
        )
        return checkpoint, {
            'checkpoint_format_version': 0,
            'epoch': None,
            'next_lr': None,
            'next_wd': None,
            'all_lr': None,
            'all_wd': None,
            'training_loss': None,
            'validation_loss': None,
        }

    raise ValueError(f'Unsupported checkpoint format in {path}')

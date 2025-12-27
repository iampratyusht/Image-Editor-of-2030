import logging
import torch
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location=None, strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    # Auto-detect device: use CPU if CUDA not available, otherwise use CUDA
    if map_location is None:
        map_location = 'cpu' if not torch.cuda.is_available() else 'cuda'
    
    # PyTorch 2.6+ requires weights_only parameter
    # Set to False for compatibility with older checkpoint formats
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        state = torch.load(path, map_location=map_location)
    
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model

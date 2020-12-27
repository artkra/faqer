import logging
import pickle
from pathlib import Path
from django.conf import settings

import torch

from .models import EmbeddingModel
from .train import VOCAB_SIZE_KEY


logger = logging.getLogger(__file__)


def load_embedder(pkl_name=None) -> EmbeddingModel:
    pkl_dir = Path(settings.PICKLE_DIR)

    if pkl_name is None:
        model_paths = [f for f in pkl_dir.iterdir() if f.name.startswith(settings.EMBEDDER_MODEL_PREFIX)]
        if model_paths:
            pkl_name = sorted(model_paths)[0].name

    if pkl_name is None:
        logger.error('No embedder model found')
        return None

    logger.info(f'Loading embedder model: {pkl_name}')

    words_path = Path.joinpath(pkl_dir, f'{settings.VOCABULARY_PREFIX}{pkl_name}')
    if not words_path.exists():
        logger.info('No vocabulary for embedder found.')
        return None

    with open(words_path, 'rb') as f:
        words_dict = pickle.load(f)
        vocab_size = words_dict.get(VOCAB_SIZE_KEY, None)
        if vocab_size is None:
            logger.error(f'Failed to load vocabulary for embedder model {pkl_name}')
            return None
        try:
            model = EmbeddingModel(vocab_size)
            model.load_state_dict(
                torch.load(Path.joinpath(pkl_dir, pkl_name))
            )
            return model
        except Exception as e:
            logger.error(f'Failed to load embedder model {pkl_name}: {e}')
            return None

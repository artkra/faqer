import logging
import pickle
from time import time
from pathlib import Path

from django.conf import settings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import EmbeddingModel
from ..data.utils import get_text, tokenize_text


EPOCHS = 50
VOCAB_SIZE_KEY = '__SIZE'

logger = logging.getLogger(__file__)
torch.manual_seed(1)


def get_tokens():
    return tokenize_text(get_text())


def train(tokens):
    trigrams = [([tokens[i], tokens[i + 1]], tokens[i + 2])
                for i in range(len(tokens) - 2)]

    vocab = set(tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_function = nn.NLLLoss()
    model = EmbeddingModel(len(vocab))
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    TIKS = EPOCHS*len(trigrams)

    with tqdm(total=TIKS) as pbar:
        for _ in range(EPOCHS):
            total_loss = 0
            for context, target in trigrams:
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
                model.zero_grad()
                log_probs = model(context_idxs)
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)
            losses.append(total_loss)

    pkl_name = f'{settings.EMBEDDER_MODEL_PREFIX}.{int(time())}.pkl'

    torch.save(model.state_dict(), Path.joinpath(settings.PICKLE_DIR, pkl_name))
    with open(Path.joinpath(settings.PICKLE_DIR, f'{settings.VOCABULARY_PREFIX}{pkl_name}'), 'wb') as fd:
        word_to_ix[VOCAB_SIZE_KEY] = len(vocab)
        pickle.dump(word_to_ix, fd)
    return model, losses

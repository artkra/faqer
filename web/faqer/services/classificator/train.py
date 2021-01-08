import logging

from ..data.utils import get_text, tokenize_text


EPOCHS = 50
VOCAB_SIZE_KEY = '__SIZE'

logger = logging.getLogger(__file__)


def get_tokens(limit=None):
    text = get_text()
    if limit is not None:
        text = text[0:limit]
    return tokenize_text(text)


def train(tokens):
    pass

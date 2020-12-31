import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from django.conf import settings

import gensim
import torch
from torch.nn import PairwiseDistance
from navec import Navec
from slovnet.model.emb import NavecEmbedding

from core.utils.meta import Singleton
from .models import EmbeddingModel
from .train import VOCAB_SIZE_KEY
from faqer.services.data.utils import get_trigrams, stem_word


logger = logging.getLogger(__file__)


DISTANCE_THRESHOLD = float(3)


def load_embedder_with_vocabulary(pkl_name=None) -> Tuple[EmbeddingModel, dict]:
    pkl_dir = Path(settings.PICKLE_DIR)

    if pkl_name is None:
        model_paths = [f for f in pkl_dir.iterdir() if f.name.startswith(settings.EMBEDDER_MODEL_PREFIX)]
        if model_paths:
            pkl_name = sorted(model_paths, reverse=True)[0].name

    if pkl_name is None:
        logger.error('No embedder model found')
        return None, None

    logger.info(f'Loading embedder model: {pkl_name}')

    words_path = Path.joinpath(pkl_dir, f'{settings.VOCABULARY_PREFIX}{pkl_name}')
    if not words_path.exists():
        logger.info('No vocabulary for embedder found.')
        return None, None

    with open(words_path, 'rb') as f:
        words_dict = pickle.load(f)
        vocab_size = words_dict.get(VOCAB_SIZE_KEY, None)
        if vocab_size is None:
            logger.error(f'Failed to load vocabulary for embedder model {pkl_name}')
            return None, None
        try:
            model = EmbeddingModel(vocab_size)
            model.load_state_dict(
                torch.load(Path.joinpath(pkl_dir, pkl_name))
            )
            return model, words_dict
        except Exception as e:
            logger.error(f'Failed to load embedder model {pkl_name}: {e}')
            return None, None


class BaseDistCalculator:

    def dist_words(self, word1, word2) -> Optional[float]:
        raise NotImplementedError()


class RDTModel(BaseDistCalculator, metaclass=Singleton):
    
    def __init__(self) -> None:
        super().__init__()
        w2v_fpath = settings.RTD_MODEL_PATH
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
        self.w2v.init_sims(replace=True)
    
    def _prepare_word(self, word) -> str:
        return word.lower().strip()

    def dist_words(self, word1, word2) -> Optional[float]:
        word1 = self._prepare_word(word1)
        word2 = self._prepare_word(word2)
        if word1 not in self.w2v.vocab or word2 not in self.w2v.vocab:
            return None
        return self.w2v.distance(word1, word2)

    def get_synonyms(self, word, limit=3) -> List[str]:
        if word in self.w2v.vocab:
            return self.w2v.most_similar(word)[:limit]
        return []


class IOWDistCalculator(BaseDistCalculator):

    def __init__(self, model_name=None) -> None:
        super().__init__()
        self.model, self.vocab = load_embedder_with_vocabulary(model_name)
        self.rdt_model = RDTModel()

    def _prepare_word(self, word) -> str:
        word = word.lower().strip()
        if word in self.vocab:
            return word
        if stem_word(word) in self.vocab:
            return stem_word(word)
        synonyms = self.rdt_model.get_synonyms(word)
        for syn in synonyms:
            # syn is a tuple of word - probability
            syn = syn[0]
            if syn in self.vocab:
                return syn
            if stem_word(syn) in self.vocab:
                return stem_word(syn)

        return None

    def dist_words(self, word1, word2) -> Optional[float]:
        word1 = self._prepare_word(word1)
        word2 = self._prepare_word(word2)

        if not all([word1, word2]):
            return None
        return float(PairwiseDistance()(
            self.model.embeddings(torch.tensor([self.vocab[word1]], dtype=torch.long)),
            self.model.embeddings(torch.tensor([self.vocab[word2]], dtype=torch.long)),
        )[0])


class NavecDistCalculator(BaseDistCalculator, metaclass=Singleton):

    def __init__(self) -> None:
        super().__init__()
        self.navec = Navec.load(settings.NAVEC_DATAFILE_PATH)
        self.model = NavecEmbedding(self.navec)

    def _prepare_word(self, word) -> str:
        return word.lower().strip()

    def dist_words(self, word1, word2) -> Optional[float]:
        word1 = self._prepare_word(word1)
        word2 = self._prepare_word(word2)
        if word1 not in self.navec.vocab or word2 not in self.navec.vocab:
            return None
        return float(PairwiseDistance()(
            self.model(torch.tensor([self.navec.vocab.get(word1, self.navec.vocab.unk_id)])),
            self.model(torch.tensor([self.navec.vocab.get(word2, self.navec.vocab.pad_id)]))
        )[0])


class SentencesDistCalculator(metaclass=Singleton):

    def __init__(self) -> None:
        self.iow_calc = IOWDistCalculator()
        self.nav_calc = NavecDistCalculator()
        super().__init__()

    def reduce_sentence(self, calc: BaseDistCalculator, tokens: List[str]) -> Optional[torch.tensor]:
        return None

    def dist_sentences(self, sent1, sent2) -> Optional[torch.tensor]:
        sent1_iow_emb = self.reduce_sentence(self, self.iow_calc, sent1)
        sent2_iow_emb = self.reduce_sentence(self, self.iow_calc, sent2)

        sent1_nav_emb = self.reduce_sentence(self, self.nav_calc, sent1)
        sent2_nav_emb = self.reduce_sentence(self, self.nav_calc, sent2)

        dist_iow = PairwiseDistance()(sent1_iow_emb, sent2_iow_emb) \
            if all([sent1_iow_emb, sent2_iow_emb]) else None

        dist_nav = PairwiseDistance()(sent1_nav_emb, sent2_nav_emb) \
            if all([sent1_nav_emb, sent2_nav_emb]) else None

        return min([dist_iow, dist_nav])

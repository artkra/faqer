import logging

from typing import List, Optional

from django.conf import settings

import gensim

from core.utils.meta import Singleton


logger = logging.getLogger(__file__)


DISTANCE_THRESHOLD = float(3)


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

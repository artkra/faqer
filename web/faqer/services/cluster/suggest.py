import json
from copy import copy
import logging
from threading import Lock
from typing import List

from django.conf import settings
from gensim.summarization import keywords
from numpy import ndarray
from collections import Counter
from statistics import mean
from sklearn.cluster import DBSCAN

from faqer.services.data.utils import (
    MSG_FILE, get_lines, get_trigrams, to_include, tokenize_text
)
from faqer.services.classificator.eval import RDTModel


logger = logging.getLogger(__file__)


EPS = 0.99
MIN_SAMPLES = 19
MAX_KEYWORDS_PER_CLUSTER = 10


class CategoriesService:

    UNCATEGORIZED = {
        'id': None,
        'category_name': 'UNCATEGORIZED',
        'keywords': []
    }

    def __init__(self) -> None:
        self.rdt_calc = RDTModel()
        self.trigram_vectors = []
        self.trigrams = []
        self.labels = []
        self.n_clusters = 0
        self._suggest_threshold = 0.5
        self._categories = None
        self._suggested_categories = None
        self._cached_categories = None
        self.lock = Lock()

    def _prepare_keywords(self, filepath=None):
        if filepath is None:
            filepath = MSG_FILE
        text = ''
        for line in get_lines(filepath):
            if '?' in line:
                text += line

        self.kwds = set(keywords(text).split())

    def _prepare_vectors(self, filepath=None):
        self._prepare_keywords(filepath=filepath)
        for line in get_lines(filepath=filepath):
            if '?' not in line:
                continue
            tokens = tokenize_text(line, do_stem=False)
            tokens = [t for t in tokens if t in self.kwds]
    
            for input_trigram in get_trigrams(tokens):
                sum_trigram = sum([
                    self.rdt_calc.w2v.word_vec(w)
                    for w in input_trigram if w in self.rdt_calc.w2v.vocab
                ])
                if isinstance(sum_trigram, ndarray):
                    self.trigram_vectors.append(sum_trigram)
                    self.trigrams.append(input_trigram)

    def _clasterize(self, filepath=None):
        self._prepare_vectors(filepath=filepath)
        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(self.trigram_vectors)

        self.labels = dbscan.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def suggest_categories(self, filepath=None):
        self._clasterize(filepath=filepath)
        clusters_summary = []
        for c in range(self.n_clusters): 
            clust_words = Counter()
            for i, k in enumerate(self.labels):
                if k==c:
                    clust_words += Counter([x for x in self.trigrams[i] if to_include(x)])
            mean_freq = mean(list(clust_words.values()))
            clust_keywords = [w for w in clust_words.keys() if clust_words[w] > mean_freq]
            if len(clust_keywords) < MAX_KEYWORDS_PER_CLUSTER:
                # enrich with synonyms
                clusters_summary.append(clust_keywords)
        self.clusters_summary_keywords = clusters_summary
        return self.clusters_summary_keywords

    def load_categories(self):
        if self._categories is not None:
            raise ValueError('Categories already loaded. Instantiate another scanner.')
        base_cat_ids = set()
        cache_cat_ids = set()

        self._categories = {}
        self._suggested_categories = {}
        self._cached_categories = {}
        with open(settings.CATEGORIES_SUGGESTED_PATH, 'r') as f:
            categories = json.load(f)
            for cat in categories:
                base_cat_ids.add(cat['id'])
                self._categories[cat['id']] = cat
                self._suggested_categories[cat['id']] = cat

        with open(settings.CATEGORIES_CACHE_PATH, 'r') as f:
            try:
                cache_categories = json.load(f)
                for cat in cache_categories:
                    cache_cat_ids.add(cat['id'])
                    self._categories[cat['id']] = cat
                    self._cached_categories[cat['id']] = cat
            except Exception as e:
                logger.error(f'Failed to load cache categories data: {e}')

        # cache can remove unused categories
        for rm_id in base_cat_ids.difference(cache_cat_ids):
            del self._categories[rm_id]

        return self

    @property
    def categories(self):
        return [v for v in self._categories.values()]

    @property
    def suggested_categories(self):
        return [v for v in self._suggested_categories.values()]

    @property
    def cached_categories(self):
        return [v for v in self._cached_categories.values()]

    def update_categories(self, categories: List[dict]):
        with self.lock:
            for cat in categories:
                self._categories[cat['id']] = cat
            with open(settings.CATEGORIES_CACHE_PATH, 'w') as f:
                json.dump(self.categories, f)

    def predict_category(self, sentence):
        if len(self._categories) < 1:
            raise ValueError('No categories to suggest. Clasterize data first.')

        dists = []

        for i, cat in self._categories.items():
            clust_dists = []
            for word in tokenize_text(sentence, do_stem=False):
                for kwrd in cat.get('keywords', []):
                    dist = self.rdt_calc.dist_words(word, kwrd)
                    if dist is not None:
                        clust_dists.append(dist)
            if len(clust_dists) > 0:
                _min_dist = min(clust_dists)
                if _min_dist < self._suggest_threshold:
                    dists.append((i, _min_dist))
        if len(dists) > 0:
            suggested_category_tuple = sorted(dists, key=lambda x: x[1])[0]
            res = copy(self._categories.get(suggested_category_tuple[0]))
            res['distance'] = suggested_category_tuple[1]
            return res

        return self.UNCATEGORIZED


categories_service = CategoriesService().load_categories()

from gensim.summarization import keywords
from numpy import ndarray
from collections import Counter
from statistics import mean
from sklearn.cluster import DBSCAN

from faqer.services.data.utils import get_lines, get_trigrams, to_include, tokenize_text
from faqer.services.classificator.eval import RDTModel


EPS = 0.99
MIN_SAMPLES = 19
MAX_KEYWORDS_PER_CLUSTER = 10


class CategoriesScanner:

    def __init__(self) -> None:
        self.rdt_calc = RDTModel()
        self.trigram_vectors = []
        self.trigrams = []
        self.labels = []
        self.n_clusters = 0

        text = ''
        for line in get_lines():
            if '?' in line:
                text += line

        self.kwds = set(keywords(text).split())

    def prepare_vectors(self):
        for line in get_lines():
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

    def clasterize(self):
        self.prepare_vectors()
        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(self.trigram_vectors)

        self.labels = dbscan.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def suggest_categories(self):

        def enrich_with_synonyms(word):
            return [syn[0] for syn in self.rdt_calc.get_synonyms(word)]

        self.clasterize()
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
        self.clusters_summary = clusters_summary
        return self.clusters_summary

    def predict_cat(self, sentence):
        dists = []
        bag = set().union(*self.clusters_summary)
        for i, summary in enumerate(self.clusters_summary):
            clust_dists = []
            for word in tokenize_text(sentence, do_stem=False):
                if word in bag:
                    for kwrd in summary:
                        dist = self.rdt_calc.dist_words(word, kwrd)
                        if dist:
                            clust_dists.append(dist)
            dists.append((i, mean(clust_dists)))
        return dists

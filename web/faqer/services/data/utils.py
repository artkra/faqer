from collections import Counter
from pathlib import Path
from typing import List

from django.conf import settings
import nltk
from nltk.stem.snowball import RussianStemmer


STEMMER = RussianStemmer()
MSG_FILE = Path.joinpath(settings.FAQER_DATA_DIR, Path('slack/messages.txt'))
CUSTOM_STOPWORDS = ('http', 'https', 'это')


def get_text():
    with open(MSG_FILE, 'r') as fr:
        return fr.read()


def get_lines(filepath=None):
    if filepath is None:
        filepath = MSG_FILE
    with open(filepath, 'r') as fr:
        return fr.readlines() 


def stem_word(word):
    return STEMMER.stem(word)


def prepare_word(word):
    return word.strip().lower()


def get_trigrams(line: List[str]):
    return [[line[i], line[i + 1], line[i + 2]] for i in range(len(line) - 2)]


def to_include(word):
        if any([
            not word.isalpha(),
            word in nltk.corpus.stopwords.words('russian'),
            word in nltk.corpus.stopwords.words('english'),
            word in CUSTOM_STOPWORDS
        ]):
            return False
        return True


def tokenize_text(text, do_stem=True):
    if do_stem:
        return [stem_word(prepare_word(w)) for w in nltk.tokenize.word_tokenize(text) if to_include(w)]
    else:
        return [prepare_word(w) for w in nltk.tokenize.word_tokenize(text) if to_include(w)]


def count_freq(tokens):
    c = Counter(tokens)
    return sorted(c, key=lambda x: c[x])

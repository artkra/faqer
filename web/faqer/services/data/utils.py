from collections import Counter
from pathlib import Path

from django.conf import settings
import nltk
from nltk.stem.snowball import RussianStemmer


STEMMER = RussianStemmer()
MSG_FILE = Path.joinpath(settings.FAQER_DATA_DIR, Path('slack/messages.txt'))


def get_text():
    with open(MSG_FILE, 'r') as fr:
        return fr.read()


def get_lines():
    with open(MSG_FILE, 'r') as fr:
        return fr.readlines() 


def stem_word(word):
    return STEMMER.stem(word.lower())


def tokenize_text(text):
    def to_include(word):
        if any([
            not word.isalpha(),
            word in nltk.corpus.stopwords.words('russian'),
            word in nltk.corpus.stopwords.words('english')
        ]):
            return False
        return True

    tokens = [stem_word(w) for w in nltk.tokenize.word_tokenize(text) if to_include(w)]
    return tokens


def count_freq(tokens):
    c = Counter(tokens)
    return sorted(c, key=lambda x: c[x])

from collections import Counter
from pathlib import Path

from django.conf import settings
import nltk
from nltk.stem.snowball import RussianStemmer


def get_text():
    msg_file = Path.joinpath(settings.FAQER_DATA_DIR, Path('slack/messages.txt'))
    with open(msg_file, 'r') as fr:
        return fr.read()


def prepare_word(word):
    stemmer = RussianStemmer()
    return stemmer.stem(word.lower())


def tokenize_text(text):
    nltk.data.path.append(settings.FAQER_DATA_DIR)
    nltk.download('stopwords', settings.FAQER_DATA_DIR)
    nltk.download('punkt')

    def to_include(word):
        if any([
            not word.isalpha(),
            word in nltk.corpus.stopwords.words('russian'),
            word in nltk.corpus.stopwords.words('english')
        ]):
            return False
        return True

    tokens = [prepare_word(w) for w in nltk.tokenize.word_tokenize(text) if to_include(w)]
    return tokens


def count_freq(tokens):
    c = Counter(tokens)
    return sorted(c, key=lambda x: c[x])

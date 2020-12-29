from django.apps import AppConfig
from django.conf import settings
import nltk


class Config(AppConfig):
    name = 'faqer'

    def ready(self) -> None:
        nltk.data.path.append(settings.FAQER_DATA_DIR)
        nltk.download('stopwords', settings.FAQER_DATA_DIR)
        nltk.download('punkt')
        return super().ready()

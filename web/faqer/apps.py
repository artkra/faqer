from django.apps import AppConfig


class Config(AppConfig):
    name = 'faqer'

    def ready(self) -> None:
        return super().ready()

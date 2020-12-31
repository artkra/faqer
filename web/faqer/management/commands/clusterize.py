import logging

from django.core.management import BaseCommand

from faqer.services.cluster.suggest import CategoriesScanner


logger = logging.getLogger(__file__)


class Command(BaseCommand):
    help = 'Clusterize messages.'

    def handle(self, *args, **options):
        scanner = CategoriesScanner()
        logger.info(f'Suggested {len(scanner.suggest_categories())} question categories.')
        # TODO: save to categories.json
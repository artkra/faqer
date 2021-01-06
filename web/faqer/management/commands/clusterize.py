import json
import logging

from django.conf import settings
from django.core.management import BaseCommand

from faqer.services.cluster.suggest import CategoriesService


logger = logging.getLogger(__file__)


class Command(BaseCommand):
    help = 'Clusterize messages.'

    def handle(self, *args, **options):
        scanner = CategoriesService()
        categories_keywords = scanner.suggest_categories()
        logger.info(f'Suggested {len(categories_keywords)} question categories.')
        
        with open(settings.CATEGORIES_SUGGESTED_PATH, 'w') as f:
            categories = []
            for i, keywords in enumerate(categories_keywords):
                categories.append({
                    'id': i,
                    'name': '',
                    'keywords': keywords
                })

            json.dump(categories, f)
            logger.info(
                f'Suggested categories dumped in {settings.CATEGORIES_SUGGESTED_PATH}.'
            )

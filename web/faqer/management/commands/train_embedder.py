import logging

from django.core.management import BaseCommand

from faqer.services.classificator.train import get_tokens, train


logger = logging.getLogger(__file__)


class Command(BaseCommand):
    help = 'Synchronize the current state of ContentPackage objects from the external integrated system.'

    def handle(self, *args, **options):
        logger.info('Preparing tokens...')
        tokens = get_tokens()
        logger.info(f'Prepared {len(tokens)} tokens.')
        logger.info('Starting training.')
        train(tokens[0:10])
        logger.info('Model trained.')

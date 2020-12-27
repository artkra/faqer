import logging

from django.core.management import BaseCommand

from faqer.services.classificator.train import get_tokens, train


logger = logging.getLogger(__file__)


class Command(BaseCommand):
    help = 'Train base embedder.'

    def handle(self, *args, **options):
        logger.info('Preparing tokens...')
        tokens = get_tokens()
        logger.info(f'Prepared {len(tokens)} tokens.')
        logger.info('Starting training.')
        train(tokens[0:100])
        logger.info('Model trained.')

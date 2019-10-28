import logging

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

from .loader import LossMapFetcher

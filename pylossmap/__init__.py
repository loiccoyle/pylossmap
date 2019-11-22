try:
    from pkg_resources import get_distribution, DistributionNotFound
    __version__ = get_distribution(__name__).version
except (ImportError, DistributionNotFound):  # pragma: no cover
    __version__ = 'unknown'
import logging


logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

from .fetcher import BLMDataFetcher

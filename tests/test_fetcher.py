import unittest
import pandas as pd
from pylossmap import BLMDataFetcher

ITER_ADT_SHAPE = [((25, 3595), (571, 3595)),
                  ((25, 3595), (172, 3595))]
LOADER = BLMDataFetcher(mute=True)


class TestLoader(unittest.TestCase):

    def test_from_datetimes(self):
        t1 = pd.to_datetime('2018-06-20 00:00:00').tz_localize('Europe/Zurich')
        t2 = pd.to_datetime('2018-06-20 01:00:00').tz_localize('Europe/Zurich')
        BLM_data = LOADER.from_datetimes(t1, t2)
        self.assertEqual(BLM_data.df.shape, (3600, 3595))

    def test_from_fill(self):
        BLM_data = LOADER.from_fill(7247, beam_modes=['INJPROT'])
        self.assertEqual(BLM_data.df.shape, (1906, 3595))

    def test_bg_from_INJPROT(self):
        bg_data = LOADER.bg_from_INJPROT(7247)
        self.assertEqual(bg_data.df.shape, (403, 3595))

    def test_iter_from_ADT(self):
        t1 = pd.to_datetime('2018-04-08 23:00:00').tz_localize('Europe/Zurich')
        t2 = pd.to_datetime('2018-04-08 23:59:59').tz_localize('Europe/Zurich')
        adt_iter = LOADER.iter_from_ADT(t1,
                                        t2,
                                        look_back='5S',
                                        look_forward='20S',
                                        planes=['H', 'V'],
                                        beams=[1],
                                        yield_background=True,
                                        include=['trigger', 'amp', 'length', 'gate'])
        for (trigger, trigger_bg), (trigger_shape, trigger_bg_shape) in zip(adt_iter, ITER_ADT_SHAPE):
            self.assertEqual(trigger.df.shape, trigger_shape)
            self.assertEqual(trigger_bg.df.shape, trigger_bg_shape)
            # TODO: add check on context

    def test_bg_from_ADT_trigger(self):
        t1 = pd.to_datetime('2018-04-08 22:00:00').tz_localize('Europe/Zurich')
        bg_data = LOADER.bg_from_ADT_trigger(t1)
        self.assertEqual(bg_data.df.shape, (170, 3595))

    def test_logging_header(self):
        t1 = pd.to_datetime('2018-04-08 22:00:00').tz_localize('Europe/Zurich')
        blms = LOADER.fetch_logging_header(t1)
        self.assertEqual(len(blms), 3747)

    def test_force_header(self):
        t1 = pd.to_datetime('2018-04-08 22:00:00').tz_localize('Europe/Zurich')
        blms = LOADER.fetch_force_header(t1)
        self.assertEqual(len(blms), 3761)

    def test_timber_header(self):
        t1 = pd.to_datetime('2018-04-08 22:00:00').tz_localize('Europe/Zurich')
        blms = LOADER.fetch_timber_header(t1)
        self.assertEqual(len(blms), 3761)

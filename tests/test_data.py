import unittest
import pandas as pd
from pylossmap import BLMDataFetcher

LOADER = BLMDataFetcher(mute=True)
BLM_DATA = LOADER.from_fill(7247,
                            beam_modes=['INJPHYS'],
                            unique_beam_modes=True)


class TestBLMData(unittest.TestCase):

    def test_init(self):
        self.assertEqual(BLM_DATA.data.shape, (1687, 3595))

    def test_find_max(self):
        self.assertEqual(BLM_DATA.find_max().shape, (1, 8))

    def test_iter_max(self):
        expected = [(('INJPHYS',
                      pd.to_datetime('2018-10-02 10:14:24.736000061+0200').tz_convert('Europe/Zurich')),
                     ['BLMTI.06L3.B1I10_TCP.6L3.B1',
                      'BLMTI.06L7.B1E10_TCP.C6L7.B1',
                      'BLMTI.06L7.B1E10_TCP.B6L7.B1']),
                    (('INJPHYS',
                      pd.to_datetime('2018-10-02 10:21:31.855000019+0200').tz_convert('Europe/Zurich')),
                     ['BLMTI.06L7.B1E10_TCP.D6L7.B1']),
                    (('INJPHYS',
                      pd.to_datetime('2018-10-02 10:40:34.857000113+0200').tz_convert('Europe/Zurich')),
                     ['BLMTI.06R3.B2E10_TCP.6R3.B2']),
                    (('INJPHYS',
                      pd.to_datetime('2018-10-02 10:40:03.769000053+0200').tz_convert('Europe/Zurich')),
                     ['BLMTI.06R7.B2I10_TCP.B6R7.B2',
                      'BLMTI.06R7.B2I10_TCP.C6R7.B2',
                      'BLMTI.06R7.B2I10_TCP.D6R7.B2'])]
        self.assertEqual(list(BLM_DATA.iter_max()), expected)

    def test_get_beam_meta(self):
        out = BLM_DATA.get_beam_meta('intensity', beam=1)
        self.assertEqual(out.shape, (1686, 1))

        out = BLM_DATA.get_beam_meta('filling_scheme')
        self.assertEqual(out.iloc[0][0],
                         '25ns_2556b_2544_2215_2332_144bpi_20injV3')

        out = BLM_DATA.get_beam_meta('number_bunches', beam=1)
        self.assertEqual(out.iloc[0][0],
                         2556.0)

        # out = BLM_DATA.get_beam_meta('energy')
        # self.assertEqual()

    def test_lossmap(self):
        LM = BLM_DATA.loss_map(datetime='2018-10-02 10:14:15.771000147+02:00',
                               context={'foo': 'bar'},
                               background=BLM_DATA.data.mean())
        self.assertEqual(LM.data.shape, (3595, 3))
        self.assertEqual(LM.data['data'].mean(), 2.794673741307371e-07)
        self.assertEqual(LM.context, {'foo': 'bar'})
        self.assertEqual(LM.get_background().data['data'].mean(),
                         3.8321408915168095e-07)

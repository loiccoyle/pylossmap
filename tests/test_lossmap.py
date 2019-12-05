import unittest
import pandas as pd
from pylossmap import BLMDataFetcher

LOADER = BLMDataFetcher(mute=True)
BLM_DATA = LOADER.from_fill(7247,
                            beam_modes=['INJPHYS'],
                            unique_beam_modes=True)
LM = BLM_DATA.loss_map(datetime='2018-10-02 10:14:15.771000147+02:00',
                       context={'foo': 'bar'},
                       background=BLM_DATA.df.mean())


class TestLossMap(unittest.TestCase):
    def test_init(self):
        expected = pd.to_datetime('2018-10-02 10:14:15.771000147+02:00')\
            .tz_convert('Europe/Zurich')
        self.assertEqual(LM.datetime, expected)

        self.assertEqual(LM.df.shape, (3595, 3))
        self.assertEqual(LM.context, {'foo': 'bar'})

    def test_normalize(self):
        out = LM.normalize()
        self.assertEqual(out.df['data'].max(), 1.0)

    def test_filtering(self):
        self.assertEqual(LM.beam(1).df.shape, (1392, 3))
        self.assertEqual(LM.IR(7, 5).df.shape, (971, 3))
        self.assertEqual(LM.side('R').df.shape, (1813, 3))
        self.assertEqual(LM.cell(1, 2, 3).df.shape, (147, 3))
        self.assertEqual(LM.TCL().df.shape, (32, 3))
        self.assertEqual(LM.TCP().df.shape, (10, 3))
        self.assertEqual(LM.TCTP().df.shape, (16, 3))
        self.assertEqual(LM.TCS().df.shape, (36, 3))
        self.assertEqual(LM.TCLI().df.shape, (4, 3))
        self.assertEqual(LM.DS().df.shape, (99, 3))
        self.assertEqual(LM.type('coll').df.shape, (162, 3))

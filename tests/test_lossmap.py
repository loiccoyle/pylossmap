import unittest
import pandas as pd
from pylossmap import BLMDataFetcher

LOADER = BLMDataFetcher(mute=True)
BLM_DATA = LOADER.from_fill(7247,
                            beam_modes=['INJPHYS'],
                            unique_beam_modes=True)
LM = BLM_DATA.loss_map(datetime='2018-10-02 10:14:15.771000147+02:00',
                       context={'foo': 'bar'},
                       background=BLM_DATA.data.mean())


class TestLossMap(unittest.TestCase):
    def test_init(self):
        expected = pd.to_datetime('2018-10-02 10:14:15.771000147+02:00')\
            .tz_convert('Europe/Zurich')
        self.assertEqual(LM.datetime, expected)

        self.assertEqual(LM.data.shape, (3595, 3))
        self.assertEqual(LM.context, {'foo': 'bar'})

    def test_normalize(self):
        out = LM.normalize()
        self.assertEqual(out.data['data'].max(), 1.0)

    def test_filtering(self):
        self.assertEqual(LM.beam(1).data.shape, (1392, 3))
        self.assertEqual(LM.IR(7, 5).data.shape, (971, 3))
        self.assertEqual(LM.side('R').data.shape, (1813, 3))
        self.assertEqual(LM.cell(1, 2, 3).data.shape, (147, 3))
        self.assertEqual(LM.TCL().data.shape, (32, 3))
        self.assertEqual(LM.TCP().data.shape, (10, 3))
        self.assertEqual(LM.TCTP().data.shape, (16, 3))
        self.assertEqual(LM.TCS().data.shape, (36, 3))
        self.assertEqual(LM.TCLI().data.shape, (4, 3))
        self.assertEqual(LM.DS().data.shape, (99, 3))
        self.assertEqual(LM.type('coll').data.shape, (162, 3))

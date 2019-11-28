import unittest
import pandas as pd
import numpy as np
from pylossmap import utils


class TestUtils(unittest.TestCase):

    def test_uniquify(self):
        self.assertEqual(list(utils.uniquify(['A', 'A', 'B', 'B', 'C'])),
                         ['A', 'A_2', 'B', 'B_2', 'C'])

    def test_to_datetime(self):
        expected = pd.to_datetime('2019-11-22 16:57:26+0100')\
            .tz_convert('Europe/Zurich')
        self.assertEqual(utils.to_datetime(1574438246),
                         expected)

    def test_fill_from_time(self):
        out = utils.fill_from_time('2018-06-01 00:00:00')
        expected = {'fillNumber': 6748,
                    'startTime': 1527797940.62,
                    'endTime': 1527812753.788,
                    'beamModes': [{'mode': 'SETUP',
                                   'startTime': 1527799038.5219998,
                                   'endTime': 1527800145.6739998},
                                  {'mode': 'INJPROT',
                                   'startTime': 1527800145.676,
                                   'endTime': 1527807062.758},
                                  {'mode': 'INJPHYS',
                                   'startTime': 1527807062.7600002,
                                   'endTime': 1527809628.986},
                                  {'mode': 'NOBEAM',
                                   'startTime': 1527809628.988,
                                   'endTime': 1527812751.376},
                                  {'mode': 'CYCLING',
                                   'startTime': 1527812751.3779998,
                                   'endTime': 1527812753.788}]}
        self.assertEqual(out, expected)

    def test_beammode_from_time(self):
        out = utils.beammode_from_time('2018-06-01 00:00:00')
        expected = {'mode': 'INJPROT',
                    'startTime': 1527800145.676,
                    'endTime': 1527807062.758}
        self.assertEqual(out, expected)

    def test_beammode_to_df(self):
        inp = [{'mode': 'SETUP',
                'startTime': 1527799038.5219998,
                'endTime': 1527800145.6739998}]
        out = utils.beammode_to_df(inp)
        self.assertEqual(out['SETUP']['startTime'],
                         utils.to_datetime(1527799038.5219998))

        self.assertEqual(out['SETUP']['endTime'],
                         utils.to_datetime(1527800145.6739998))

    def test_row_from_time(self):
        times = [pd.to_datetime('2019-01-01 00:00:00').tz_localize('Europe/Zurich'),
                 pd.to_datetime('2019-01-02 00:00:00').tz_localize('Europe/Zurich'),
                 pd.to_datetime('2019-01-03 00:00:00').tz_localize('Europe/Zurich')]
        data = [1, 2, 3]
        data_2 = ['a', 'b', 'c']
        df = pd.DataFrame({'timestamp': times,
                           'data': data,
                           'data_2': data_2})
        df.set_index('timestamp', inplace=True)
        row = utils.row_from_time(df, '2019-01-02 01:00:00', method='nearest')
        self.assertEqual(row['data'], df.iloc[1]['data'])
        self.assertEqual(row['data_2'], df.iloc[1]['data_2'])

    def test_coll_meta(self):
        out = utils.coll_meta()
        self.assertEqual(out.shape, (98, 2))

        out = utils.coll_meta(augment_b2=False)
        self.assertEqual(out.shape, (49, 2))

    def test_angle_convert(self):
        self.assertEqual(utils.angle_convert(0), 0)
        self.assertEqual(utils.angle_convert(np.pi/2), 1)
        self.assertEqual(utils.angle_convert(np.pi/4), 0.5)
        self.assertEqual(utils.angle_convert(np.pi/6), 1/3)

    # def test_get_ADT(self):
    #     out = utils.get_ADT()
    #     pass

    def test_sanitize_t(self):
        inp = '2019-11-22 16:57:26+0100'
        self.assertEqual(utils.sanitize_t(inp),
                         pd.to_datetime(inp).tz_convert('Europe/Zurich'))

        inp = 1574438246
        self.assertEqual(utils.sanitize_t(inp),
                         utils.to_datetime(inp))

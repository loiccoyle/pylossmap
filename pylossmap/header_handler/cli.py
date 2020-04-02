import logging

from argparse import ArgumentParser

from .header_maker import HeaderMaker
from .header_manager import HeaderManager


def main():
    '''Quick cli interface to run HeaderMaker.
    '''

    parser = ArgumentParser()
    parser.add_argument('t', help='''\
Time at which to create de header.
    int or float: assumes utc time, converts to pd.Timestamp and to \
Europe/Zurich timezone.
    str: a pd.to_datetime compatible str, assumes utc, converts to pd.\
Timestamp.''')
    parser.add_argument('-t2',
                        help=('Same type logic as "t", if provided will '
                              'ignore any "look_forward" or "look_back" '
                              'arguments and use the provided "t" and "t2" '
                              'arguments.'))
    parser.add_argument('-f', '--look_forward',
                        help=('Look forward amount, time format string, "1M", '
                              '"4H", ...'),
                        type=str,
                        default='1H')
    parser.add_argument('-b', '--look_back',
                        help=('Look back amount, time format string, "1M", '
                              '"4H", ...'),
                        type=str,
                        default='1H')
    parser.add_argument('-n', '--n_jobs',
                        help='Number of parallel jobs.',
                        default=-1,
                        type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()

    if args.verbose == 1:
        logging.getLogger('pylossmap').setLevel(logging.INFO)
    elif args.verbose == 2:
        logging.getLogger('pylossmap').setLevel(logging.DEBUG)

    hm = HeaderMaker(args.t,
                     t2=args.t2,
                     look_back=args.look_back,
                     look_forward=args.look_forward,
                     n_jobs=args.n_jobs)
    header = hm.make_header()
    if args.save:
        hm = HeaderManager()
        hm.add_header(args.t, header)
    else:
        print(header)

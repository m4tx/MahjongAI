import logging
from os import listdir
from os.path import isfile, join

import numpy as np

from tenhou_log_utils.io import load_mjlog
from tenhou_log_utils.parser import parse_mjlog
from tenhou_log_utils.state import process_round

_LG = logging.getLogger(__name__)
logging.getLogger('tenhou_log_utils.parser').setLevel(logging.WARN)


def process_file(path, x_data_discard, y_data_discard, x_data_pon, y_data_pon, x_data_chii, y_data_chii, x_data_riichi, y_data_riichi):
    data = parse_mjlog(load_mjlog(path, bzip2=True))

    rounds = data['rounds']
    for round_data in rounds:
        process_round(round_data, x_data_discard, y_data_discard, x_data_pon, y_data_pon, x_data_chii, y_data_chii, x_data_riichi, y_data_riichi)


def load_data(path):
    x_data_discard = []
    y_data_discard = []
    x_data_pon = []
    y_data_pon = []
    x_data_chii = []
    y_data_chii = []
    x_data_riichi = []
    y_data_riichi = []

    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for i, file in enumerate(files, 1):
        print(f'[{i}/{len(files)}] Processing {file}')
        process_file(file, x_data_discard, y_data_discard, x_data_pon, y_data_pon, x_data_chii, y_data_chii, x_data_riichi, y_data_riichi)

    rv = [x_data_discard, y_data_discard, x_data_pon, y_data_pon, x_data_chii, y_data_chii, x_data_riichi, y_data_riichi]
    rv = [np.stack(x) if x is not None else None for x in rv]

    return rv


def main():
    x_data, y_data = load_data('test')


if __name__ == '__main__':
    main()

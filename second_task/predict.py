import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from xgboost import XGBRegressor


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

PARAMS = {"subsample": [1],
          "n_estimators": [100],
          "colsample_bytree": [1],
          "max_depth": [3],
          "min_child_weight": [5],
          "learning_rate": [0.01]
          }


def parse_opt() -> argparse.Namespace:
    """Parse script arguments

    Returns
    -------
    argparse.Namespace
        Arguments with their values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalization', action='store_true',
                        help='perform normalization on dataset')
    parser.add_argument('--weights-name', '--name', type=str, default=ROOT/'model.json',
                        help='XGBoost model weights name')
    parser.add_argument('--input', '-i', type=str, default=ROOT/'hidden_test.csv',
                        help='path to the data')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='data delimiter')
    parser.add_argument('--output', '-o', type=str, default=ROOT/'results.csv',
                        help='result path')
    opt = parser.parse_args()

    return opt


def prepare_data(opt: argparse.Namespace) -> np.ndarray:
    """Read and prepare data

    Parameters
    ----------
    opt : argparse.Namespace
        Script arguments

    Returns
    -------
    np.ndarray
        Data for the inference
    """
    data = pd.read_csv(opt.input, delimiter=opt.delimiter).values

    if opt.normalization:
        normalizer = preprocessing.Normalizer()
        data = normalizer.fit_transform(data)

    return data


def main(opt: argparse.Namespace):
    """Load XGBoost Regressor model and run inference on data

    Parameters
    ----------
    opt : argparse.Namespace
        Script arguments with values
    """
    print(opt)
    model = XGBRegressor()
    model.load_model(opt.weights_name)

    data = prepare_data(opt)

    results = model.predict(data)
    pd.DataFrame(results).to_csv(opt.output, sep=opt.delimiter, index=False)


if __name__ == '__main__':
    opt = parse_opt()

    main(opt)

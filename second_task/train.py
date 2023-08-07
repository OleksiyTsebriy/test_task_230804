import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

PARAMS = {
    "subsample": [0.75, 1],
    "n_estimators": [100, 400, 800],
    "colsample_bytree": [0.75, 1],
    "max_depth": [3, 6, 9],
    "min_child_weight": [1, 5],
    "learning_rate": [0.01, 0.1, 0.2]
}


def parse_opt() -> argparse.Namespace:
    """Parse script arguments

    Returns
    -------
    argparse.Namespace
        Arguments with their values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', '--train', type=str, default=ROOT/'train.csv',
                        help='path to training data')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='data delimiter')
    parser.add_argument('--objective', type=str, default='reg:squarederror',
                        help='XGBoost regression objective')
    parser.add_argument('--normalization', action='store_true',
                        help='perform normalization on dataset')
    parser.add_argument('--weights-name', '--name', type=str, default=ROOT/'model.json',
                        help='XGBoost model weights name')
    parser.add_argument('--test-split', '--split', type=float, default=0.2,
                        help='datatest split')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--metric', type=str,
                        default='rmse', help='evaluation metric')

    opt = parser.parse_args()
    print(opt)

    return opt


def prepare_data(opt: argparse.Namespace) -> tuple[np.ndarray]:
    """Read and split data into train and test sets

    Parameters
    ----------
    opt : argparse.Namespace
        Script arguments

    Returns
    -------
    tuple[np.ndarray]
        List containing train-test split of inputs
    """
    data = pd.read_csv(opt.train_path, delimiter=opt.delimiter).values

    X, y = data[:, :-1], data[:, -1]

    return train_test_split(X, y, test_size=opt.test_split, random_state=opt.seed)


def main(opt: argparse.Namespace):
    """Perform XGBoost Regressor train

    Parameters
    ----------
    opt : argparse.Namespace
        Script arguments
    """
    X_train, X_test, y_train, y_test = prepare_data(opt)

    if opt.normalization:
        normalizer = preprocessing.Normalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

    estimator = XGBRegressor(objective=opt.objective,
                             eval_metric=opt.metric, early_stopping_rounds=10)

    search_cv = RandomizedSearchCV(estimator, PARAMS, n_jobs=-1, random_state=opt.seed,
                                   scoring='neg_root_mean_squared_error')
    search_cv = search_cv.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    model = search_cv.best_estimator_

    model.save_model(opt.weights_name)


if __name__ == '__main__':
    opt = parse_opt()

    main(opt)

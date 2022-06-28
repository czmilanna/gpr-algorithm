import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from gpr_algorithm import GPR


def test_type1diabetes():
    random.seed(0)
    df = pd.read_csv(
        Path(__file__).parent.joinpath('data').joinpath('type1diabetes.csv')
    )

    target_names = ['sick', 'healthy']
    feature_names = [
        'age', 'weight', 'height', 'step_count',
        'sedentary', 'light', 'moderate', 'vigorious'
    ]

    labels = df['healthy'].values
    attributes = df[feature_names].values
    attributes_normalized = MinMaxScaler().fit_transform(attributes)

    gpr = GPR(
        target_names=target_names,
        feature_names=feature_names,
        max_n_of_rules=1,
        eval_fun=accuracy_score,
        n_generations=10,
        n_populations=10,
        verbose=False
    )

    gpr.fit(attributes_normalized, labels)
    pred_labels = gpr.predict(attributes_normalized)

    acc = accuracy_score(labels, pred_labels)
    np.testing.assert_almost_equal(acc, 0.808695652173913)

    assert gpr.rules == [
        'IF step_count is High THEN healthy | Support: 0.5288',
        'ELSE sick'
    ]

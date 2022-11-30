import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from gpr_algorithm import GPR


def test_haberman():
    random.seed(0)
    df = pd.read_csv(
        Path(__file__).parent.joinpath('data').joinpath('haberman.csv')
    )

    target_names = ['Died', 'Survived']
    feature_names = [
        'Age', 'Year', 'Positive'
    ]

    labels = df['Survival'].astype("category").cat.codes.values
    attributes = df[feature_names].values
    attributes_normalized = MinMaxScaler().fit_transform(attributes)

    gpr = GPR(
        target_names=target_names,
        feature_names=feature_names,
        max_n_of_rules=5,
        max_n_of_ands=5,
        n_generations=20,
        n_populations=20,
        verbose=False
    )

    gpr.fit(attributes_normalized, labels)
    pred_labels = gpr.predict(attributes_normalized)

    acc = accuracy_score(labels, pred_labels)
    np.testing.assert_almost_equal(acc, 0.761437908496732)

    assert gpr.rules == [
        'IF Age is Medium THEN Survived | Support: 0.2108',
        'IF Positive is High THEN Survived | Support: 0.1434',
        'IF Age is High AND Positive is High THEN Survived | Support: 0.0609',
        'ELSE Died'
    ]

    assert gpr.ranking == [
        'Age: 0.6000', 'Positive: 0.4000'
    ]
    print(gpr.hall_of_fame[0])

    """
    rules equal to chromosome:
    sum(
        mul(a000_is_high, a002_is_high),
        a002_is_high,
        mul(a000_is_high, a000_is_low)
    )
    witch equal to equation:
    age_is_high * age_is_low + positive_is_high + age_is_high * positive_is_high
    """

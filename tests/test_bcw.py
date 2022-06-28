import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from gpr_algorithm import GPR


def test_bwc():
    """
    Missing values '?' in bcw.csv replaced to '5', 16 instances.
    """
    random.seed(0)
    df = pd.read_csv(
        Path(__file__).parent.joinpath('data').joinpath('bcw.csv')
    )

    target_names = ['benign', 'malignant']
    feature_names = [
        'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
    ]

    labels = df['Class'].values
    labels[labels == 2] = 0
    labels[labels == 4] = 1
    attributes = df[feature_names].values
    attributes_normalized = MinMaxScaler().fit_transform(attributes)

    gpr = GPR(
        target_names=target_names,
        feature_names=feature_names,
        max_n_of_rules=3,
        max_n_of_ands=3,
        n_generations=20,
        n_populations=20,
        verbose=False
    )

    gpr.fit(attributes_normalized, labels)
    pred_labels = gpr.predict(attributes_normalized)

    acc = accuracy_score(labels, pred_labels)

    np.testing.assert_almost_equal(acc, 0.9585121602288984)

    assert gpr.rules == [
        'IF Bare Nuclei is High THEN malignant | Support: 0.7340',
        'IF Uniformity of Cell Size is High THEN malignant | Support: 0.6192',
        'ELSE benign'
    ]

    """
    Code:
    gpr = GPR(
        target_names=target_names,
        feature_names=feature_names,
        max_n_of_rules=3,
        max_n_of_ands=3,
        n_generations=1000,
        n_populations=1000,
        verbose=False
    )

    gpr.fit(attributes_normalized, labels)
    pred_labels = gpr.predict(attributes_normalized)
    acc = accuracy_score(labels, pred_labels)
    print(acc)
    for rule in gpr.rules:
        print(rule)
    After 1hr 39 min produces:
    0.9699570815450643
    IF Bare Nuclei is High THEN malignant | Support: 0.7340
    IF Uniformity of Cell Size is High THEN malignant | Support: 0.6192
    IF Clump Thickness is High AND Single Epithelial Cell Size is Low AND Normal Nucleoli is High THEN malignant |
     Support: 0.1765
    ELSE benign
    """

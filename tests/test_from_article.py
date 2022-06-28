import random

import numpy as np
from sklearn import preprocessing

from gpr_algorithm import GPR


def test_from_article():
    """
    Input
    z1          z2           label
   -1.0000     2.5000        1
    2.5000     2.0000        0
    3.0000     7.0000        1
   -0.2000     6.3000        1
    0.5000     5.0000        1

    ---------------------------------------------------------------------------------------------
    Normalized input (sklearn normalizes each attribute individually, users can use other normalization approach)
    y1          y2           label
    0.0000     0.1000        1
    0.8750     0.0000        0
    1.0000     1.0000        1
    0.2000     0.8600        1
    0.3750     0.6000        1

    ---------------------------------------------------------------------------------------------
    Complimented input  ( y1 is low  =  1 - y1 is High)
    y1 is Low   y2 is Low   y1 is High  y2 is High     label
    1.00000     0.90000     0.00000     0.10000        1
    0.12500     1.00000     0.87500     0.00000        0
    0.00000     0.00000     1.00000     1.00000        1
    0.80000     0.14000     0.20000     0.86000        1
    0.62500     0.40000     0.37500     0.60000        1


    ---------------------------------------------------------------------------------------------
    Support for rules (the degree of match to the rule's predecessor, firing degree,
    mean of rule calculation result for only data that has label = 1):

    Rule                       Support
    y2 is High                 (0.10000 + 1.00000 + 0.86000 + 0.60000) / 4       ~=   0.6400
    y1 is Low                  (1.00000 + 0.00000 + 0.80000 + 0.62500) / 4       ~=   0.6062

    Hypothetical rule          Support
    y2 is High and y1 is Low   ( (0.10000 * 1.00000) + (1.00000 * 0.00000) +
                                 (0.86000 * 0.80000) + (0.60000 * 0.62500) ) / 4  ~=  0.2907

    """

    random.seed(1)

    labels = np.array([
        1,
        0,
        1,
        1,
        1
    ])

    attributes = np.array([
        [-1.0, 2.5],
        [2.5, 2.0],
        [3.0, 7.0],
        [-0.2, 6.3],
        [0.5, 5.0]
    ])

    attributes_normalized = preprocessing.MinMaxScaler().fit_transform(attributes)

    cls = GPR(
        feature_names=['y1', 'y2'],
        max_n_of_rules=2, max_n_of_ands=2,
        verbose=False
    )

    cls.fit(attributes_normalized, labels)
    pred_labels = cls.predict(attributes_normalized)

    assert np.all(labels == pred_labels)  # combined rules accuracy = 1.000
    rules = cls.rules
    assert rules == [
        'IF y2 is High THEN 1 | Support: 0.6400',
        'IF y1 is Low THEN 1 | Support: 0.6062',
        'ELSE 0'
    ]

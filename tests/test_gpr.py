#!/usr/bin/env python

"""Tests for `gpr` package."""
import random

import numpy as np
import pytest
from cacp import ClassificationDataset
from cacp.dataset import ClassificationFoldDataNormalizer
from sklearn.metrics import accuracy_score

from gpr_algorithm import GPR


@pytest.fixture
def dataset():
    ds = ClassificationDataset('wdbc')
    return ds


@pytest.fixture
def target_names():
    return ['Malignant', 'Benign']


@pytest.fixture
def feature_names(dataset):
    feature_names = list(dataset._attributes.keys())
    feature_names.remove(dataset.output_name)
    return feature_names


@pytest.fixture
def fold(dataset):
    fold = next(dataset.folds())
    normalizer = ClassificationFoldDataNormalizer()
    fold = normalizer.modify(fold)
    return fold


def test_simple_sick_first():
    random.seed(1)
    feature_names = ['weight', 'height']
    target_names = ['sick', 'healthy']
    cls = GPR(
        feature_names=feature_names,
        target_names=target_names,
        max_n_of_rules=2, max_n_of_ands=2, n_generations=10, n_populations=10,
        verbose=False
    )
    attributes = np.array([
        [.9, .1],  # sick
        [1., .9],  # sick
        [0., .9],
        [.1, .1]
    ])
    labels = np.array([
        0,  # sick
        0,  # sick
        1,
        1
    ])
    cls.fit(attributes, labels)
    pred_labels = cls.predict(attributes)
    assert np.all(labels == pred_labels)
    rules = cls.rules
    assert rules == ['IF weight is Low THEN healthy | Support: 0.9500', 'ELSE sick']


def test_simple_healthy_first():
    random.seed(1)
    feature_names = ['weight', 'height']
    target_names = ['healthy', 'sick']
    cls = GPR(
        feature_names=feature_names,
        target_names=target_names,
        max_n_of_rules=2, max_n_of_ands=2, n_generations=10, n_populations=10,
        verbose=False
    )
    attributes = np.array([
        [.8, .1],  # sick
        [1., .8],  # sick
        [0., .9],
        [.1, .1]
    ])
    labels = np.array([
        1,  # sick
        1,  # sick
        0,
        0
    ])
    cls.fit(attributes, labels)
    pred_labels = cls.predict(attributes)
    assert np.all(labels == pred_labels)
    rules = cls.rules
    assert rules == ['IF weight is High THEN sick | Support: 0.9000', 'ELSE healthy']


def test_happy_path(fold, feature_names, target_names):
    random.seed(1)
    cls = GPR(
        feature_names=feature_names,
        target_names=target_names,
        max_n_of_rules=3, max_n_of_ands=3, n_generations=10, n_populations=10,
        verbose=False
    )
    cls.fit(fold.x_train, fold.y_train)
    pred = cls.predict(fold.x_test)
    acc = accuracy_score(fold.y_test, pred)

    np.testing.assert_almost_equal(acc, 0.9137931034482759)
    assert cls.rules == [
        'IF Radius1 is High THEN Benign | Support: 0.4937',
        'IF Texture3 is High AND Concavity1 is High THEN Benign | Support: 0.1685',
        'ELSE Malignant'
    ]
    assert cls.ranking == [
        'Texture3: 0.3333', 'Concavity1: 0.3333', 'Radius1: 0.3333'
    ]

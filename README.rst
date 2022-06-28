GPR Algorithm
=============

.. image:: https://img.shields.io/pypi/v/gpr-algorithm.svg
        :target: https://pypi.python.org/pypi/gpr-algorithm

.. image:: https://github.com/czmilanna/gpr-algorithm/actions/workflows/tox.yml/badge.svg
        :target: https://github.com/czmilanna/gpr-algorithm/actions/workflows/tox.yml


.. image:: https://readthedocs.org/projects/gpr-algorithm/badge/?version=latest
        :target: https://gpr-algorithm.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

* Free software: MIT license
* Documentation: https://gpr-algorithm.readthedocs.io.

An implementation of an extremely simple classifier (GPR_) that consists of highly interpretable fuzzy metarules
and is suitable for many applications. GPR is effective in accuracy and area under the receiver operating characteristic
(ROC) curve. We provide a Python implementation of the GPR algorithm to enable the use of the algorithm without using
commercial software tools and open access to the research community. We also added enhancements to facilitate the
reading and interpretation of the rules.

.. _GPR: https://doi.org/10.1016/j.ins.2021.05.041

Example usage
--------------

.. code:: python3

    import numpy as np
    from gpr_algorithm import GPR

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
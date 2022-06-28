===============
Example Results
===============

Example 1
--------------

**An example usage of GPR algorithm on the original** article_ **sample data.**

.. _article: https://www.sciencedirect.com/science/article/abs/pii/S0020025521005016?via%3Dihub

.. code-block:: python
   :linenos:

    import random
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from gpr_algorithm import GPR

    random.seed(1)

    labels = np.array(
        [1, 0, 1, 1, 1]
    )
    attributes = np.array(
        [[-1.0, 2.5], [2.5, 2.0], [3.0, 7.0], [-0.2, 6.3], [0.5, 5.0]]
    )
    attributes_normalized = MinMaxScaler().fit_transform(attributes)

    gpr = GPR(
        feature_names=['y1', 'y2'],
        max_n_of_rules=2, max_n_of_ands=2,
        verbose=False
    )
    gpr.fit(attributes_normalized, labels)
    predicted_labels = gpr.predict(attributes_normalized)

    print('Accuracy:')
    print(accuracy_score(labels, predicted_labels))
    print('Rules:')
    for rule in gpr.rules:
        print(rule)

**Terminal output**

.. code-block:: bash

   Accuracy:
   1.0
   Rules:
   IF y2 is High THEN 1 | Support: 0.6400
   IF y1 is Low THEN 1 | Support: 0.6062
   ELSE 0

Example 2
--------------

**An example usage of GPR algorithm on** diabetes_ **dataset.**

.. _diabetes: https://www.mdpi.com/2076-3417/9/12/2555


.. code-block:: python
   :linenos:

    import random
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from gpr_algorithm import GPR

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
        verbose=False
    )
    gpr.fit(attributes_normalized, labels)
    predicted_labels = gpr.predict(attributes_normalized)
    for rule in gpr.rules:
        print(rule)


**Terminal output**

.. code-block:: bash

   IF step_count is High THEN healthy | Support: 0.5288
   ELSE sick

Example 3
--------------

**An example usage of GPR algorithm on** BCW_ **dataset.**

.. _BCW: https://doi.org/10.1073/pnas.87.23.9193


.. code-block:: python
   :linenos:

    import random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from gpr_algorithm import GPR

    random.seed(0)
    df = pd.read_csv(
        Path(__file__).parent.joinpath('data').joinpath('bcw.csv')
    )

    target_names = ['benign', 'malignant']
    feature_names = [
        'Clump Thickness', 'Uniformity of Cell Size',
        'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei',
        'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
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

**Terminal output**

.. code-block:: bash

   IF Bare Nuclei is High THEN malignant | Support: 0.7340
   IF Uniformity of Cell Size is High THEN malignant | Support: 0.6192
   ELSE benign
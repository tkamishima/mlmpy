.. _distfeature:

特徴の分布の学習
================

クラスの分布と同様に， :ref:`nbayes1-fit1-feature` の特徴の分布もブロードキャストの機能を用いて実装します．
特徴ごとの事例数を数え上げる :class:`NaiveBayes1` の実装は次のようなものでした．

.. code-block:: python

   nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
   for i in xrange(n_samples):
       for j in xrange(n_features):
           nXY[j, X[i, j], y[i]] += 1

クラスの分布の場合と同様に，各特徴値ごとに，対象の特徴値の場合にのみカウンタを増やすような実装にします．

.. code-block:: python

    nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
    for i in xrange(n_samples):
        for j in xrange(n_features):
            for yi in xrange(n_classes):
                for xi in xrange(n_fvalues):
                    if y[i] == yi and X[i, j] == xi:
                        nXY[j, xi, yi] += 1

それでは，この実装を，特徴の分布と同様に書き換えます．

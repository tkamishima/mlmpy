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

.. _distfeature-assign:

ループ変数の次元への割り当て
----------------------------

まず，ループ変数は :obj:`i` ， :obj:`j` ， :obj:`yi` ，および :obj:`xj` の四つがあります．
よって，出力配列の次元数は 4 とし，各ループ変数を次元に次のように割り当てます．

.. csv-table::
    :header-rows: 1

    次元, ループ変数, 大きさ, 意味
    0, :obj:`i` , :obj:`n_samples` , 事例
    1, :obj:`j` , :obj:`n_features` , 特徴
    2, :obj:`xi` , :obj:`n_fvalues` , 特徴値
    3, :obj:`yi` , :obj:`n_classes` , クラス

この割り当てで考慮すべきは，最終結果を格納する :obj:`nXY` です．
この変数 :obj:`nXY` の第0次元は特徴，第1次元は特徴値，そして第3次元はクラスなので，この順序は同じになるように割り当てています．
最後に凝集演算をしたあとに，次元の入れ替えも可能ですが，入れ替えが不要で，実装が簡潔になるように予め割り当てておきます．


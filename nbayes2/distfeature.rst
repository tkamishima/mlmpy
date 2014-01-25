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

.. _distfeature-arygen:

計算に必要な配列の生成
----------------------

ループ内での要素ごとの演算は ``y[i] == yi and X[i, j] == xi`` です．
ループ変数 :obj:`yi` と :obj:`xi` に対応する配列は次のようになります．

.. code-block:: python

    ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]

``y[i]`` に対応する配列は，全事例の :obj:`y` の値を，事例に対応する第0次元に割り当て，その他の次元の大きさを1にした配列となります．

.. code-block:: python

    ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]

``X[i, j]`` に対応する配列は，全事例の ``X[:, j]`` の値を，事例に対応する第0次元に，そして全特徴の ``X[i, :]`` の値を，特徴に対応する第1次元に割り当て，その他の第2と第3次元の大きさを1にした配列となります．

.. code-block:: python

    ary_X = X[:, :, np.newaxis, np.newaxis]

以上で演算に必要な値を得ることができました．

.. _distfeature-computation:

要素ごとの演算と凝集演算
------------------------

まず要素ごとの比較演算 ``y[i] == yi and X[i, j] == xi`` を配列間で実行します．
``y[i] == yi`` と ``X[i, j] == xi`` に対応する計算は， :obj:`==` がユニバーサル関数なので，次のように簡潔に実装できます．

.. code-block:: python

    cmp_X = (ary_X == ary_xi)
    cmp_y = (ary_y == ary_yi)

ここで， :obj:`and` は Python の組み込み関数で，ユニバーサル関数ではありません．
そこで，ユニバーサル関数である :func:`np.logical_and` を用います [1]_ ．

.. index:: logical_and

.. function::  np.logical_and(x1, x2[, out]) = <ufunc 'logical_and'>

    Compute the truth value of x1 AND x2 elementwise.

実装結果は次のようになります．

.. code-block:: python

    cmp_Xandy = np.logical_and(cmp_X, cmp_y)

つぎに，全ての事例についての総和を求める凝集演算を行います．
総和を求める :func:`np.sum` を，事例に対応する第0次元に適用します [2]_ ．

.. code-block:: python

    nXY = np.sum(cmp_Xandy, axis=0)

以上の配列の生成と，演算を全てをまとめると次のようになります．

.. code-block:: python

    ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]
    ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]
    ary_X = X[:, :, np.newaxis, np.newaxis]

    cmp_X = (ary_X == ary_xi)
    cmp_y = (ary_y == ary_yi)
    cmp_Xandy = np.logical_and(cmp_X, cmp_y)

    nXY = np.sum(cmp_Xandy, axis=0)

そして，中間変数への代入を適宜整理します．

.. code-block:: python

    cmp_X = (X[:, :, np.newaxis, np.newaxis] ==
             np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis])
    cmp_y = (y[:, np.newaxis, np.newaxis, np.newaxis] ==
             np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :])
    nXY = np.sum(np.logical_and(lX, ly), axis=0)

以上で，各特徴，各特徴値，そして各クラスごとの事例数を数え上げることができました．

.. [1]
    同様の関数に， :obj:`or` ， :obj:`not` ，および :obj:`xor` の論理演算に，それぞれ対応するユニバーサル関数 :func:`logical_or` ，:func:`logical_not` ，および :func:`logical_xor` があります．

.. [2]
    もし同時に二つ以上の次元について同時に凝集演算をする必要がある場合には， :func:`np.apply_over_axes` を用います．

    .. index:: apply_over_axes

    .. function::  numpy.apply_over_axes(func, a, axes)

        Apply a function repeatedly over multiple axes.

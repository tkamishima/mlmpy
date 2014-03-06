.. _nbayes2-distfeature:

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

.. _nbayes2-distfeature-assign:

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
この変数 :obj:`nXY` の第0次元は特徴，第1次元は特徴値，そして第3次元はクラスなので，この順序は同じになるように割り当てています [#]_ ．
最後に凝集演算をしたあとに，次元の入れ替えも可能ですが，入れ替えが不要で，実装が簡潔になるように予め割り当てておきます．

.. [#]
    もしも軸の順序を揃えることができない場合は， :func:`np.swapaxes` 関数を用いて軸の順序を入れ換えます．

    .. index:: swapaxes

    .. function:: np.swapaxes(a, axis1, axis2)

        Interchange two axes of an array.

.. _nbayes2-distfeature-arygen:

計算に必要な配列の生成
----------------------

ループ内での要素ごとの演算は ``y[i] == yi and X[i, j] == xi`` です．
よって，必要な配列は ``y[i]`` ， :obj:`yi` ， ``X[i, j]`` ，および :obj:`xi` となります．

ループ変数 :obj:`yi` と :obj:`xi` に対応する配列は次のようになります．

.. code-block:: python

    ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]

``y[i]`` は， :ref:`nbayes2-distclass` の場合とは，次元数とループの次元への割り当てが異なるだけです．
ループ変数 :obj:`i` は第0次元に対応するので，これに対応する変数は次のとおりです．

.. code-block:: python

    ary_i = np.arange(n_samples)[:, np.newaxis, np.newaxis, np.newaxis]

すると， ``y[i]`` に対応する配列は次のようになります．

.. code-block:: python

    ary_y = y[ary_i]

これは， :ref:`nbayes2-distclass` の場合と同様に次のように簡潔に実装できます．

.. code-block:: python

    ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]

この実装では，全事例の :obj:`y` の値を，事例に対応する第0次元に割り当て，その他の次元の大きさを 1 である配列を求めています．

``X[i, j]`` はループ変数を2個含んでいるので，これまでとは状況が異なります．
``X[ary_ij]`` のような形式で，2個以上のインデックスを含み，かつ :const:`np.newaxis` による次元の追加が可能な :obj:`ary_ij` の作成方法を著者は知りません [#]_ ．
そこで，ループ変数の値に対応した配列を考えず， :obj:`X` の要素を，ループを割り当てた次元に対応するように配置した配列を直接的に生成します．
これは，全事例の ``X[:, j]`` の値を，事例に対応する第0次元に，そして全特徴の ``X[i, :]`` の値を，特徴に対応する第1次元に割り当て，その他の第2と第3次元の大きさを1にした配列となります．
すなわち，ループ変数 :obj:`xi` と :obj:`yi` に対応する次元を :obj:`X` に追加します．

.. code-block:: python

    ary_X = X[:, :, np.newaxis, np.newaxis]

以上で演算に必要な値を得ることができました．

.. [#]
    もし :const:`np.newaxis` による次元の追加が不要であれば， :func:`np.ix_` を用いて， ``ary_ij = np.ix_(np.arange(n_samples), np.arange(n_features))`` のような記述が可能です．

    .. index:: ix_

    .. function:: np.ix_(*args)[source]¶

        Construct an open mesh from multiple sequences.

.. _nbayes2-distfeature-computation:

要素ごとの演算と凝集演算
------------------------

``y[i] == yi and X[i, j] == xi`` の式のうち，比較演算を実行します．
``y[i] == yi`` と ``X[i, j] == xi`` に対応する計算は， :obj:`==` がユニバーサル関数なので，次のように簡潔に実装できます．

.. code-block:: python

    cmp_X = (ary_X == ary_xi)
    cmp_y = (ary_y == ary_yi)

次にこれらの比較結果の論理積を求めますが， :obj:`and` は Python の組み込み関数で，ユニバーサル関数ではありません．
そこで，ユニバーサル関数である :func:`np.logical_and` を用います [#]_ ．

.. index:: logical_and

.. function::  np.logical_and(x1, x2[, out]) = <ufunc 'logical_and'>

    Compute the truth value of x1 AND x2 elementwise.

実装は次のようになります．

.. code-block:: python

    cmp_Xandy = np.logical_and(cmp_X, cmp_y)

最後に，全ての事例についての総和を求める凝集演算を行います．
総和を求める :func:`np.sum` を，事例に対応する第0次元に適用します [#]_ ．

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

そして，中間変数への代入を整理します．

.. code-block:: python

    ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]
    ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]
    ary_X = X[:, :, np.newaxis, np.newaxis]

    nXY = np.sum(np.logical_and(ary_X == ary_xi, ary_y == ary_yi), axis=0)

以上で，各特徴，各特徴値，そして各クラスごとの事例数を数え上げることができました．

.. [#]
    同様の関数に， :obj:`or` ， :obj:`not` ，および :obj:`xor` の論理演算に，それぞれ対応するユニバーサル関数 :func:`logical_or` ，:func:`logical_not` ，および :func:`logical_xor` があります．

.. [#]
    もし同時に二つ以上の次元について同時に集約演算をする必要がある場合には， :func:`np.apply_over_axes` を用います．

    .. index:: apply_over_axes

    .. function::  np.apply_over_axes(func, a, axes)

        Apply a function repeatedly over multiple axes.

.. _nbayes2-distfeature-prob:

特徴値の確率の計算
------------------

最後に :obj:`nXY` と，クラスごとの事例数 :obj:`nY` を用いて，クラスが与えられたときの，各特徴値が生じる確率を計算します．
それには :obj:`nXY` を，対応するクラスごとにクラスごとの総事例数 :obj:`nY` で割ります．
:obj:`nY` を :obj:`nXY` と同じ次元数にし，そのクラスに対応する第2次元に割り当てるようにすると ``nY[np.newaxis, np.newaxis, :]`` となります．
あとは，実数の結果を返す割り算のユニバーサル関数 :func:`np.true_divide` を適用すれば，特徴値の確率を計算できます．

.. code-block:: python

    self.pXgY_ = np.true_divide(nXY, nY[np.newaxis, np.newaxis, :])

.. _nbayes2-distfeature-run:

実行
----

.. index:: sample; nbayes2.py, sample; run_nbayes2.py, class; NaiveBayes2

以上の，ブロードキャスト機能を活用した訓練メソッド :meth:`fit` を実装した :class:`NaiveBayes2` と，その実行スクリプトは，以下より取得できます．
この :class:`NaiveBayes2` クラスの実行可能な状態のファイルは

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/nbayes2.py

.. only:: html and not epub

  :download:`NaiveBayes2 クラス：nbayes2.py <../source/nbayes2.py>`

であり，実行ファイルは

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/run_nbayes2.py

.. only:: html and not epub

  :download:`NaiveBayes2 実行スクリプト：run_nbayes2.py <../source/run_nbayes2.py>`

です．
実行すると， :class:`NaiveBayes1` と :class:`NaiveBayes2` で同じ結果が得られます．

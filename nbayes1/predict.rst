.. _nbayes1-predict:

予測メソッドの実装
==================

学習したモデルパラメータを使って，未知の事例のクラスを予測する :meth:`predict` メソッドを，できるだけ NumPy 配列の利点を活用生かす方針で実装します．

このメソッドは，未知の特徴ベクトルをいくつか集めた配列 :obj:`X` を引数とします．
そして， :obj:`X` 中の各特徴ベクトルに対する予測ベクトルをまとめた配列 :obj:`y` を返します．
最初に， :meth:`fit` メソッドと同様に， :obj:`n_samples` や :obj:`n_features` などの定数を設定します．

.. _nbayes1-predict-unseenvec:

未知ベクトルの抽出
------------------

.. index:: ndarray, slice

次に， :obj:`X` から未知ベクトルを一つずつ抽出します．
:ref:`nbayes1-ndarray-access` では，配列の要素を一つずつ参照する方法を紹介しました．
これに加え，NumPy 配列は，リストや文字列などのスライスと同様の方法により，配列の一部分をまとめて参照することもできます．

1次元配列の場合は，リストのスライス表記と同様の ``開始:終了:増分`` の形式を用います．

.. code-block:: ipython

   In [10]: x = np.array([0, 1, 2, 3, 4])
   In [11]: x[1:3]
   Out[11]: array([1, 2])
   In [12]: x[0:5:2]
   Out[12]: array([0, 2, 4])
   In [13]: x[::-1]
   Out[13]: array([4, 3, 2, 1, 0])
   In [14]: x[-3:-1]
   Out[14]: array([2, 3])

NumPy 配列やリストを使って複数の要素を指定し，それらをまとめた配列を作ることもできます．
これは，配列 :obj:`x` からリストと NumPy 配列を使って選んだ要素を並べた配列を作る例です．

.. code-block:: ipython

   In [20]: x = np.array([10, 20, 30, 40, 50])
   In [21]: x[[0, 2, 1, 2]]
   Out[21]: array([10, 30, 20, 30])
   In [22]: x[np.array([3, 3, 1, 1, 0, 0])]
   Out[22]: array([40, 40, 20, 20, 10, 10])

2次元以上の配列でも同様の操作が可能です．
特に， ``:`` のみを使って，行や列全体を取り出す操作はよく使われます．

.. code-block:: ipython

   In [30]: x = np.array([[11, 12, 13], [21, 22, 23]])
   In [31]: x
   Out[31]:
   array([[11, 12, 13],
          [21, 22, 23]])
   In [32]: x[0, :]
   Out[32]: array([11, 12, 13])
   In [33]: x[:, 1]
   Out[33]: array([12, 22])
   In [34]: x[:, 1:3]
   Out[34]:
   array([[12, 13],
          [22, 23]])

それでは，配列 :obj:`X` から一つずつ行を取り出してみます．
そのために :obj:`for` ループで :obj:`i` 行目を順に取り出します．

.. code-block:: python

   for i in xrange(n_samples):
       xi = X[i, :]

:class:`np.ndarray` は，最初の次元を順に走査するイテレータの機能も備えています．
具体的には，1次元配列なら要素を順に返し，2次元配列なら行列の行を順に返し，3次元配列なら2次元目と3次元目で構成される配列を順に返します．
次の例では，行のインデックスを変数 :obj:`i` に，行の内容を変数 :obj:`xi` に同時に得ることができます．

.. code-block:: python

   for i, xi in enumerate(X):
       pass

.. _nbayes1-predict-jointprob:

対数同時確率の計算
------------------

方針（１）
^^^^^^^^^^

.. index:: universal function, log

次に，この未知データ :obj:`xi` のクラスラベルを， :ref:`nbayes1-nbayes` の式(6)を用いて予測します．
すなわち， :obj:`xi` に対し， :math:`y` が 0 と 1 それぞれの場合の対数同時確率を計算し，その値が大きな方を予測クラスラベルとします．

まず :math:`y` が 0 と 1 の場合を個別に計算するのではなく， NumPy の利点の一つであるユニバーサル関数を用いてまとめて計算する方針で実装します．
ユニバーサル関数は，入力した配列の各要素に関数を適用し，その結果を入力と同じ形の配列にします．
式(6)の最初の項 :math:`\log\Pr[y]` は，クラスの事前分布のパラメータ :obj:`self.pY_` に対数関数を適用して計算します．
このとき，対数関数として :mod:`math` パッケージの対数関数 :func:`math.log` ではなく，ユニバーサル関数の機能をもつ NumPy の対数関数 :func:`np.log` [1]_ を用います．

.. code-block:: python

   logpXY = np.log(self.pY_)

式(6)の第2項の総和の中 :math:`\log\Pr[x_j^\mathrm{new} | y]` の計算に移ります．
計算に必要な確率関数は，モデルパラメータ :obj:`self.pXgY` の :obj:`j` 番目の要素で，もう一方の未知特徴ベクトルの値は， :obj:`xi` の :obj:`j` 番目の要素で得られます．
最後の :math:`y` については， ``:`` を使うことで 0 と 1 両方の値を同時に得ます．
これを全ての特徴 :obj:`j` について求め，それらを :obj:`logpXY` に加えます．

.. code-block:: python

    for j in xrange(n_features):
        logpXY = logpXY + np.log(self.pXgY_[j, xi[j], :])

:func:`np.log` と同様に， ``+`` や ``*`` などの四則演算もユニバーサル関数としての機能を持っています．
同じ大きさの配列 :obj:`a` と :obj:`b` があるとき， ``a + b`` は要素ごとの和をとり，入力と同じ大きさの配列を返します．
``*`` については，内積や行列積ではなく，要素ごとの積が計算されることに注意して下さい．

.. code-block:: ipython

    In [40]: a = np.array([1, 2])
    In [41]: b = np.array([3, 4])
    In [42]: a + b
    Out[42]: array([4, 6])
    In [43]: a * b
    Out[43]: array([3, 8])

方針（２）
^^^^^^^^^^

以上のような :obj:`for` ループを用いた実装をさらに改良し，NumPy の機能をさらに生かした実装を紹介します．
具体的には，(1) NumPy 配列 :obj:`self.pXgY_` の要素を，一つずつではなくまとめて取り出して (2) それらの総和を計算します．

まず(1)には，NumPy 配列やリストを使って複数の要素を指定し，それらをまとめた配列を作る機能を利用します．
:obj:`for` 文によって :obj:`j` を変化させたとき ``self.pXgY_[j, xi[j], :]`` の1番目の添え字は ``0`` から ``n_features - 1`` の範囲で変化します．
2番目の引数は， :obj:`xi` の要素を最初から最後まで並べたもの，すなわち :obj:`xi` そのものになります．
以上のことから， :obj:`self.pXgY_` の要素をまとめて取り出すとき，2番目の添え字には :obj:`xi` を与え，3番目の引数は ``:`` でこの軸の全要素を指定できるので，あとは1番目の添え字が指定できれば目的を達成できます．
1番目の添え字は 0 から ``n_features - 1`` の整数を順にならべたものです．
このような，等差級数の数列を表す配列は :func:`np.arange` 関数で生成できます．

.. index:: arange

.. function:: np.arange([start], stop[, step], dtype=None)

   Return evenly spaced values within a given interval.

使い方はビルトインの :func:`range` 関数と同様で，開始，終了，増分を指定します．
ただし，リストではなく1次元の配列を返すことや，配列の :attr:`dtype` 属性を指定できる点が異なります．
NumPy 配列の添え字として与える場合には :attr:`dtype` 属性は整数でなくてはなりません．
ここでは， ``np.arange(n_features)`` と記述すると，引数が整数ですので，規定値で整数型の配列がちょうど得られます．
以上のことから ``self.pXgY_[np.arange(n_features), xi, :]`` によって，各行が， :obj:`j` を 0 から ``n_features - 1`` まで変化させたときの， ``self.pXgY_[j, xi[j], :]`` の結果になっている配列が得られます．
なおこの配列の :attr:`shape` 属性は ``(n_features, n_classes)`` となっています．

この配列の各要素ごとに対数をとり， :obj:`j` が変化する方向，すなわち列方向の和をとれば目的のベクトルが得られます．
まず， :func:`np.log` を適用すれば，ユニバーサル関数の機能によって，配列の全要素について対数をとることができます．

列方向の和をとるには :func:`np.sum` 関数を利用します．

.. index:: sum

.. function:: np.sum(a, axis=None, dtype=None)

   Sum of array elements over a given axis.

引数 ``a`` で指定された配列の，全要素の総和を計算します．
ただし， ``axis`` を指定すると，配列の指定された次元方向の和を計算します．
``dtype`` は，返り値配列の :attr:`dtype` 属性です．

``axis`` 引数について補足します．
``axis`` は， 0 から :attr:`ndim` で得られる次元数より 1 少ない値で指定します．
行列に相当する2次元配列では， ``axis=0`` は列和に， ``axis=1`` は行和になります．
計算結果の配列は，指定した次元は和をとることで消えて次元数が一つ減ります．
指定した次元以外の ``shape`` 属性はそのまま保存されます．

対数同時確率は，これまでの手順をまとめた次のコードで計算できます．

.. code-block:: python

    logpXY = np.log(self.pY_) + \
             np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :]),
                    axis=0)

.. only:: not latex

   .. rubric:: 注釈

.. [1]
   :func:`np.log` や :func:`np.sin` などの NumPy の初等関数は， :mod:`math` のものと比べて，ユニバーサル関数であることの他に， :func:`np.seterr` でエラー処理の方法を変更できたり，複素数を扱えるといった違いもあります．

.. _nbayes1-predict-select:

予測クラスの決定
----------------

以上で， :math:`y` が 0 と 1 に対応する値を含む配列 :obj:`logpXY` が計算できました．
このように計算した :obj:`logpXY` のうち最も大きな値をとる要素が予測クラスになります．
これには，配列中で最大値をとる要素の添え字を返す関数 :func:`np.argmax` を用います [2]_ ．

.. index:: argmax, argmin

.. function:: np.argmax(a, axis=None)

   Indices of the maximum values along an axis.

逆に最小値をとる要素の添え字を返すのは :func:`np.argmin` です．

.. function:: np.argmin(a, axis=None)

   Return the indices of the minimum values along an axis.

予測クラスを得るコードは次のとおりです．

.. code-block:: python

   y[i] = np.argmax(logpXY)

この例では，予め確保しておいた領域 :obj:`y` に予測クラスを順に格納しています．

.. index:: sample; nbayes1.py

以上で， :class:`NaiveBayes1` クラスの実装は完了しました．
実行可能な状態のファイルは，以下より取得できます．

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/release/source/nbayes1.py

.. only:: html and not epub

  :download:`NaiveBayes1 クラス：nbayes1.py <../source/nbayes1.py>`

.. only:: not latex

   .. rubric:: 注釈

.. [2]
   NumPy 配列のメソッド :meth:`np.ndarray.argmax` を使う方法もあります．

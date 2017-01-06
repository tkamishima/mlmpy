.. _nbayes1-fit1:

学習メソッドの実装（１）
========================

モデルパラメータを，訓練データから学習する :meth:`fit` メソッドを，単純に多次元配列として，NumPy 配列を利用する方針で実装します．
実は，この実装方針では NumPy の利点は生かせませんが，後の :ref:`nbayes2` 章で，NumPy のいろいろな利点を順に紹介しながら，この実装を改良してゆきます．

.. _nbayes1-fit1-const:

定数の設定
----------

まず，メソッド内で利用する定数を定義します．
このメソッドの引数は，訓練データの特徴ベクトル集合 :obj:`X` とクラスラベル集合 :obj:`y` であると :ref:`nabyes1-spec-class` で定義しました．
最初に，この引数から特徴数や訓練事例数などの定数を抽出します．
:obj:`X` は，行数が訓練事例数に，列数が特徴数に等しい行列に対応した2次元配列です．
そこでこの変数の :attr:`shape` 属性のタプルから訓練事例数と特徴数を得ます．

.. code-block:: python

   n_samples = X.shape[0]
   n_features = X.shape[1]

実装する単純ベイズは，クラスも特徴も全て二値としましたが，このことを定義する定数も定義しておきます．

.. code-block:: python

   n_classes = 2
   n_fvalues = 2

特徴の事例数とクラスラベルの事例数は一致していなくてはならないので，そうでない場合は :exc:`ValueError` を送出するようにします．
:obj:`y` の :attr:`shape` 属性を調べてもよいのですが，これは1次元配列なので長さを得る関数 :func:`len` [#]_ を用いて実装してみます．

.. code-block:: python

   if n_samples != len(y):
       raise ValueError('Mismatched number of samples.')

以上で，モデルパラメータを学習する準備ができました．

.. only:: not latex

   .. rubric:: 注釈

.. [#]
   2次元以上の NumPy 配列に :func:`len` を適用すると :attr:`shape` 属性の最初の要素を返します．

.. _nbayes1-fit1-class:

クラスの分布の学習
------------------

:ref:`nbayes1-nbayes` の式(4)のクラスの分布のパラメータを求めます．
計算に必要な量は総事例数 :math:`N` とクラスラベルが :math:`y` である事例数 :math:`N[y_i=y]` です．
:math:`N` はすでに :obj:`n_samples` として計算済みです．
:math:`N[y_i=y]` は， :math:`y\in\{0,1\}` について計算する必要があります．
よって，大きさ :obj:`n_classes` の大きさのベクトル :obj:`nY` を作成し，各クラスごとに事例を数え上げます．

.. code-block:: python

   nY = np.zeros(n_classes, dtype=int)
   for i in range(n_samples):
       nY[y[i]] += 1

モデルパラメータ :obj:`self.pY_` は式(4)に従って計算します．
なお，後で値を書き換えるので :func:`np.empty` で初期化したあと，各クラスの確率を計算して格納します [#]_ ．

.. code-block:: python

   self.pY_ = np.empty(n_classes, dtype=float)
   for i in range(n_classes):
       self.pY_[i] = nY[i] / n_samples

.. only:: not latex

   .. rubric:: 注釈

.. [#]
    Python3 では，整数同士の除算の結果も実数ですが，Python2 では切り捨てした整数となります．
    これを避けるために，Python2 では :obj:`n_samples` を ``float(n_samples)`` のように実数型に変換しておく必要があります．

.. _nbayes1-fit1-feature:

特徴の分布の学習
----------------

:ref:`nbayes1-nbayes` の式(5)の特徴の分布のパラメータを求めます．
計算に必要な量のうち :math:`N[y_i=y]` は，すでに式(4)の計算で求めました．
もう一つの量 :math:`N[x_{ij}=x_j, y_i=y]` は，特徴 :math:`j=1,\ldots,K` それぞれについて，特徴の値 :math:`x_j\in\{0,1\}` とクラス :math:`y\in\{0,1\}` について計算する必要があります．
よって，この量を保持する配列は3次元で，その :attr:`shape` 属性は ``(n_features, n_fvalues, n_classes)`` とする必要があります．
この大きさの 0 行列を確保し，各特徴それぞれについて，各特徴値と各クラスごとに事例を数え上げます．

.. code-block:: python

   nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=int)
   for i in range(n_samples):
       for j in range(n_features):
           nXY[j, X[i, j], y[i]] += 1

モデルパラメータ :obj:`self.pXgY_` は式(5)に従って計算します．

.. code-block:: python

   self.pXgY_ = np.empty((n_features, n_fvalues, n_classes),
                         dtype=float)
   for j in range(n_features):
       for xi in range(n_fvalues):
           for yi in range(n_classes):
               self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / float(nY[yi])

以上で，単純ベイズのモデルパラメータの学習を完了しました．

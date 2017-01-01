.. _lr-fit:

学習メソッドの実装
==================


それでは，前節の :ref:`lr-optimization` で紹介した :func:`minimize` を用いて，学習メソッド :meth:`fit` を実装します．
:func:`minimize` とパラメータをやりとりするために，構造化配列を用いる方法についても紹介します．

.. _lr-fit-fit:

学習メソッド
------------

あてはめを行う :meth:`fit` メソッドでは，まずデータ数と特徴数を設定しておきます．

.. code-block:: python

    def fit(self, X, y):
        """
        Fitting model
        """

        # constants
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]

そして，最適化関数 :func:`mimimze` で最適なパラメータを求めます．

.. index:: minimize

.. code-block:: python

    # optimize
    res = minimize(fun=self.loss,
                   x0=np.zeros(self.n_features_ + 1, dtype=float),
                   jac=self.grad_loss,
                   args=(X, y),
                   method='CG')

:func:`minimize` を呼び出して，ロジスティック回帰モデルをあてはめてて，その結果を:class:`OptimizeResult` のインスタンスとして受け取り， :obj`res` に保持しています．

最適化手法には ``method`` で ``CG`` ，すなわち共役勾配降下法を指定しました．
:func:`minimize` の引数 ``fun`` と ``jac`` には，それぞれロジスティック回帰の目的関数とその勾配ベクトル，すなわち :ref:`lr-lr` の式(2)と式(4)を計算するメソッドを与えています．
これらのメソッドについては次節の :ref:`lr-loss` で詳しく述べます．

最適解を探索する初期パラメータ ``x0`` には :func:`np.zeros` で生成した実数の0ベクトルを与えています．
目的関数のパラメータ配列の大きさは，この初期パラメータの大きさになります．
ここでは，重みベクトル :math:`\mathbf{w}` の次元数，すなわち特徴数に，切片パラメータ (intercept)  :math:`b` のための ``1`` を加えた数にしています．

目的関数と勾配ベクトルを計算するにはモデルのパラメータの他にも訓練データの情報が必要です．
そこで，これらの情報を ``args`` に指定して，目的関数・勾配ベクトルを計算するメソッドに引き渡されるようにしています．

最適化が終わったら， :obj:`res` の属性 :attr:`x` に格納されているパラメータを取り出します．

.. code-block:: python

    # get result
    self.coef_ = res.x.view(self._param_dtype)['coef'][0, :].copy()
    self.intercept_ = res.x.view(self._param_dtype)['intercept'][0]

このロジスティック回帰のクラスでは，重みベクトル :math:`\mathbf{w}` と切片 :math:`b` のパラメータを，それぞれ属性 :attr:`coef_` と :attr:`intercept_` に保持します．
しかし， これらのパラメータはまとめて1次元配列 :obj:`res.x` に格納されています．
そこで，このあとすぐ紹介する :meth:`view` と構造化配列を使って分離する必要があります．
なお，ローカル変数である :obj:`res` は :meth:`fit` メソッドの終了時にその内容が失われるので， :meth:`copy` メソッドで配列の実体をコピーしていることに注意して下さい．

.. _lr-fit-sarray:

.. index:: ! structured array, ndarray; dtype

構造化配列
----------

まとめて1次元の配列に格納されているパラメータを分離するために，ここでは構造化配列を利用します．
そこで，まずこの構造化配列について紹介します．

構造化配列 (structured array) とは，通常のNumPy配列と次のような違いがあります．

* 通常のNumPy配列では要素が全て同じ型でなければならないのに対し，構造化配列では列ごとに型を変更可能
* 文字列による名前で列を参照可能
* 配列の要素として配列を指定可能

.. index:: dtype

構造化配列は今まで紹介した :class:`ndarray` とは， :attr:`dtype` 属性の値が異なります．
構造化配列では，列ごとにその要素が異なるので，各列の定義をリストとして並べます．

    ``[(field_name, field_dtype, field_shape), ...]``

``field_name`` は列を参照するときの名前で，辞書型のキーとして利用できる文字列を指定します．
``field_dtype`` はこの列の型で， :ref:`nbayes1-ndarray-access` で紹介したNumPyの型を表すクラス :class:`np.dtype` を指定します．
``field_shape`` は省略可能で，省略したり，単に ``1`` と指定すると通常の配列と同じ0次元配列，すなわちスカラーになります [#]_ ．
2以上の整数を指定すると，指定した大きさの1次元配列が要素に，整数のタプルを指定すると， :attr:`shape` がそのタプルの値である :class:`ndarray` が要素になります [#]_ ．

それでは，実際に構造化配列を生成してみます．

.. code-block:: ipython

    In [1]: a = np.array(
       ...:     [('red', 0.2, (255, 0, 0)),
       ...:     ('yellow', 0.5, (255, 255, 0)),
       ...:     ('green', 0.8, (0, 255, 0))],
       ...:     dtype=[('label', 'U10'), ('state', float), ('color', int, 3) ])

.. todo: Python3 では文字列の u が消える

:func:`np.array` を用いて構造化配列を生成しています．
最初の引数は配列の内容で，各行の内容を記述したタプルのリストで表します．
配列の型を :attr:`dtype` 属性で指定しています．
最初の列は名前が ``label`` で，その型は長さ10のUnicode文字列です．
次の列 ``state`` はスカラーの実数，そして最後の列 ``color`` は大きさ3の1次元の整数型配列です．

次は，生成した構造化配列の内容を参照します．
型を指定した時の列の名前 ``field_name`` の文字列を使って，構造化配列 :obj:`a` の列は ``a[field_name]`` の記述で参照できます．
それでは，上記の構造化配列 :obj:`a` の要素を参照してみます．

.. code-block:: ipython

    In [2]: a['label']
    Out[2]:
    array([u'red', u'yellow', u'green'],
          dtype='<U10')
    In [3]: a['color']
    Out[3]:
    array([[255,   0,   0],
           [255, 255,   0],
           [  0, 255,   0]])
    In [4]: a['state'][1]
    Out[4]: 0.5

最初の ``a['label']`` は，名前が ``label`` の列，すなわち第1列を参照します．
要素がUnicode文字列である1次元配列が得られています．
2番目の ``a['color']`` は最後の列 ``color`` を参照しています．
各行の要素が大きさ3の整数配列なので，それらを縦に連結した ``(3, 3)`` の配列が得られます．
最後の ``a['state'][1]`` は， ``a['state']`` で :obj:`a` の第2列 ``state`` で1次元の実数配列が得られ， ``[1]`` によってインデックスが 1 の要素，すなわち2番目の要素が抽出されます．

.. only:: not latex

   .. rubric:: 注釈

.. [#]

    ``1`` ではなく， ``(1,)`` と指定すると，スカラーではなく，1次元の大きさ1の配列になります．

.. [#]

    その他，構造化配列の :attr:`dtype` を指定する方法は他にも用意されています．
    詳細はNumPyマニュアルの `Structured Array <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ の項目を参照して下さい．

.. _lr-fit-implementation:

構造化配列を用いた実装
----------------------

それでは，この構造化配列を使って，ロジスティック回帰のパラメータを表してみます．
:meth:`fit` メソッドで， 最適化を実行する前に，次のように実装しました．

.. code-block:: python

    # dtype for model parameters to optimize
    self._param_dtype = np.dtype([
        ('coef', float, self.n_features_),
        ('intercept', float)
    ])

第1列目の ``coef`` は重みベクトル :math:`\mathbf{w}` を表すものです．
1次元で大きさが特徴数 :attr:`n_features_` に等しい実数ベクトルとして定義しています．
第2列目の ``intercept`` は切片 :math:`b` に相当し，スカラーの実数値としています．
この構造化配列の型を :class:`np.dtype` クラスのインスタンスとしてロジスティック回帰クラスの属性 :attr:`_param_dtype` 保持しておきます．

.. class:: np.dtype

    Create a data type object.

    :ivar obj: Object to be converted to a data type object.

それでは， :func:`minimize` の結果を格納した :obj:`res.x` から，構造化配列を使ってパラメータを分離する次のコードをもう一度見てみます．

.. index:: ndarray ; view

.. code-block:: python

    # get result
    self.coef_ = res.x.view(self._param_dtype)['coef'][0, :].copy()
    self.intercept_ = res.x.view(self._param_dtype)['intercept'][0]

:meth:`view` は，配列自体は変更や複製をすることなく，異なる型の配列として参照するメソッドです．
C言語などの共用体と同様の動作をします．
:obj:`res.x` は大きさが ``n_features_ + 1`` の実数配列ですが，重みベクトルと切片のパラメータをまとめた :attr:`_param_dtype` 型の構造化配列として参照できます．

:attr:`_param_dtype` 型では，列 ``coef`` は大きさが ``n_features_`` の1次元配列です．
よって， ``res.x.view(self._param_dtype)['coef']`` によって :attr:`shape` が ``(1, n_features_)`` の配列を得ることができます．
その後の ``[0, :]`` によって，この配列の1行目の内容を参照し，これを重みベクトルとして取り出しています．
もう一方の列 ``intercept`` はスカラーの実数なので， ``res.x.view(self._param_dtype)['intercept']`` によって大きさが1の1次元実数配列を参照できます．
この配列の最初の要素を参照し，これを切片として取り出しています．

以上で， :ref:`lr-lr` の式(3)を解いて，得られた重みベクトル :math:`mathbf{w}` と切片 :math:`b` を，ロジスティック回帰の属性 :attr:`coef_` と :attr:`intercept_` とにそれぞれ格納することができました．

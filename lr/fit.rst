.. _lr-fit:

学習メソッドの実装
==================

.. _lr-fit-fit:

学習メソッド
------------

それではロジスティック回帰でのあてはめを :func:`minimize` を用いて実装します．
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

.. code-block:: python

    # optimize
    res = minimize(fun=self.loss,
                   x0=np.zeros(self.n_features_ + 1, dtype=float),
                   jac=self.grad_loss,
                   args=(X, y),
                   method='CG')

:func:`minimize` を呼び出して，ロジスティック回帰モデルをあてはめてて，その結果を:class:`OptimizeResult` のインスタンスとして受け取り， :obj`res` に保持しています．
最適化手法は ``method`` で，
:func:`minimize` の引数 ``fun`` と ``jac`` には，それぞれロジスティック回帰の目的関数とその勾配ベクトル，すなわち :ref:`lr-lr` の式(2)と式(4)を計算するメソッドを与えています．
これらのメソッドについては次の :ref:`lr-loss` で詳しく述べます．
最適解を探索する初期値 ``x0`` には :func:`np.zeros` で生成した実数の0ベクトルを与えています．
パラメータの総数は，特徴数に切片 (intercept) の分を加えた数にしています．
目的関数と勾配ベクトルを計算するにはモデルのパラメータの他にも訓練データの情報が必要です．
そこで，これらの情報を ``args`` に指定して，目的関数・勾配ベクトルを計算するメソッドに引き渡されるようにしています．





.. _lr-fit-sarray:

構造化配列
----------

.. code-block:: python

    # dtype for model parameters to optimize
    self._param_dtype = np.dtype([
        ('coef', float, self.n_features_),
        ('intercept', float)
    ])

.. code-block:: python

    # get result
    self.coef_ = res.x.view(self._param_dtype)['coef'][0].copy()
    self.intercept_ = res.x.view(self._param_dtype)['intercept'][0]



最適化が終わったら， :obj:`res` の属性 :attr:`x` に格納されているパラメータを取り出します．
ロジスティック回帰のクラスでは，重みベクトル :math:`\mathbf{w}` と切片 :math:`b` のパラメータを，それぞれ属性 :attr:`coef_` と :attr:`intercept_` に保持します．
しかし， これらのパラメータはまとめて1次元配列 :obj:`res.x` に格納されているので，それを :meth:`view` を使って分離しています．
この処理については次の :ref:`lr-loss` で詳しく述べます．
なお，ローカル変数である :obj:`res.x` は :meth:`fit` メソッドの終了時にその内容が失われるので， :meth:`copy` メソッドで実体をコピーしていることに注意して下さい．


.. _lr-fit:

学習メソッドの実装
==================

.. _lr-fit-fit:

それでは，前節の :ref:`lr-optimization` で紹介した :func:`minimize` を用いて，学習メソッド :meth:`fit` を実装します．
:func:`minimize` とパラメータをやりとりするために，構造化配列を用いる方法についても紹介します．

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

構造化配列
----------




.. code-block:: python

    # dtype for model parameters to optimize
    self._param_dtype = np.dtype([
        ('coef', float, self.n_features_),
        ('intercept', float)
    ])


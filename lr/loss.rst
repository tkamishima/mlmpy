.. _lr-loss:

損失関数とその勾配
==================

:ref:`lr-fit` では， :func:`minimize` を用いて，データにあてはめて，パラメータを推定しました．
しかし， :func:`minimize` に引き渡す目的関数とその勾配関数はまだ実装していませんでした．
ここでは，これらを実装して，ロジスティック回帰の学習部分を完成させます．

.. index:: minimize

.. _lr-loss-loss:

損失関数
--------

まず :ref:`lr-lr` の式(2)で示した損失関数を実装します．
この損失関数を :func:`minimize` に引数 ``fun`` として渡すことで，この関数値が最小になるようなパラメータを求めます．

関数は，次のようにメソッドとして定義します．

.. code-block:: python

    def loss(self, params, X, y):
        """ A loss function
        """

この関数は :func:`minimze` から呼び出されます．
このとき，``self`` の次の第1引数には目的関数のパラメータが渡されます．
このパラメータは :func:`minimize` の初期値パラメータ ``x0`` と同じ大きさの実数型の1次元配列です．
2番目以降の引数は :func:`minimize` で引数 ``args`` で指定したものが渡されます．
損失関数の計算には訓練データが必要なので， :func:`minimize` では :obj:`X` と :obj:`y` を :meth:`fit` では渡していましたので，これらがこの損失関数に引き渡されています．

:meth:`fit` メソッドでは，ロジスティック回帰モデルのパラメータは，構造化配列を使って1次元配列にまとめていました．
これを再び，重みベクトル :math:`mathbf{w}` と 切片 :math:`b` それぞれに相当する :obj:`coef` と :obj:`intercept` に分けます．

.. code-block:: python

        # decompose parameters
        coef = params.view(self._param_dtype)['coef'][0, :]
        intercept = params.view(self._param_dtype)['intercept'][0]

このように， :meth:`view` メソッドを使って :ref:`lr-fit-implementation` で紹介したのと同じ方法で分けることができます．


これで損失関数の計算に必要なデータやパラメータが揃いました．
あとは， :ref:`lr-lr` の式(2)に従って損失を計算し， メソッドの返り値としてその値を返せば完成です．

.. code-block:: python

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # likelihood
        l = np.sum((1.0 - y) * np.log(1.0 - p) + y * np.log(p))

        # L2 regularizer
        r = np.sum(coef * coef) + intercept * intercept

        return - l + 0.5 * self.C * r

``p`` は， :math:`\Pr[y | \mathbf{x}; \mathbf{w}, b]` ， ``l`` は大数尤度，そして ``r`` は :math:`L_2` 正則化項にそれぞれ該当します．

.. _lr-loss-grad:

損失関数の勾配
--------------

.. code-block:: python

    def grad_loss(self, params, X, y):
        """ A gradient of a loss function
        """

        # decompose parameters
        coef = params.view(self._param_dtype)['coef'][0, :]
        intercept = params.view(self._param_dtype)['intercept'][0]

        # create empty gradient
        grad = np.empty_like(params)
        grad_coef = grad.view(self._param_dtype)['coef']
        grad_intercept = grad.view(self._param_dtype)['intercept']

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # gradient of weight coefficients
        grad_coef[0, :] = np.dot(p - y, X) + self.C * coef

        # gradient of an intercept
        grad_intercept[0] = np.sum(p - y) + self.C * intercept

        return grad




:func:`np.zeros` ， :func:`np.ones` ，および :func:`np.empty` には，それぞれ今までに生成した配列と同じ大きさの配列を生成する関数 :func:`np.zeros_like` ， :func:`np.ones_like` ，および :func:`np.empty_like` があります．

.. index:: zeros_like

.. function:: np.zeros_like(a, dtype=None)

   Return an array of zeros with the same shape and type as a given array.

.. index:: ones_like

.. function:: np.ones_like(a, dtype=None)

   Return an array of ones with the same shape and type as a given array.

.. index:: empty_like

.. function:: np.empty_like(a, dtype=None)

   Return a new array with the same shape and type as a given array.


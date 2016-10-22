.. _lr-loss:

損失関数とその勾配
==================

:ref:`lr-fit` では， :func:`minimize` を用いて，データにあてはめて，パラメータを推定しました．
しかし， :func:`minimize` に引き渡す目的関数とその勾配関数はまだ実装していませんでした．
ここでは，これらを実装して，ロジスティック回帰の学習部分を完成させます．

.. _lr-loss-loss:

損失関数
--------


.. code-block:: python

    def loss(self, params, X, y):
        """ A loss function
        """

        # decompose parameters
        coef = params.view(self._param_dtype)['coef'][0, :]
        intercept = params.view(self._param_dtype)['intercept'][0]

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # likelihood
        # \sum_{x,s,y in D} (1 - y) log(1 - sigma) + y log(sigma)
        l = np.sum((1.0 - y) * np.log(1.0 - p) + y * np.log(p))

        # L2 regularizer
        r = np.sum(coef * coef) + intercept * intercept

        return - l + 0.5 * self.C * r







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


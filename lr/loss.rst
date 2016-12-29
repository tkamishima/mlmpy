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

今度は :ref:`lr-lr` の式(4)で示した損失関数の勾配を実装し，これを :func:`minimize` に引数 ``jac`` として渡します．
勾配関数に引き渡される引数は，損失関数のそれと同じになります．
また，パラメータは重みベクトルと切片に，損失関数と同じ方法で分けます．

スカラーである損失とは異なり，勾配はパラメータと同じ大きさの配列です．
そこでパラメータと同じ大きさの1次元配列を用意し，そこに重みベクトルと切片のための領域を割り当てます．

.. code-block:: python

        # create empty gradient
        grad = np.empty_like(params)
        grad_coef = grad.view(self._param_dtype)['coef']
        grad_intercept = grad.view(self._param_dtype)['intercept']

入力パラメータ ``params`` と同じ大きさの配列を確保するのに，ここでは :func:`np.empty_like` を用います．
:func:`np.zeros_like` ， :func:`np.ones_like` ，および :func:`np.empty_like` は，今までに生成した配列と同じ大きさの配列を生成する関数で，それぞれ :func:`np.zeros` ， :func:`np.ones` ，および :func:`np.empty` に対応しています．

.. index:: zeros_like

.. function:: np.zeros_like(a, dtype=None)

   Return an array of zeros with the same shape and type as a given array.

.. index:: ones_like

.. function:: np.ones_like(a, dtype=None)

   Return an array of ones with the same shape and type as a given array.

.. index:: empty_like

.. function:: np.empty_like(a, dtype=None)

   Return a new array with the same shape and type as a given array.

この確保した領域 ``grad`` を，重みベクトルと切片にそれぞれ対応する， :obj:`grad_coef` と :obj:`grad_intercept` に分けます．
これには :meth:`view` メソッドを用いますが，今までのパラメータ値の読み出しだけの場合と異なり，値を後で代入する必要があります．
そのため，最初の要素を取り出すことはせず，配列のまま保持します．

これで勾配の計算に必要なものが揃いましたので，  :ref:`lr-lr` の式(4)に従って勾配を計算します．

.. code-block:: python

        # predicted probabilities of data
        p = self.sigmoid(np.dot(X, coef) + intercept)

        # gradient of weight coefficients
        grad_coef[0, :] = np.dot(p - y, X) + self.C * coef

        # gradient of an intercept
        grad_intercept[0] = np.sum(p - y) + self.C * intercept

        return grad

``p`` は，損失関数と同じく :math:`\Pr[y | \mathbf{x}; \mathbf{w}, b]` です．
重みベクトルについての勾配を計算したあと，保持していた配列 ``grad_coef`` の第1行目に代入しています．
切片についての勾配も，同様に ``grad_intercept`` の最初の要素に代入します．
これら二つの勾配は ``grad`` にまとめて格納できているので，これを返します．

この勾配を計算するのに， :func:`np.dot` を用いていますので，この関数を最後に紹介します．

.. index:: dot

.. function:: np.dot(a, b)

    Dot product of two arrays.

3次元以上の配列についても動作が定義されていますが，ここでは2次元までの配列についての動作について紹介します．
1次元配列同士では，ベクトルの内積になります．

.. code-block:: ipython

    In [10]: a = np.array([10, 20])
    In [10]: b = np.array([[1, 2], [3, 4]])
    In [11]: np.dot(a, a)
    Out[11]: 500

2次元配列同士では行列積になります．

.. code-block:: ipython

    In [12]: np.dot(b, b)
    Out[12]:
    array([[ 7, 10],
           [15, 22]])

1次元配列と2次元配列では，横ベクトルと行列の積になります．

.. code-block:: ipython

    In [13]: np.dot(a, b)
    Out[13]: array([ 70, 100])

2次元配列と1次元配列では，行列と縦ベクトルの積になります．

.. code-block:: ipython

    In [14]: np.dot(b, a)
    Out[14]: array([ 50, 110])

以上で，損失関数とその勾配を求めるメソッドが実装できました．
これにより :ref:`lr-fit` で実装した :meth:`fit` メソッドでロジスティック回帰モデルの学習ができるようになりました．

.. index:: matmul

.. [#]

    Python 3.5 以上では，行列積演算子 ``@`` が利用できますが， :func:`np.dot` とは若干異なる :func:`np.matmul` が適用されます．
    すなわち ``a @ b`` は ``np.matmul(a, b)`` と等価です．
    3次元の配列での挙動と，スカラー同士の演算が許されない点が異なります．

.. _nbayes2-broadcasting:

ブロードキャスト
================

この節では，NumPy 配列の次元数や大きさを操作する方法と，NumPy の強力な機能であるブロードキャストを紹介します．

.. _nbayes2-broadcasting-shape:

.. index:: ! ndarray; shape, ! ndarray; ndim

NumPy 配列の次元数や大きさの操作
--------------------------------

ブロードキャストを紹介する前に， :ref:`nbayes1-ndarray` で紹介した，NumPy の配列クラス :class:`np.ndarray` の属性 :attr:`ndim` と :attr:`shape` を操作する方法を紹介します．

:attr:`ndim` は，配列の次元数を表す属性で，ベクトルでは 1 に，配列では 2 になります．
:attr:`shape` は，スカラーや，タプルによって配列の各次元の大きさを表す属性です．
例えば，大きさが 5 のベクトルはスカラー ``5`` によって， :math:`2 \times 3` の行列はタプル ``(2, 3)`` となります．

.. index:: transpose

次元数を操作する必要がある例として配列の転置の例を紹介します．
転置した配列を得るには，属性 :attr:`T` か，メソッド :meth:`transpose` を用います．
2次元の配列である行列を転置してみましょう：

.. code-block:: ipython

    In [10]: a = np.array([[1, 3], [2, 1]])
    In [11]: a
    Out[11]:
    array([[1, 3],
           [2, 1]])
    In [12]: a.T
    Out[12]:
    array([[1, 2],
           [3, 1]])
    In [13]: a.transpose()
    Out[13]:
    array([[1, 2],
           [3, 1]])

今度は，1次元配列であるベクトルを転置してみます：

.. code-block:: ipython

    In [14]: b = np.array([10, 20])
    In [15]: b
    Out[15]: array([10, 20])
    In [16]: b.T
    Out[16]: array([10, 20])

転置しても，縦ベクトルになることはありません．属性 :attr:`T` やメソッド :meth:`transpose` は，次元数 :attr:`ndim` が 1 以下であれば，元と同じ配列を返します．


.. _nbayes2-broadcasting-broadcasting:

.. index:: ! broadcasting

ブロードキャスト
----------------

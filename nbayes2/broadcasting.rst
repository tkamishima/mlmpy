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



.. _nbayes2-broadcasting-broadcasting:

.. index:: ! broadcasting

ブロードキャスト
----------------

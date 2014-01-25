.. _nbayes2-distlearn:

.. index:: broadcasting

分布の学習の実装
================

前節で説明したブロードキャストの機能を用いると， :obj:`for` ループを用いなくても，複数の要素に対する演算をまとめて行うことができます．
この節では，その方法を，単純ベイズの学習でのクラスや特徴の分布の計算の実装を通じて説明します．

.. _nbayes2-distlearn-classlearn:

クラス分布の学習
----------------

:ref:`nbayes2-fit2-fitif` 節で紹介した，クラス分布の計算の比較演算を用いた次の実装を，ブロードキャストの機能を用いて実装します．

.. code-block:: python

    nY = np.zeros(n_classes, dtype=np.int)
    for yi in xrange(n_classes):
        for i in xrange(n_samples):
            if y[i] == yi:
                nY[yi] += 1


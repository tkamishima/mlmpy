.. _lr-predict_run:

実行と予測
==========

学習したモデルを使って予測をするメソッドを実装してロジスティック回帰のクラスを完成させます．
その後，このクラスを使って，最適化手法の実行速度を比較してみます．

.. _lr-predict_run-predict:

予測
----

ロジスティック回帰では， :ref:`lr-lr` の式(5)によってクラスを予測します．
分類する事例の特徴ベクトルを各行に格納した2次元配列 :obj:`X` が入力です．
このとき，クラスが ``1`` になる確率 :math:`\Pr[y{=}1 | \mathbf{x}; \mathbf{w}^\ast, b]` を計算します．

.. code-block:: python

        # predicted probabilities of data
        p = self.sigmoid(np.sum(X * self.coef_[np.newaxis, :], axis=1) +
                         self.intercept_)

学習により獲得した重みベクトルは :obj:`self.coef_` に，切片は :obj:`self.intercept_` に格納されています．
:obj:`X` の各行の特徴ベクトルと重みベクトルの内積に，切片を加えて，シグモイド関数を適用することで，各事例のクラスが ``1`` になる確率を要素とする1次元の配列を得ます．

この確率が :math:`0.5` 未満かどうかでクラスを予測します．
これには3項演算子に該当する :func:`np.where` を用います．

.. index:: where

.. function:: (condition, x, y)

    Return elements, either from x or y, depending on condition.

条件が成立したときは ``x`` を，そうでないときは ``y`` の値になります．
クラスが :math:`1` になる確率が :math:`0.5` 未満であれば，クラス ``0`` に，それ以外で ``1`` に分類する実装は次のとおりです．

.. code-block:: python

        return np.where(p < 0.5, 0, 1)

:func:`np.where` はユニバーサル関数なので，このように全ての事例をまとめて分類することができます．

.. _lr-predict_run-run:

実行と速度比較
--------------

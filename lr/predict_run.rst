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

実行
----

実行可能な状態の :class:`LogisticRegression` の実行スクリプトは，以下の場所から取得できます．
実行時には ``lr.py`` と ``iris2.tsv`` [#]_ がカレントディレクトリに必要です．

.. index:: sample; run_lr.py

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/run_lr.py

.. only:: html and not epub

  :download:`LogisticRegression 実行スクリプト：run_lr.py <../source/run_lr.py>`

.. index:: structured array, genfromtxt

このスクリプトでは，データを :func:`np.genfromtxt` で読み込むときに，構造化配列を利用しました．

.. code-block:: python

    # load data
    data = np.genfromtxt('iris2.tsv',
                         dtype=[('X', float, 4), ('y', int)])

最初の4列は実数型の特徴ベクトルとして ``X`` で参照できるように，残りの1列は整数型のクラスとして ``y`` で参照できるようにしています．
すると，次のように特徴ベクトルとクラスを分けて :meth:`fit` メソッドに渡すことができます．

.. code-block:: python

    clr.fit(data['X'], data['y'])

.. only:: not latex

   .. rubric:: 注釈

.. [#]
    ``iris2.tsv`` は UCI Repository の
    `Iris Data Set <https://archive.ics.uci.edu/ml/datasets/Iris>`_
    をもとに作成したものです．
    Fisherの判別分析の論文で用いられた著名なデータです．
    3種類のアヤメのうち， Iris Versicolour と Iris Virginica の2種類を取り出しています．

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
これには3項演算子に該当する :func:`where` を用います．

.. index:: where

.. function:: np.where(condition, x, y)

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

.. index:: optimization

.. _lr-predict_run-compare:

最適化手法の比較
----------------

最後に最適化手法の違いについて調べてみます．
``lr.py`` の :meth:`fit` メソッドでの最適化関数 :func:`minimize` の呼び出しを次のように変更してみます．

.. code-block:: python

    res = minimize(fun=self.loss,
                   x0=np.zeros(self.n_features_ + 1, dtype=float),
                   jac=self.grad_loss,
                   args=(X, y),
                   method='Powell',
                   options={'disp': True})

これは勾配情報を使わないPowell法を指定し，さらに最適化の結果を表示するように変更しています．
``run_lr.py`` スクリプトを実行すると，勾配利用しなかった警告が表示されたあと，最適化の結果が次のように表示されます::

    Optimization terminated successfully.
             Current function value: 31.685406
             Iterations: 18
             Function evaluations: 1061

収束するまでに18回の反復が必要で，損失関数の呼び出しは1061回です．
次に，損失関数の勾配を用いる共役勾配法を試してみます．

.. code-block:: python

    res = minimize(fun=self.loss,
                   x0=np.zeros(self.n_features_ + 1, dtype=float),
                   jac=self.grad_loss,
                   args=(X, y),
                   method='CG',
                   options={'disp': True})

十分に収束しなかった旨の警告が表示されますが，上記のPowell法と同等の損失関数値が達成できています::

    Warning: Desired error not necessarily achieved due to precision loss.
             Current function value: 31.685406
             Iterations: 21
             Function evaluations: 58
             Gradient evaluations: 46

収束までの反復数は21回とやや多いですが，損失関数とその勾配の呼び出しはそれぞれ58回と46回とずっと少なくなっています．
最後に，二次の微分であるヘシアンも計算するBFGS法を試してみます．

.. code-block:: python

    res = minimize(fun=self.loss,
                   x0=np.zeros(self.n_features_ + 1, dtype=float),
                   jac=self.grad_loss,
                   args=(X, y),
                   method='CG',
                   options={'disp': True})

最適化は収束し，今までと同等の損失関数値が達成できています::

    Optimization terminated successfully.
             Current function value: 31.685406
             Iterations: 11
             Function evaluations: 15
             Gradient evaluations: 15

反復数は11と最も速く収束しており，損失関数やその勾配の評価回数も，共役勾配法より減少しています．

以上の結果からすると，収束が速く，関数の評価回数も少ないBFGS法が優れているように見えます．
しかし，BFGS法は2次微分であるヘシアン行列を計算するため，パラメータ数が多い場合には多くの記憶領域を必要とします．
よって問題の性質や規模に応じて最適化手法は選択する必要が生じます．

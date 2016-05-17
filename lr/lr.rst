.. _lr-lr:

ロジスティック回帰の形式的記述
==============================

ロジスティック回帰を実装する前に，この手法について簡単に復習します．

.. index:: logistic regression

変数を定義します．
:ref:`nbayes1-nbayes` とほぼ同じ表記を用いますが，特徴はカテゴリ値ではなく，実数値をとります．

* 特徴量 :math:`\mathbf{x}_i=(x_{i1}, \ldots, x_{iK})` の要素 :math:`x_{ij}` は実変数です．
  ただし， :math:`K` は特徴の種類数です．
* クラス :math:`y` は， :math:`\{0, 1\}` のうちの一つをとります．
* データ集合は :math:`\mathcal{D}=\{\mathbf{x}_i, y_i\},\,i=1,\ldots,N` です．
  ただし， :math:`N` はデータ数です．

ロジスティック回帰では， :math:`\mathbf{x}` が与えられたときの :math:`y` の条件付き確率を次式でモデル化します．

.. math::
   :label: eq-lr-model

    \Pr[y{=}1 | \mathbf{x}; \mathbf{w}, b] = \mathrm{sig}(\mathbf{w}^\top \mathbf{x} + b)

ただし， :math:`\mathbf{w}` は次元数 :math:`K` の重みパラメータ， :math:`b` はバイアスパラメータ（切片パラメータ）です．
また， :math:`\mathrm{sig}(a)` は次のシグモイド関数です．

.. math::

    \mathrm{sig}(a) = \frac{1}{1 + \exp(-a)}

学習には，正則化の度合いを決める超パラメータ :math:`\lambda` を導入した次の目的関数を用います．

.. math::
   :label: eq-lr-objective

    \mathcal{L}(\mathbf{w}, b; \mathcal{D}) & =
    - \log \sum_{(\mathbf{x}_i, y_i) \in \mathcal{D}}
    \Pr[y_i | \mathbf{x}_i; \mathbf{w}, b]
    + \frac{\lambda}{2} \left({\|\mathbf{w}\|_2}^2 + b^2\right) \\
    &= - \sum_{(\mathbf{x}_i, y_i) \in \mathcal{D}}
    \left\{ (1 - y_i) \log(1 - \Pr[y_i{=}1 | \mathbf{x}_i]) + \right.\\
    & \qquad\left.
    y_i \log \Pr[y_i{=}1 | \mathbf{x}_i] \right\}
    + \frac{\lambda}{2} \left({\|\mathbf{w}\|_2}^2 + b^2\right)

なお， :math:`\Pr[y|\mathbf{x}]` 中のパラメータは簡潔のため省略しました．
この目的関数は，訓練データ集合 :math:`\mathcal{D}` に対する負の対数尤度に :math:`L_2` 正則化項を加えたものです．
パラメータ :math:`\mathbf{w}` と :math:`b` はこの目的関数を最小化することで学習します．

.. math::
   :label: eq-lr-learning

    \{\mathbf{w}, b\} =
    \arg \min_{\{\mathbf{w}, b\}} \mathcal{L}(\mathbf{w}, b; \mathcal{D})

この最小化問題は反復再重み付け最小二乗法 (iteratively reweighted least squares method) により求めるのが一般的です．
しかし，本章では，他の多くの最適化問題として定式化された機械学習手法の実装の参考となるように， `SciPy` の非線形最適化ルーチンを利用して解きます．
非線形最適化では目的関数の勾配も利用するので，ここに追記しておきます．

.. math::

    \frac{\partial}{\partial\mathbf{w}}
    \mathcal{L}(\mathbf{w}, b; \mathcal{D}) & =
    \sum_{(\mathbf{x}_i, y) \in \mathcal{D}}
    (\Pr[y_i{=}1 | \mathbf{x}_i] - y_i) \mathbf{x} + \lambda \, \mathbf{w} \\
    \frac{\partial}{\partial b}
    \mathcal{L}(\mathbf{w}, b; \mathcal{D}) & =
    \sum_{(\mathbf{x}, y) \in \mathcal{D}}
    (\Pr[y_i{=}1 | \mathbf{x}_i] - y_i) 1 + \lambda \, b

パラメータ :math:`b` に対する勾配で :math:`1` を明示しているのは，後の実装上の有利になるためです．

学習したパラメータ :math:`\mathbf{w}` と :math:`b` を式 :eq:`eq-lr-model` に代入した分布 :math:`\Pr[y | \mathbf{x}]` 用いて，新規入力データ :math:`\mathbf{x}^\mathrm{new}` に対するラベル :math:`y` は次式で予測できます．

.. math::
   :label: eq-lr-class-dist

   y =
   \begin{cases}
        1, \text{ if } \Pr[y | \mathbf{x}] \ge 0.5 \\
        0, \text{otherwise}
   \end{cases}

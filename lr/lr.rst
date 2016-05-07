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

    \Pr[y | \mathbf{x}] = \mathrm{sig}(\mathbf{w}^\top \mathbf{x} + b)

ただし， :math:`\mathbf{w}` は次元数 :math:`K` の重みパラメータ， :math:`b` はバイアスパラメータ（切片パラメータ）である．
また， :math:`\mathrm{sig}(a)` は次のシグモイド関数である．

.. math::
   :label: eq-lr-sigmoid

   \mathrm{sig}(a) = \frac{1}{1 + \exp(-x)}

学習したパラメータ :math:`\mathbf{w}` と :math:`b` を式 :eq:`eq-lr-model` に代入した分布 :math:`\Pr[y | \mathbf{x}]` 用いて，新規入力データ :math:`\mathbf{x}^\mathrm{new}` に対するラベル :math:`y` は次式で予測する．

.. math::
   :label: eq-lr-class-dist

   y =
   \begin{cases}
        1, \text{ if } \Pr[y | \mathbf{x}] \ge 0.5 \\
        0, \text{otherwise}
   \end{cases}

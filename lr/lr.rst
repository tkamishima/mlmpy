.. _lr-logisticreg:

ロジスティック回帰
==================

ロジスティック回帰を実装する前に，この手法について簡単に復習します．

.. index:: logistic regression

変数を定義します．
:ref:`nbayes1-fit1-feature` とほぼ同じ表記を用いますが，特徴はカテゴリ値ではなく，実数値をとります．

* 特徴量 :math:`\mathbf{x}_i=(x_{i1}, \ldots, x_{iK})` の要素 :math:`x_{ij}` は実変数です．
  ただし， :math:`K` は特徴の種類数です．
* クラス :math:`y` は， :math:`\{0, 1\}` のうちの一つをとります．
* データ集合は :math:`\mathcal{D}=\{\mathbf{x}_i, y_i\},\,i=1,\ldots,N` です．
  ただし， :math:`N` はデータ数です．



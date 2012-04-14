.. _nbayes1-preliminary:

準備
====

単純ベイズ法を実装する前の準備として，実装するアルゴリズムについて簡単に復習し，実装の説明に関し Numpy の配列について最低限必要なことを述べます．

.. _nbayes1-preliminary-nbayes:

特徴がカテゴリ変数である単純ベイズ
----------------------------------

ここでは，特徴がカテゴリ変数である場合の単純ベイズ法による分類について，ごく簡単にまとめます．

.. index::
   single: naive Bayes

* 特徴量 :math:`\mathbf{x}_i=(x_{i1}, \ldots, x_{ip})` の要素 :math:`x_{ij}`
  はカテゴリ変数で， :math:`m_j` 個の値のうちの一つをとります．
* クラス :math:`y` は， :math:`c` 個の値のうちの一つをとります．
* 特徴 :math:`\mathbf{X}` は，クラス :math:`Y` が与えられたとき条件付き独立であることから，
  :math:`\mathbf{X}` と :math:`Y` の同時分布は次式で与えられます．

.. math::
   :label: nbayes1-joint-prob

   \Pr[\mathbf{X}, Y] = \Pr[Y] \prod_{j=1}^k \Pr[X_j | Y]

* 大きさ :math:`n` のデータ集合 :math:`\mathcal{D}=\{\mathbf{x}_i, y_i\}`
  に対する，対数尤度は次式で与えられます．

.. math::
   :label: nbayes1-likelihood

   \mathcal{L}(\mathcal{D}; \Theta) = \sum_{(\mathbf{x}_i, y_i)\in\mathcal{D}} \ln\Pr[\mathbf{x}_i, y_i]

さらに， :math:`\Pr[Y]` と :math:`\Pr[X_j|Y]` の分布をカテゴリ分布（離散分布）とします．すると，以下の量をパラメータとして求めれば，単純ベイズの分類器が学習できます．

.. math::
   :label: nbayes1-param

   \Pr[y],&\quad y=1, \ldots, c\\
   \Pr[x_i | y],&\quad y=1,\ldots,c,\; i=1,\ldots,p,\;x_i=1,\ldots,m_i

これらのパラメータは，データ集合 :math:`\mathcal{D}` 中のデータの分割表を作成すれば計算できます．
ここでは，さらに簡単に， :math:`c` や :math:`m_j` は全て 2 とします．
すなわち，クラスや各特徴量は全て2値変数となり，これにより :math:`\Pr[Y]` と :math:`\Pr[X_j|Y]` は，カテゴリ分布の特殊な場合であるベルヌーイ分布に従うことなります．

推論をするときには，入力ベクトル :math:`\mathbf{x}_\mathrm{new}` が与えられたときのクラスの事後確率を最大にするクラスを，次式で求めます．

.. math::
   :label: nbayes1-inferance

   \hat{y} &= \arg\max_y \Pr[y|\mathbf{x}_\mathrm{new}] \\
           &= \arg\max_y \frac{\Pr[y, \mathbf{x}_\mathrm{new}]}{\sum_{y'} \Pr[y']\Pr[y', \mathbf{x}_\mathrm{new}]}

.. _nbayes1-preliminary-array:

Numpy 配列の基礎
----------------

.. index::
   single: np.ndarray

Numpy で最も重要なクラスである :class:`np.ndarray` について，基本的な機能を紹介します．
:ref:`intro-intro` では，機能を単独では説明しないと述べましたが，そうした説明を全くしないで書くのは難しすぎるので，この節だけ忍耐強く読んで下さい．

.. todo::
   配列の作り方
   要素の参照（行や列の取り出しはやらない）
   ndim, shape, dtypeの説明
   dtype が全部同じ場合，Structured arrayは存在のみ示唆
   
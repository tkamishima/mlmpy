.. _nbayes2-summary:

まとめ
======

:ref:`nbayes2` の章では，単純ベイズ法の実装を改良することで，以下の内容を紹介しました．

* :ref:`nbayes2-class`

  * 抽象クラスを用いて，実装の一部だけを変更したクラスを設計する方法

* :ref:`nbayes2-fit2`

  * 比較演算を行うユニバーサル関数
  * :func:`np.sum` を用いた数え上げの方法

* :ref:`nbayes2-shape`

  * :const:`np.newaxis` による，配列の次元数と :attr:`shape` の変更
  * :meth:`reshape` メソッドや :func:`np.reshape` 関数による :attr:`shape` の変更
  * :attr:`T` 属性や :func:`np.transepose` 関数による配列の転置

* :ref:`nbayes2-broadcasting`

  * ブロードキャスト機能：次元数を統一する規則，出力配列の :attr:`shape` の決定方法，ブロードキャスト可能性の判定，および演算要素の対応付け

* :ref:`nbayes2-distclass`

  * ブロードキャスト機能を用いた実装例
  * 実数を返す割り算関数 :func:`np.true_divide`

* :ref:`nbayes2-distfeature`

  * ブロードキャスト機能を用いた実装例
  * 論理演算のユニバーサル関数 :func:`np.logical_and`

* :ref:`nbayes2-timeit`

  * ``ipython`` 内での， ``%timeit`` による関数の実行速度の計測

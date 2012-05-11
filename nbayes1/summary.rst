.. _nbayes1-summary:

まとめ
======

:ref:`nbayes1` の章では，単純ベイズ法の実装を通じて以下の内容を紹介しました．

* :ref:`nbayes1-ndarray`

  * NumPy 配列 :class:`np.ndarray` の特徴
  * :func:`np.array` による NumPy 配列の生成
  * :func:`np.zeros` など，その他の関数による NumPy 配列の生成
  * NumPy 配列 :class:`np.ndarray` クラスの属性
  * NumPy 値の型 :class:`np.dtype`
  * NumPy 配列の値の参照方法
  * NumPy 配列と，数学のベクトルや行列との対応

* :ref:`nbayes1-nbayes`

  * 特徴がカテゴリ変数である場合の単純ベイズ法

* :ref:`nabyes1-spec`

  * 入力データの仕様例
  * 機械学習アルゴリズムをクラスとして実装する利点
  * scikit-learn モジュールのAPI基本仕様
  * 機械学習アルゴリズムのクラスの仕様例

* :ref:`nbayes1-fit1`

  * NumPy 配列の基本的な参照を用いたアルゴリズムの実装

* :ref:`nbayes1-predict`

  * NumPy 配列のスライスを使った参照
  * ユニバーサル関数によるベクトル化演算
  * :obj:`for` ループを用いない実装の例
  * :func:`np.sum` の紹介．特に， ``axis`` 引数について
  * :func:`np.argmax` ， :func:`np.argmin`

* :ref:`nbayes1-run`

  * :func:`np.genfromtxt` を用いたテキスト形式ファイルの読み込み
  * scikit-learn API基本仕様に基づくクラスの利用

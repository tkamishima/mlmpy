.. _lr-summary:

まとめ
======

:ref:`lr` の章では，ロジスティック回帰法の実装を通じて以下の内容を紹介しました．

* :ref:`lr-lr`

    * ロジスティック回帰法

* :ref:`lr-sigmoid`

    * 静的メソッドによる数値関数の実装
    * ネピアの数や円周率などの定数
    * :func:`np.seterr` による浮動小数点エラーの処理方法の設定
    * オーバーフロー・アンダーフローへの対策
    * :func:`np.vectorize` を用いたユニバーサル関数への変換
    * :func:`np.piecewize` による区分関数の定義
    * 数値を一定の範囲に収める :func:`np.clip` 関数

* :ref:`lr-optimization`

    * SciPy の非線形最適化関数 :func:`sp.optimize.minimize_scalar` と :func:`sp.optimize.minimize` の紹介
    * 最適化の結果を返すためのクラス :class:`OptimizeResult` の紹介
    * 各種の最適化手法の特徴

* :ref:`lr-fit`

    * 最適化関数を用いた，モデルのパラメータの学習
    * 構造化配列
    * 構造化配列と :meth:`view` メソッドによる同一領域の異なる参照方法

* :ref:`lr-loss`

    * :func:`sp.optimize.minimize` からのコールバック
    * :func:`np.empty_like` などを用いた行列の生成
    * :func:`np.dot` による内積と行列積

* :ref:`lr-predict_run`

    * 3項演算を行う :func:`np.where` 関数
    * 構造化配列を用いたデータの読み込み
    * 最適化手法の実行結果の比較

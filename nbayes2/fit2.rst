.. _nbayes2-fit2:

単純ベイズの実装 （２）
=======================

:ref:`nbayes1` の :ref:`nbayes1-fit1` 節では，単純ベイズ法の学習を，NumPy 配列を単に多次元配列として利用し， :obj:`for` ループで，訓練データを数え挙げて，単純ベイズの学習を実装しました．
ここでは， :func:`np.sum` 関数などを用いて，NumPy の利点を生かして実装をします．

.. _nbayes2-fit2-pre:

予測メソッドの実装の準備
------------------------

それでは， :class:`NaiveBayes1` クラスとは，学習メソッドの実装だけが異なる :class:`NaiveBayes2`
クラスの作成を始めます．
コンストラクタや予測メソッドは :class:`NaiveBayes1` クラスと共通なので，この :class:`NaiveBayes2`
クラスも，抽象クラス :class:`BaseBinaryNaiveBayes` の下位クラスとして作成します．
クラスの定義と，コンストラクタの定義は，クラス名を除いて :class:`NaiveBayes1` クラスと同じです．

.. code-block:: python

    class NaiveBayes2(BaseBinaryNaiveBayes):
        """
        Naive Bayes class (2)
        """

        def __init__(self):
            super(NaiveBayes2, self).__init__()


学習を行う :meth:`fit` メソッドも，引数などの定義は :class:`NaiveBayes1` クラスのそれと全く同じです．
さらに，サンプル数 :obj:`n_samples` などのメソッド内の定数の定義も， :ref:`nbayes1-fit1-const` 節で述べたものと共通です．

.. _nbayes2-fit2-fitif:

比較演算を利用したクラスごとの事例数の計算
------------------------------------------

:ref:`nbayes1-nbayes` の式(4)のクラスの分布のパラメータを求めるために，各クラスごとの事例数を :class:`NaiveBayes1` クラスでは，次のようにして求めました．

.. code-block:: python

    nY = np.zeros(n_classes, dtype=np.int)
    for i in xrange(n_samples):
        nY[y[i]] += 1

この実装は，クラスの対応する添え字の要素のカウンタを一つずつ増やす実装になっていました．
これを，各クラスごとに，そのクラスの事例かどうかを判断し，もしそうであったなら対応する要素のカウンタを一つずつ増やす実装にします．

.. code-block:: python

    nY = np.zeros(n_classes, dtype=np.int)
    for yi in xrange(n_classes):
        for i in xrange(n_samples):
            if y[i] == yi:
                nY[yi] += 1

外側のループの添え字 :obj:`yi` は処理対象のクラスを指定し，その次のループの添え字 :obj:`i` は処理対象の事例を指定しています．
ループの内部では，対象事例のクラスが，現在の処理対象クラスであるかどうかを，等号演算によって判定し，もし結果が真であれば，対応するカウンタの値を一つずつ増やしています．

.. _nbayes2-fit2-fitif-ufunc:

ユニバーサル関数の利用
^^^^^^^^^^^^^^^^^^^^^^

このコードの中で，内側のループでは全ての事例について等号演算を適用していますが，これを，ユニバーサル関数の機能を利用してまとめて処理します．
等号演算 :obj:`==` を適用すると，次の関数が実際には呼び出されます．

.. index:: equal

.. function:: np.equal(x1, x2[, out]) = <ufunc 'equal'>

    Return (x1 == x2) element-wise.

この関数は :obj:`x1` と :obj:`x2` を比較し，その真偽値を論理型で返します．
:obj:`out` が指定されていれば，結果はその配列に格納され，指定されていなければ結果を格納する配列を新たに作成します．

この関数はユニバーサル関数であるため，``y == yi`` を実行すると，配列 :obj:`y` 各要素と，添え字 :obj:`yi` とを比較した結果をまとめた配列を返します．
すなわち， :obj:`y` の要素が :obj:`yi` と等しいときには :const:`True` ，それ以外は :const:`False` を要素とする配列を返します．

この比較結果を格納した配列があれば，このうち :const:`True` の要素の数を数え挙げれば，クラスが :obj:`yi` に等しい事例の数が計算できます．
この数え挙げには，合計を計算する :func:`np.sum` を用います．
論理型の定数 :const:`True` は，整数型に変換すると ``1`` に，もう一方の :const:`False` は変換すると ``0`` になります．
このことを利用すると， :func:`np.sum` を ``y == yi`` に適用することで，配列 :obj:`y` のうち，その値が :obj:`yi` に等しい要素の数が計算できます．

以上のことを利用して，各クラスごとの事例数を数え挙げるコードは次のようになります．

.. code-block:: python

    nY = np.empty(n_classes, dtype=np.int)
    for yi in xrange(n_classes):
        nY[yi] = np.sum(y ==yi)

なお，配列 :obj:`nY` は ``0`` で初期化しておく必要がないため， :func:`np.zeros` ではなく， :func:`np.empty` で作成しています．

.. _nbayes2-fit2-fitif-try:

配列要素の一括処理の試み
^^^^^^^^^^^^^^^^^^^^^^^^

コードは簡潔になりましたが，まだクラスについてのループが残っていますので，さらに簡潔に記述できるか検討します．
ここで， :ref:`nbayes1-predict-logjprob` 節の :ref:`nbayes1-predict-logjprob-2` で紹介した，配列の要素をまとめて処理するテクニックを利用します．
それは，ループの添え字がとりうる値をまとめた配列を :func:`np.arange` 関数によって作成し，対応する添え字がある部分と置き換えるというものでした．

では，添え字 :obj:`yi` について検討します．
この変数は，ループ内で ``0`` から ``n_classes - 1`` まで変化するので， ``np.arange(n_classes)`` により，それらの値をまとめた配列を作成できます．
この配列を導入した，クラスごとの事例数の数え挙げのコードは次のようになります．

.. code-block:: python

    nY = np.sum(y == yi)

しかし，このコードは期待した動作をしません．
ここでは， :obj:`y` 内の要素それぞれが， :obj:`yi` 内の要素それぞれと比較され，それらの和が計算されることを期待していました．
しかし， :obj:`y` も :obj:`yi` も共に1次元の配列であるため，単純に配列の最初から要素同士を比較することになってしまいます．
この問題を避けて， :obj:`y` の各要素と :obj:`yi` 内の各要素をそれぞれ比較するには，それぞれの配列を2次元にして，ブロードキャスト (broadcasting) という機能を利用する必要があります．
次の節では，このブロードキャストについて説明したあとで，ブロードキャストを使った実装について説明します．
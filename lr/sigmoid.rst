.. _lr-sigmoid:

シグモイド関数
==============

.. index:: sigmoid function

ここでは :ref:`lr-lr` の式(1)で用いる次のシグモイド関数を実装します．

.. math::
    :label: eq-lr-sigmoid

    \mathrm{sig}(a) = \frac{1}{1 + \exp(-a)}

この関数の実装を通じ，数値演算エラーの扱い，ユニバーサル関数の作成方法，数学関数の実装に便利な関数などについて説明します．

.. _lr-sigmoid-straightforward:

直接的な実装とその問題点
------------------------

式 :eq:`eq-lr-sigmoid` をそのまま実装してみます．
モジュール :mod:`lr1` [#]_ 中のロジスティック回帰クラスの定義のうち，ここでは関数 :meth:`sigmoid` の部分のみを示します．

.. index:: sample; lr1.py

.. code-block:: python

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

.. index:: staticmethod

なお， `@staticmethod` のデコレータを用いて，静的メソッドとして定義してあります．
:meth:`sigmoid` は数学関数であり，値はその引数だけに依存し，オブジェクトやクラスの内容や状態には依存しないので，このように定義しています．

では，実行してみましょう．
:meth:`sigmoid` は静的メソッドなので，オブジェクトを生成しなくても実行できます．

.. code-block:: ipython

    In [10]: from lr1 import LogisticRegression
    In [11]: LogisticRegression.sigmoid(0.0)
    Out[11]: 0.5

    In [12]: LogisticRegression.sigmoid(1.0)
    Out[12]: 0.7310585786300049
    In [13]: 1.0 / (1.0 + 1.0 / np.e)
    Out[13]: 0.7310585786300049

    In [14]: LogisticRegression.sigmoid(-1.0)
    Out[14]: 0.2689414213699951
    In [15]: 1.0 / (1.0 + np.e)
    Out[15]: 0.2689414213699951

いずれも正しく計算できています．
なお， :const:`np.e` はネピアの数 [#]_ を表す定数です．

さらに，いろいろな値でテストしてみます．

.. code-block:: ipython

    In [20]: LogisticRegression.sigmoid(1000.)
    Out[20]: 1.0

    In [21]: LogisticRegression.sigmoid(-1000.)
    lr1.py:62: RuntimeWarning: overflow encountered in exp
      return 1.0 / (1.0 + np.exp(-x))
    Out[21]: 0.0

シグモイド関数は 1.0 や 0.0 といった値になることは，式 :eq:`eq-lr-sigmoid` の定義からはありえません．
しかし， NumPy での実数演算は，有限精度の浮動小数点を用いて行っているため，絶対値が大きすぎるオーバーフローや，小さすぎるアンダーフローといった浮動小数点エラーを生じ，意図したとおりの計算結果を得ることができません．
そのため，浮動小数点演算の制限を意識してプログラミングする必要があります．

.. [#]

    .. only:: epub or latex

        https://github.com/tkamishima/mlmpy/blob/master/source/lr1.py

    .. only:: html and not epub

        :download:`LogisticRegresshon クラス：lr1.py <../source/lr1.py>`

.. index:: e, pi, sp.constants

.. [#]

    NumPy には，このネピアの数を表す :const:`np.e` の他に，円周率を表す :const:`np.pi` の定数があります．
    SciPy の :mod:`sp.consants` モジュール内には，光速や重力定数などの物理定数が定義されています．

.. _lr-sigmoid-errhandling:

浮動小数点エラーの処理
^^^^^^^^^^^^^^^^^^^^^^

意図した計算結果を得ることができないこの問題の他に，オーバーフローが生じていることの警告メッセージが表示されてしまう問題も生じています．
もちろん，この警告メッセージは意図した結果が得られていないことを知るために役立つものです．
しかし，浮動小数点エラーを，無視してかまわない場合や，例外として処理したい場合など，警告メッセージ表示以外の動作が望ましい場合もあります．
このような場合には，次の :func:`np.seterr` を用いて，浮動小数点演算のエラーに対する挙動を変更できます．

.. index:: seterr

.. function:: np.seterr(all=None, divide=None, over=None, under=None,
    invalid=None)[source]

    Set how floating-point errors are handled.

:obj:`divide` は0で割ったときの0除算，:obj:`over` は計算結果の絶対値が大きすぎる場合のオーバーフロー，:obj:`under` は逆に小さすぎる場合のアンダーフロー，そして :obj:`invalid` は対数の引数が負数であるなど不正値の場合です．
:obj:`all` はこれら全ての場合についてまとめて挙動を変更するときに用います．

そして，``np.seterr(all=`ignore`)`` のように，キーワード引数の形式で下記の値を設定することで挙動を変更します．

* :const:`warn`: 警告メッセージを表示するデフォルトの挙動です．
* :const:`ignore`: 数値演算エラーを無視します．
* :const:`raise`: 例外 :exc:`FloatingPointError` を送出します．

その他 :const:`call` ， :const:`print` ，および :const:`log` の値を設定できます．

.. _lr-sigmoid-fpcheck:

浮動小数点の制限を考慮した実装
------------------------------

それでは，浮動小数点エラーを生じないシグモイド関数の実装に戻ります．
ここでは，シグモイド関数の入力が小さすぎる場合や，大きすぎる場合に処理を分けることでエラーを生じないようにします．
シグモイド関数の出力値の範囲を次のような区間に分けて処理することにします．

* :math:`10^{-15}` より小さくなる場合では :math:`10^{-15}` の定数を出力．
* :math:`10^{-15}` 以上 :math:`1 - 10^{-15}` 以下の場合では式 :eq:`eq-lr-sigmoid` のとおりの値を出力．
* :math:`1 - 10^{-15}` より大きくなる場合では :math:`1 - 10^{-15}` の定数を出力．

簡単な計算により， ``sigmoid_range = 34.538776394910684`` とすると，入力値が ``-sigmoid_range`` 以上， ``+sigmoid_range`` 以下の範囲であれば式 :eq:`eq-lr-sigmoid` に従って計算し，それ以外では適切な定数を出力すればよいことが分かる．
これを実装すると次のようになります [#]_ ．

.. code-block:: python

    @staticmethod
    def sigmoid(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

それでは，大きな値や小さな値を入力してみます．

.. code-block:: ipython

    In [30]: from lr1 import LogisticRegression
    In [31]: LogisticRegression.sigmoid(1000.)
    Out[31]: 0.999999999999999
    In [32]: LogisticRegression.sigmoid(-1000.)
    Out[32]: 1e-15

今度は，大きな入力に対しては ``1`` よりわずかに小さな数，逆に，小さな入力に対しては ``0`` よりわずかに大きな数が得られるようになりました．
こうして，シグモイド関数で浮動小数点エラーを生じないようにすることができました．

.. [#]

    .. only:: epub or latex

        https://github.com/tkamishima/mlmpy/blob/master/source/lr2.py

    .. only:: html and not epub

        :download:`LogisticRegresshon クラス：lr2.py <../source/lr2.py>`

.. _lr-sigmoid-ufunc:

ユニバーサル関数の作成
----------------------

.. index:: universal function, ufunc

ここでは，シグモイド関数をユニバーサル関数にする方法を紹介します．
:ref:`nbayes1-predict-logjprob` で紹介しましたが， NumPy配列を引数に与えると，その要素ごとに関数を適用した結果を， :attr:`shape` が入力と同じ配列にまとめて返すのがユニバーサル関数です．

前節で作成したシグモイド関数はユニバーサル関数としての機能がありません．
このことを確認してみます．

.. code-block:: ipython

    In [40]: from lr2 import LogisticRegression
    In [41]: x = np.array([ -1.0, 0.0, 1.0 ])
    In [42]: LogisticRegression.sigmoid(x)

    ... omission ...

    ValueError: The truth value of an array with more than one element
    is ambiguous. Use a.any() or a.all()

if文は配列 :obj:`x` の要素を個別に処理しないので，このようにエラーとなってしまいます．

.. index:: vectorize

そこで，通常の関数をユニバーサル関数に変換する :func:`vectorize` があります．

.. function:: np.vectorize(pyfunc, otypes='', doc=None, excluded=None,
    cache=False)

    Define a vectorized function which takes a nested sequence of objects or numpy arrays as inputs and returns a numpy array as output. The vectorized function evaluates *pyfunc* over successive tuples of the input arrays like the python map function, except it uses the broadcasting rules of numpy.

この :func:`vectorize` は，通常の関数を入力すると，その関数を，引数の配列の各要素に適用するユニバーサル関数を返す関数です．
簡単なステップ関数の例を見てみましょう．

.. code-block:: ipython

    In [50]: def step(x):
   ...:     return 0.0 if x < 0.0 else 1.0
   ...:

三項演算子は入力配列の要素を個別に処理しないのでこの関数はユニバーサル関数ではありません．
そこで次のように :func:`vectorize` を用いてユニバーサル関数に変換します．

.. code-block:: ipython

    In [51]: vstep = np.vectorize(step)
    In [52]: x = np.arange(7) - 3
    In [53]: x
    Out[53]: array([-3, -2, -1,  0,  1,  2,  3])
    In [54]: vstep(x)
    Out[54]: array([ 0.,  0.,  0.,  1.,  1.,  1.,  1.])

関数を入力として関数を返す関数は Python のデコレータとして使うことができます．

.. code-block:: python

    @staticmethod
    @np.vectorize
    def sigmoid(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

先ほど定義したシグモイド関数の， ``@staticmethod`` デコレータの下に，関数 :func:`vectorize` を ``@np.vectorize`` のような形式でデコレータとして与えます [#]_ ．
これでユニバーサル関数となったかを確かめてみます．

.. code-block:: ipython

    In [60]: from lr3 import LogisticRegression
    In [61]: x = np.array([-1.0, 0.0, 1.0])
    In [62]: LogisticRegression.sigmoid(x)
    Out[62]: array([ 0.26894142,  0.5       ,  0.73105858])

配列 :obj:`x` の各要素にシグモイド関数を適用した結果を配列として得ることができました．
このようにしてユニバーサル関数を定義することができました．

.. index:: frompyfunc

なお，入力引数が複数の関数をユニバーサル関数にする :func:`frompyfunc` もあります．

.. function:: np..frompyfunc(func, nin, nout)

    Takes an arbitrary Python function and returns a Numpy ufunc.


.. [#]

    .. only:: epub or latex

        https://github.com/tkamishima/mlmpy/blob/master/source/lr3.py

    .. only:: html and not epub

        :download:`LogisticRegresshon クラス：lr3.py <../source/lr3.py>`

.. _lr-sigmoid-utils:

便利な関数を用いた実装
----------------------

ここまで，他の数学関数の実装にも使える汎用的な手法を紹介しました．
さらに，NumPy にはシグモイド関数の実装に使える便利な関数があり，これらを使って実装することもできます．
そうした関数として :func:`pieceswise` と :func:`clip` を紹介します．

piecewise
^^^^^^^^^

.. index:: piecewise

:func:`piecewise` はHuber関数や三角分布・切断分布の密度関数など，入力の範囲ごとに異なる数式でその出力が定義される区分関数を実装するのに便利です．

.. function:: np.piecewise(x, condlist, funclist, *args, **kw)

    Evaluate a piecewise-defined function.

:ref:`_lr-sigmoid-fpcheck` で実装したシグモイド関数は，浮動小数点エラーを防ぐために入力の範囲に応じて出力を変えています．
:func:`piecewise` を用いて実装したシグモイド関数は次のようになります [#]_ ．

.. code-block:: python

    @staticmethod
    def sigmoid(x):
        sig_r = 34.538776394910684
        condlist = [x < -sig_r, (x >= -sig_r) & (x < sig_r), x >= sig_r]
        funclist = [1e-15, lambda a: 1.0 / (1.0 + np.exp(-a)), 1.0 - 1e-15]

        return np.piecewise(x, condlist, funclist)

:func:`piecewise` の，第2引数は区間を定義する条件のリスト [#]_ で，第3引数はそれらの区間ごとの出力のリストを定義します．
条件のリストで :const:`True` になった位置に対応する出力値が :func:`piecewise` の出力になります．
出力リストが条件のリストより一つだけ長い場合は，出力リストの最後はデフォルト値となります．
条件リストが全て :const:`False` であるときに，このデフォルト値が出力されます．

.. [#]

    複数の条件に対して対応する値を出力する関数は他にも :func:`select` などがあります．

    .. function::  numpy.select(condlist, choicelist, default=0)

    Return an array drawn from elements in choicelist, depending on conditions.

    しかし，条件が満たされるかどうかに関わらず，全ての場合の出力値を計算するため，この場合は浮動小数点エラーを生じてしまいます．

.. [#]

    条件リスト中で ``and`` や ``or`` を使うと，これらはユニバーサル関数ではないため，x が配列の場合にうまく動作しません．
    代わりに NumPy の :func:`logical_and` や :func:`logical_or` を使うこともできます．

.. index:: clip

clip
^^^^

:func:`clip` は，区間の最大値大きい入力はその最大値に，逆に最小値より小さい入力はその最小値にする関数です [#]_ ．
シグモイド関数はこの :func:`clip` を用いると容易に実装できます．

.. code-block:: python

    @staticmethod
    def sigmoid(x):
        # restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)

        return 1.0 / (1.0 + np.exp(-x))

ユニバーサル関数であるため，特に :func:`vectorize` を用いる必要もありません．
以後は，この実装を用います．

.. [#]

    最大値か最小値の一方だけで良い場合はそれぞれ :func:`min` や :func:`max` を用います．

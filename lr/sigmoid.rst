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
モジュール :mod:`lr1a` [#]_ 中のロジスティック回帰クラスの定義のうち，ここでは関数 :meth:`sigmoid` の部分のみを示します．

.. index:: sample; lr1a.py

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

    In [10]: from lr1a import LogisticRegression
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
    lr1a.py:62: RuntimeWarning: overflow encountered in exp
      return 1.0 / (1.0 + np.exp(-x))
    Out[21]: 0.0

シグモイド関数は 1.0 や 0.0 といった値になることは，式 :eq:`eq-lr-sigmoid` の定義からはありえません．
しかし， NumPy での実数演算は，有限精度の浮動小数点を用いて行っているため，絶対値が大きすぎるオーバーフローや，小さすぎるアンダーフローといった浮動小数点エラーを生じ，意図したとおりの計算結果を得ることができません．
そのため，浮動小数点演算の制限を意識してプログラミングする必要があります．

.. _lr-sigmoid-errhandling:

浮動小数点エラーの処理
----------------------

意図した計算結果を得ることができないこの問題の他に，オーバーフローが生じていることの警告メッセージが表示されてしまう問題も生じています．
もちろん，この警告メッセージは意図した結果が得られていないことを知るために役立つものです．
しかし，浮動小数点エラーを，無視してかまわない場合や，例外として処理したい場合など，警告メッセージ表示以外の動作が望ましい場合もあります．
このような場合には，次の :func:`np.seterr` を用いて，浮動小数点演算のエラーに対する挙動を変更できます．

.. index:: seterr

.. function:: np.seterr(all=None, divide=None, over=None, under=None, invalid=None)[source]

    Set how floating-point errors are handled.

:obj:`divide` は0で割ったときの0除算，:obj:`over` は計算結果の絶対値が大きすぎる場合のオーバーフロー，:obj:`under` は逆に小さすぎる場合のアンダーフロー，そして :obj:`invalid` は対数の引数が負数であるなど不正値の場合です．
:obj:`all` はこれら全ての場合についてまとめて挙動を変更するときに用います．

そして，``np.seterr(all=`ignore`)`` のように，キーワード引数の形式で下記の値を設定することで挙動を変更します．

* :const:`warn`: 警告メッセージを表示するデフォルトの挙動です．
* :const:`ignore`: 数値演算エラーを無視します．
* :const:`raise`: 例外 :exc:`FloatingPointError` を送出します．

その他 :const:`call` ， :const:`print` ，および :const:`log` の値を設定できます．

.. [#]

    .. only:: epub or latex

        https://github.com/tkamishima/mlmpy/blob/master/source/lr1a.py

    .. only:: html and not epub

        :download:`LogisticRegresshon1a クラス：lr1a.py <../source/lr1a.py>`

.. index:: e, pi, sp.constants

.. [#]

    NumPy には，このネピアの数を表す :const:`np.e` の他に，円周率を表す :const:`np.pi` の定数があります．
    SciPy の :mod:`sp.consants` モジュール内には，光速や重力定数などの物理定数が定義されています．

.. _lr-sigmoid-icheck:

入力を検査した実装
------------------

isfinite

.. _lr-sigmoid-ufunc:

ユニバーサル関数の作成
----------------------

ufunc デコレータ


.. _lr-sigmoid-utils:

便利な関数を用いた実装
----------------------

clip
piecewise

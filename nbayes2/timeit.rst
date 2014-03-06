.. index:: ipython

.. _nbayes2-timeit:

実行速度の比較
==============

便利な実行環境である ``ipython`` を用いて，二つのクラス :class:`NaiveBayes1` と :class:`NaiveBayes2` の訓練の実行速度を比較します．
そのために， ipython を起動し，訓練に必要なデータを読み込みます．

.. code-block:: ipython

    In [10]: data = np.genfromtxt('vote_filled.tsv', dtype=np.int)
    In [11]: X = data[:, :-1]
    In [12]: y = data[:, -1]

次に，クラスを読み込み，単純ベイズ分類器を実装した二つクラス :class:`NaiveBayes1` と :class:`NaiveBayes2` のインスタンスを生成します．

.. code-block:: ipython

    In [13]: from nbayes2 import *
    In [14]: clr1 = NaiveBayes1()
    In [15]: clr2 = NaiveBayes2()

.. index:: timeit

最後に， ``%timeit`` コマンドを使って，訓練メソッド :meth:`fit` の実行速度を測ります．

.. code-block:: ipython

    In [10]: %timeit clr1.fit(X, y)
    100 loops, best of 3: 16.2 ms per loop

    In [11]: %timeit clr2.fit(X, y)
    1000 loops, best of 3: 499 us per loop

実行速度を見ると :class:`NaiveBayes1` の16.2ミリ秒に対し， :class:`NaiveBayes2` では499マイクロ秒と，後者が 32.5 倍も高速です．
:obj:`for` ループを用いた実装では Python のインタプリタ内で実行されるのに対し， NumPy で配列の演算を用いて実装すると，ほとんどがネイティブコードで実行されるため非常に高速になります．
このようにブロードキャストを活用した実装は，コードが簡潔になるだけでなく，実行速度の面でも有利になります．

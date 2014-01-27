.. index:: broadcasting

.. _nbayes2-distclass:

クラスの分布の学習
==================

前節で説明したブロードキャストの機能を用いると， :obj:`for` ループを用いなくても，複数の要素に対する演算をまとめて行うことができます．
この節では，その方法を，単純ベイズの学習でのクラスの分布の計算の実装を通じて説明します．
例として， :ref:`nbayes2-fit2-fitif` 節で紹介した，クラス分布の計算の比較演算を用いた次の実装を，ブロードキャスト機能を用いた実装に書き換えます．

.. code-block:: python

    nY = np.zeros(n_classes, dtype=np.int)
    for yi in xrange(n_classes):
        for i in xrange(n_samples):
            if y[i] == yi:
                nY[yi] += 1

.. _nbayes2-distclass-general:

書き換えの一般的な手順
----------------------

:obj:`for` ループによる実装をブロードキャストを用いて書き換える手順について，多くの人が利用している方針は見当たりません．
そこで，ここでは著者が採用している手順を紹介します．

1. 出力配列の次元数を :obj:`for` ループの数とします．
2. 各 :obj:`for` ループのループ変数ごとに，出力配列の次元を割り当てます．
3. 各ループ変数に割り当てた次元の要素は，そのループ変数がループ中でとりうる全ての値であり，その他の次元の大きさは 1 であるような配列を生成します．
4. 計算に必要な配列の生成します．
5. 冗長な配列を整理統合します．
6. 要素ごとの演算をユニバーサル関数の機能を用いて実行します．
7. :func:`np.sum` などの集約演算を適用して，最終結果を得ます．

それでは，上記のコードを例として，これらの手順を具体的に説明します．

.. _nbayes2-distclass-assign:

ループ変数の次元への割り当て
----------------------------

手順の段階1と2により，各ループを次元に割り当てます．
例題のコードでは， :obj:`for` ループは2重なので，出力配列の次元数を 2 とします．
ループ変数は外側の :obj:`yi` と内側の :obj:`i` の二つで，これらに次元を一つずつ割り当てます．
ここでは，第 0 次元に :obj:`i` を，第 1 次元に :obj:`yi` を割り当てておきます．
表にまとめると次のようになります．

.. csv-table::
    :header-rows: 1

    次元, ループ変数, 大きさ, 意味
    0, :obj:`i` , :obj:`n_samples` , 事例
    1, :obj:`yi` , :obj:`n_classes` , クラス

.. _nbayes2-distclass-indexgen:

ループ変数に対応する配列の生成
------------------------------

手順の段階3により，各ループ変数がループ内で取り得る全ての値を要素に含む配列を生成します．
これらの要素は，段階2で割り当てた次元に格納します．
まずループ変数 :obj:`i` に関するループを見ます．

.. code-block:: python

    for i in xrange(n_samples):

このループでループ変数 :obj:`i` は ``0`` から ``n_samples - 1`` までの整数をとります．
これらの値を含む配列は ``np.arange(n_samples)`` により生成できます．
次に，これらの値が，ループ変数 :obj:`i` に割り当てた次元 0 の要素になり，他の次元の大きさは 1 になるようにします．
これは， :ref:`nbayes2-shape` で紹介した :attr:`shape` の操作技法を用いて次のように実装できます．

.. code-block:: python

    ary_i = np.arange(n_samples)[:, np.newaxis]

第0次元の ``:`` により， ``np.arange(n_samples)`` の内容を第0次元に割り当て，第1次元には ``np.newaxis`` により大きさ 1 の次元を設定します．

ループ変数 :obj:`yi` についての次のループも同様に処理します．

.. code-block:: python

    for yi in xrange(n_classes):

この変数は ``0`` から ``n_classes - 1`` までの整数をとり，第1次元に割り当てられているので，この変数に対応する配列は次のようになります．

.. code-block:: python

    ary_yi = np.arange(n_classes)[np.newaxis, :]

第0次元には大きさ 1 の次元を設定し，第1次元の要素には ``np.arange(n_classes)`` の内容を割り当てています．

.. _nbayes2-distclass-arygen:

計算に必要な配列の生成
----------------------

段階4では要素ごとの演算に必要な配列を生成します
:obj:`for` ループ内で行われている配列の要素間演算は次の比較演算です．

.. code-block:: python

    y[i] == yi

右辺はループ変数 :obj:`i` で指定された位置の，配列 :obj:`y` の値です．

.. code-block:: python

    ary_y = y[ary_i]

このコードにより :obj:`ary_i` と同じ :attr:`shape` で，その要素が ``y[i]`` であるような配列を得ることができます．
左辺はループ変数 :obj:`yi` のみなので，対応する配列 :obj:``ary_yi`` がそのまま利用できます．
以上で，比較演算に必要な配列 :obj:`ary_y` と :obj:`ary_yi` が得られました．

.. _nbayes2-distclass-redundant:

冗長な配列の整理
----------------

段階5では，冗長な配列の生成を整理します．
:obj:`ary_y` は， :obj:`ary_i` を展開すると次のようになります．

.. code-block:: python

    ary_y = y[np.arange(n_samples)[:, np.newaxis]]

配列の :attr:`shape` を変えてから :obj:`y` 中の値を取り出す代わりに，先に :obj:`y` の値を取り出してから :attr:`shape` を変更するようにすると次のようになります．

.. code-block:: python

    ary_y = (y[np.arange(n_samples)])[:, np.newaxis]

ここで :obj:`y` の大きさは :obj:`n_samples` であることから， ``y[np.arange(n_samples)]`` は :obj:`y` そのものになります．
すると :obj:`ary_y` はさらに簡潔に生成できます．

.. code-block:: python

    ary_y = y[:, np.newaxis]

以上のことから， :obj:`ary_i` を生成することなく目的の :obj:`ary_y` を生成できるようになりました．

この冗長なコードの削除は次のループの書き換えと対応付けることができます．
次のループ変数 :obj:`i` を使って :obj:`y` 中の要素を取り出すコード

.. code-block:: python

    for i in xrange(n_samples):
        val_y = y[i]

は， :obj:`for` ループで :obj:`y` の要素を順に参照する次のコードと同じ :obj:`val_y` の値を得ることができます．

.. code-block:: python

    for val_y in y:
        pass

これらのコードは，それぞれ，ループ変数配列を用いた ``y[ary_i]`` と :obj:`y` の値を直接参照する ``y[:, np.newaxis]`` とに対応します．

.. _nbayes2-distclass-elementwise:

要素ごとの演算と集約演算
------------------------

段階6では要素ごとの演算を行います．
元の実装では要素ごとの演算は ``y[i] == yi`` の比較演算だけでした．
この比較演算を，全ての :obj:`i` と :obj:`yi` について実行した結果をまとめた配列は次のコードで計算できます．

.. code-block:: python

    cmp_y = (ary_y == ary_yi)

:obj:`ary_y` と :obj:`ary_yi` の :attr:`shape` はそれぞれ ``(n_samples, 1)`` と ``(1, n_classes)`` で一致していません．
しかし，ブロードキャストの機能により， ``ary_y[:, 0]`` の内容と， ``ary_yi[0, :]`` の内容が，繰り返して比較演算利用されるため，明示的に繰り返しを記述しなくても目的の結果が得られます．

.. index:: aggregation

最後の段階は集約演算です．
集約 (aggregation) とは，複数の値の代表値，例えば総和，平均，最大などを求めることです．
:ref:`nbayes2-fit2-fitif-ufunc` で述べたように，比較結果が真である組み合わせは :func:`np.sum` によって計算できます．
しかし，単純に ``np.sum(cmp_y)`` とすると配列全体についての総和になってしまいますが，計算したい値は :obj:`yi` がそれぞれの値をとるときの，全ての事例についての和です．
そのために， :func:`np.sum` 関数の :obj:`axis` 引数を指定します．
ここでは，事例に対応するループ変数 :obj:`i` を次元0に割り当てたので， ``axis=0`` と指定します．

.. code-block:: python

    nY = np.sum(cmp_y, axis=0)

以上の実装をまとめて書くと次のようになります．

.. code-block:: python

    ary_y = y[:, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, :]
    cmp_y = (ary_y == ary_yi)
    nY = np.sum(cmp_y, axis=0)

途中で，変数への代入をしないようにすると次の1行のコードで同じ結果が得られます．

.. code-block:: python

    nY = np.sum(y[:, np.newaxis] == np.arange(n_classes)[np.newaxis, :],
                axis=0)


.. _nbayes2-distclass-prob:

クラスの確率の計算
------------------

:class:`NaiveBayes1` の実装では，各クラスごとの標本数 :obj:`nY` を，総標本数 :obj:`n_samples` で割って，クラスの確率を計算しました．

.. code-block:: python

    self.pY_ = np.empty(n_classes, dtype=np.float)
    for i in xrange(n_classes):
        self.pY_[i] = nY[i] / np.float(n_samples)

この処理も，ユニバーサル関数の機能を使うと次のように簡潔に実装できます．

.. code-block:: python

    self.pY_ = np.true_divide(nY, n_samples)

Python では整数同士の割り算の解は切り捨ての整数になります．
しかし，ここでは実数の解を得たいので :func:`np.true_divide` 関数を用いて，切り捨てではない実数の解を得ます．

.. index:: true_divide

.. function:: true_divide(x1, x2[, out]) = <ufunc 'true_divide'>

    Returns a true division of the inputs, element-wise.

この関数はユニバーサル関数なので， :obj:`nY` の各要素は，それぞれ :obj:`n_samples` で割られます．

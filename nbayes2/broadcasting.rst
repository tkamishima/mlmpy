.. _nbayes2-broadcasting:

ブロードキャスト
================

この節では，NumPy 配列の次元数や大きさを操作する方法と，NumPy の強力な機能であるブロードキャストを紹介します．

.. _nbayes2-broadcasting-shape:

.. index:: ! ndarray; shape, ! ndarray; ndim

NumPy 配列の次元数や大きさの操作
--------------------------------

ブロードキャストを紹介する前に， :ref:`nbayes1-ndarray` で紹介した，NumPy の配列クラス :class:`np.ndarray` の属性 :attr:`ndim` と :attr:`shape` を操作する方法を紹介します．

:attr:`ndim` は，配列の次元数を表す属性で，ベクトルでは 1 に，配列では 2 になります．
:attr:`shape` は，スカラーや，タプルによって配列の各次元の大きさを表す属性です．
例えば，大きさが 5 のベクトルはスカラー ``5`` によって， :math:`2 \times 3` の行列はタプル ``(2, 3)`` となります．

.. index:: transpose

次元数を操作する必要がある例として配列の転置の例を紹介します．
転置した配列を得るには，属性 :attr:`T` か，メソッド :meth:`transpose` を用います．
2次元の配列である行列を転置してみましょう：

.. code-block:: ipython

    In [10]: a = np.array([[1, 3], [2, 1]])
    In [11]: a
    Out[11]:
    array([[1, 3],
           [2, 1]])
    In [12]: a.T
    Out[12]:
    array([[1, 2],
           [3, 1]])
    In [13]: a.transpose()
    Out[13]:
    array([[1, 2],
           [3, 1]])

今度は，1次元配列であるベクトルを転置してみます：

.. code-block:: ipython

    In [14]: b = np.array([10, 20])
    In [15]: b
    Out[15]: array([10, 20])
    In [16]: b.T
    Out[16]: array([10, 20])

転置しても，縦ベクトルになることはありません．属性 :attr:`T` やメソッド :meth:`transpose` は，次元数 :attr:`ndim` が 1 以下であれば，元と同じ配列を返します．

.. index:: np.newaxis, newaxis

縦ベクトルを得るには次元数や大きさを，転置する前に操作しておく必要があります．
それには，定数 :const:`np.newaxis` を使います [1]_ [2]_ ．
:const:`np.newaxis` は，添え字指定の表記の中に用います．
元の配列の大きさを維持する次元には ``:`` を指定し，新たに大きさが 1 の次元を追加するところには :const:`np.newaxis` を指定します．

.. code-block:: ipython

    In [17]: b
    Out[17]: array([10, 20])
    In [18]: b.ndim
    Out[18]: 1
    In [19]: b.shape
    Out[19]: (2,)

    In [20]: c = b[:, np.newaxis]
    In [21]: c
    Out[21]:
    array([[10],
           [20]])
    In [22]: c.ndim
    Out[22]: 2
    In [23]: c.shape
    Out[23]: (2, 1)

    In [24]: d = b[np.newaxis, :]
    In [25]: d
    Out[25]: array([[10, 20]])
    In [26]: d.ndim
    Out[26]: 2
    In [27]: d.shape
    Out[27]: (1, 2)

この例で，元の :obj:`b` の :attr:`ndim` は 1 で，その大きさは 2 です．
20行目では，第0次元 [3]_ は元のベクトルをコピーし，第1次元には大きさ 1 の新たな次元を追加しています．
その結果， :obj:`c` の :attr:`shape` は ``(2, 1)`` となり， :math:`2 \times 1` 行列，すなわち縦ベクトルになっています．
24行目では，第0次元の方に新たな次元を追加し，第1次元は元ベクトルをコピーしており，その結果，配列 :obj:`d` の :attr:`shape` は ``(1, 2)`` となります．
これは， :math:`1 \times 2` 行列，すなわち横ベクトルとなっています．

これら縦ベクトル :obj:`c` と横ベクトル :obj:`d` はそれぞれ2次元の配列，すなわち行列なので，次のように転置することができます．

.. code-block:: ipython

    In [28]: c.T
    Out[28]: array([[10, 20]])
    In [29]: d.T
    Out[29]:
    array([[10],
           [20]])

転置により，縦ベクトル :obj:`c` は横ベクトルに，横ベクトル :obj:`d` は縦ベクトルになっています．

:const:`np.newaxis` は，2次元以上の配列にも適用できます．

.. code-block:: ipython

    In [30]: e = np.array([[1, 2, 3], [2, 4, 6]])
    In [31]: e.shape
    Out[31]: (2, 3)
    In [32]: e[np.newaxis, :, :].shape
    Out[32]: (1, 2, 3)
    In [33]: e[:, np.newaxis, :].shape
    Out[33]: (2, 1, 3)
    In [34]: e[:, :, np.newaxis].shape
    Out[34]: (2, 3, 1)

:const:`np.newaxis` の挿入位置に応じて，大きさ1の新しい次元が :attr:`shape` に加わっていることが分かります．
また，同時に2個以上の新しい次元を追加することも可能です．

.. code-block:: ipython

    In [35]: e[np.newaxis, :, np.newaxis, :].shape
    Out[35]: (1, 2, 1, 3)

.. index:: reshape

ブロードキャストとは関連がありませんが， :attr:`shape` を変更する他の方法として :class:`np.ndarray` の :meth:`reshape` メソッドと，関数 :func:`np.reshape` をここで紹介しておきます．

.. function:: np.reshape(a, newshape)

    Gives a new shape to an array without changing its data.

この関数は，配列 :obj:`a` 全体の要素数はそのままで，その :attr:`shape` を ``newshape`` で指定したものに変更するものです．
同様の働きをする :meth:`reshape` メソッドもあります．

.. code-block:: ipython

    In [35]: np.arange(6)
    Out[35]: array([0, 1, 2, 3, 4, 5])
    In [36]: np.reshape(np.arange(6), (2, 3))
    Out[36]:
    array([[0, 1, 2],
           [3, 4, 5]])
    In [37]: np.arange(6).reshape((3, 2))
    Out[37]:
    array([[0, 1],
           [2, 3],
           [4, 5]])

この例では，6個の要素を含む :attr:`shape` が ``(6,)`` の配列を，それぞれ :func:`np.reshape` 関数で ``(2, 3)`` に， :meth:`reshape` メソッドで  ``(3, 2)`` に :attr:`shape` を変更しています．
ただし， :func:`np.reshape` 関数や， :meth:`reshape` メソッドでは，配列の総要素数を変えるような変更は指定できません．

.. code-block:: ipython

    In [38]: np.arange(6).reshape((3, 3))
    ValueError: total size of new array must be unchanged

この例では，総要素数が6個の配列を，総要素数が9個の :attr:`shape` ``(3, 3)`` を指定したためエラーとなっています．

.. only:: not latex

   .. rubric:: 注釈

.. [1]
   :const:`np.newaxis` の実体は :const:`None` であり， :const:`np.newaxis` の代わりに :const:`None` と書いても全く同じ動作をします．
   ここでは，記述の意味を明確にするために， :const:`np.newaxis` を用います．

.. [2]
   他にも :func:`np.expand_dims` や :func:`np.atleast_3d` などの関数を使う方法もありますが，最も自由度の高い :const:`np.newaxis` を用いる方法を紹介します．

.. [3]
   :attr:`shape` で示されるタプルの一番左側から第 0 次元，第 1 次元，… となります．

.. _nbayes2-broadcasting-broadcasting:

.. index:: ! broadcasting

ブロードキャスト
----------------

それでは，いよいよ本題のブロードキャストの説明に移ります．
ブロードキャストとは， :attr:`ndim` や :attr:`shape` が異なる入力配列の間で，これらの属性を自動的に統一する機能です．
:attr:`ndim` と :attr:`shape` を統一することで，それらの入力配列間で要素ごとの演算が可能になり，その統一された次元数と大きさの出力配列に演算結果を得ることができます．
この機能により，例えば，次元数や大きさの異なる配列 :obj:`a` ， :obj:`b` ，および :obj:`c` があったとき，これらの配列の要素ごとの和を，明示的に変換を指示しなくても ``a + b + c`` の形式で書けるようになります．

`公式サイトのブロードキャストの規則 <http://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting>`_ は次の四つです．

1. 次元数 :attr:`ndim` が最大の入力配列より次元数 :attr:`ndim` が小さい全ての入力配列は， :attr:`shape` の先頭に 1 を加えて次元数を揃える．
2. 出力配列の :attr:`shape` の各次元の大きさは，入力配列のその次元の大きさのうちの最大値になる．
3. 入力配列ある各次元の大きさが出力配列の対応する次元の大きさと一致するか，1 である場合には，その入力は計算で利用可能になる．
4. 入力配列のある次元の大きさが 1 であるとき，その最初の要素の値がその次元の全ての計算で利用される．

これらの規則のうち，第1規則は次元数の統一に関するもので，それ以外は要素間の対応付けに関するものです．
それぞれについて順に説明します．

.. _nbayes2-broadcasting-broadcasting-ndim:

次元数の統一
^^^^^^^^^^^^

第1規則は，出力配列 :attr:`ndim` の値を決定し， :attr:`ndim` に変更のあった入力配列の :attr:`shape` を変更する方法を定めます．
出力配列の :attr:`ndim` は，入力配列のうち最大のものになります．
例えば，0次元のスカラー，1次元のベクトル，そして2次元の行列の三つの入力配列が在った場合，出力配列の :attr:`ndim` はこれらの中で一番大きな 2 となります．

:attr:`ndim` を増やした入力配列では， :attr:`shape` の先頭，すなわち第0次元の位置に，大きさ 1 の次元を，必要な数だけ追加します．
入力配列が，次のスカラー :obj:`a` ，ベクトル :obj:`b` ，そして行列 :obj:`c` の三つであった場合，

.. code-block:: ipython

    In [38]: a = np.array(100)
    In [39]: a
    Out[39]: array(100)
    In [40]: b = np.array([10, 20, 30])
    In [41]: b
    Out[42]: array([10, 20, 30])
    In [43]: c = np.array([[1, 2, 3], [4, 5, 6]])
    In [44]: c
    Out[44]:
    array([[1, 2, 3],
           [4, 5, 6]])

:attr:`ndim` は全て 2 に統一され， :attr:`shape` は次のようになります．

.. csv-table::
    :header-rows: 1

    入力配列, 統一前 :attr:`shape`, 統一後 :attr:`shape`
    :obj:`a`, "``()``", "``(1, 1)``"
    :obj:`b`, "``(3,)``", "``(1, 3)``"
    :obj:`c`, "``(2, 3)``", "``(2, 3)``"

0次元の :obj:`a` では，大きさが全て 1 の :attr:`shape` となります．
1次元の :obj:`b` では，元の大きさ3の第0次元の前に，大きさ 1 の次元が挿入されます．
すなわち，統一後の :attr:`ndim` が 2 の行列になる場合には，ベクトルは横ベクトルとして扱われます．
2次元の :obj:`c` は， :attr:`ndim` は変更されていないので， :attr:`shape` も変更されません．
このように自動的に次元数を統一する機構が備わっていますが，コードが直感的に分からなくなることも多いです．
そのため，実用的には，後述の :ref:`nbayes2-distlearn` の例のように， :const:`np.newaxis` などを用いて明示的に次元数を統一してから利用することをお薦めします．

.. _nbayes1-ndarray:

NumPy 配列の基礎
================

.. index:: ! ndarray

ここでは，NumPy で最も重要なクラスである :class:`np.ndarray` について， :ref:`intro-intro` の方針に従い，最低限必要な予備知識について説明します．

:class:`np.ndarray` は， `N`-d Array すなわち，N次元配列を扱うためのクラスです．
NumPy を使わない場合， Python ではこうしたN次元配列を表現するには，多重のリストが利用されます．
:class:`np.ndarray` と多重リストには以下のような違いがあります．

* 多重リストはリンクでセルを結合した形式でメモリ上に保持されますが， :class:`np.ndarray` は C や Fortran の配列と同様にメモリの連続領域上に保持されます．
  そのため，多重リストは動的に変更可能ですが， :class:`np.ndarray` の形状変更には全体の削除・再生性が必要になります．
* 多重リストはリスト内でその要素の型が異なることが許されますが， :class:`np.ndarray` は，基本的に全て同じ型の要素で構成されていなければなりません．
  多重リストとは異なり， :class:`np.ndarray` は各次元ごとの要素数が等しくなければなりません．すなわち，行ごとに列数が異なるような2次元配列などは扱えません．
* :class:`np.ndarray` は，行や列を対象とした多くの高度な数学的操作を，多重リストより容易かつ高速に適用できます．また，配列中の全要素，もしくは一部の要素に対してまとめて演算や関数を適用することで，高速な処理が可能です．

.. nbayes1-ndarray-generation:

NumPy 配列の生成
----------------

それでは， :class:`np.ndarray` の生成方法を説明します．
N次元配列 :class:`np.ndarray` は，数学の概念で言えば，1次元の場合はベクトルに，2次元の場合は行列に，そして3次元以上の場合はテンソルに該当します．

np.array() 関数による生成
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`np.ndarray` にもコンストラクタはありますが，通常は，次の :func:`np.array` 関数 [#]_ によって生成します．

.. index:: array

.. function:: np.array(object, dtype=None)

   Create an array.

最初の引数 ``object`` には，配列の内容を，array_like という型で与えます．
この array_like という型は，配列を :class:`np.ndarray` の他，（多重）リストや（多重）タプルで表現したものです．
リストの場合は，ネストしていない直線状のリストでベクトルを表します．
行列は，ネストしていないリストで表した行を要素とするリスト，すなわち2重にネストしたリストで表します．
もう一つの引数 ``dtype`` は，配列の要素の型を指定しますが，後ほど :class:`np.ndarray` の属性のところで詳細を述べます．

要素が 1, 2, 3 である長さ 3 のベクトルの例です:

.. code-block:: ipython

   In [1]: a = np.array([1,2,3])
   In [2]: a
   Out[2]: array([1, 2, 3])

タプルを使った表現も可能です:

.. code-block:: ipython

   In [3]: a = np.array((10,20,30))
   In [4]: a
   Out[4]: array([10, 20, 30])

2重にネストしたリストで表した配列の例です:

.. code-block:: ipython

   In [5]: a = np.array([[1.5, 0], [0, 3.0]])
   In [6]: a
   Out[6]: 
   array([[ 1.5,  0. ],
          [ 0. ,  3. ]])

リストの要素に :class:`np.ndarray` やタプルを含むことも可能です:

.. code-block:: ipython

   In [7]: a = np.array([1.0, 2.0, 3.0])
   In [8]: b = np.array([a, (10, 20, 30)])
   In [9]: b
   Out[9]: 
   array([[  1.,   2.,   3.],
          [ 10.,  20.,  30.]])

その他の関数による生成
^^^^^^^^^^^^^^^^^^^^^^

:class:`np.ndarray` を作るための関数は非常にたくさんありますが，よく使われるものを簡単に紹介しておきます．

:func:`np.zeros` と :func:`np.ones` は，それぞれ要素が全て0である0行列と，全て1である1行列を生成する関数です．

.. index:: zeros

.. function:: np.zeros(shape, dtype=None)

   Return a new array of given shape and type, filled with zeros.

.. index:: ones

.. function:: np.ones(shape, dtype=None)

   Return a new array of given shape and type, filled with ones.

``shape`` は，スカラーや，タプルによって配列の各次元の長さを表したものです．
長さが 5 のベクトルはスカラー 5 によって，2行3列の行列はタプル (2, 3) によって表現します．

長さが3の0ベクトルの例です:

.. code-block:: ipython

   In [10]: np.zeros(3)
   Out[10]: array([ 0.,  0.,  0.])

3行4列の1行列の例です．引数をタプルにすることを忘れないようにして下さい:

.. code-block:: ipython

   In [11]: np.ones((3,4))
   Out[11]: 
   array([[ 1.,  1.,  1.,  1.],
          [ 1.,  1.,  1.,  1.],
          [ 1.,  1.,  1.,  1.]])

配列を生成した後，その内容をすぐ後で書き換える場合には，配列の要素全てに 0 や 1 を代入すると，無駄な計算をすることになります．
そこで，0 や 1 ではなく，要素の不定値のまま指定した大きさの配列関数 :func:`np.empty` が用意されています．

.. index:: empty

.. function:: np.empty(shape, dtype=None)

   Return a new array of given shape and type, without initializing entries.

:func:`np.zeros` ，:func:`np.ones` ，および :func:`np.empty` には，それぞれ今までに生成した配列と同じ大きさの配列を生成する関数 :func:`np.zeros_like` ，:func:`np.ones_like` ，および :func:`np.empty_like` があります．

.. index:: zeros_like

.. function:: np.zeros_like(a, dtype=None)

   Return an array of zeros with the same shape and type as a given array.

.. index:: ones_like

.. function:: np.ones_like(a, dtype=None)

   Return an array of ones with the same shape and type as a given array.

.. index:: empty_like

.. function:: np.empty_like(a, dtype=None)

   Return a new array with the same shape and type as a given array.

この例では， :math:`2\times3` の行列 :data:`a` と同じ大きさの0行列を生成します:

.. code-block:: ipython

   In [18]: a = np.array([[1,2,3], [2,3,4]])
   In [19]: np.zeros_like(a)
   Out[19]: 
   array([[0, 0, 0],
          [0, 0, 0]])

最後に，最も基本的な行列である単位行列を生成する関数 :func:`np.identity` 

.. index:: identity

.. function:: np.identity(n, dtype=None)

   Return the identity array.

``n`` は行列の大きさを表します．
例えば，4 と指定すると，単位行列は正方行列なので，大きさ :math:`4 \times 4` の行列を指定したことになります．

.. code-block:: ipython

   In [20]: np.identity(4)
   Out[20]: 
   array([[ 1.,  0.,  0.,  0.],
          [ 0.,  1.,  0.,  0.],
          [ 0.,  0.,  1.,  0.],
          [ 0.,  0.,  0.,  1.]])

その他，連続した数列を要素とする配列，対角行列，三角行列などを生成するものや，文字列など他の型のデータから配列を生成するものなど多種多様な関数が用意されていますが，これらについては，実装で必要になったときに随時説明します．

.. [#]
   
   関数の引数は他にもありますが，このチュートリアルでは説明上必要なもののみを示します．
   他の引数についてはライブラリのリファレンスマニュアルを参照して下さい．

.. _nbayes1-ndarray-access:

NumPy 配列の属性と要素の参照
----------------------------

ここでは，前節で生成した :class:`np.ndarray` の属性を説明したのち，配列の要素を参照する方法について述べます．

:class:`np.ndarray` には多数の属性がありますが，よく使われるものをまとめました．

.. class:: np.ndarray

   An array object represents a multidimensional, homogeneous array of fixed-size items.
   An associated data-type object describes the format of each element in the array (its byte-order, how many bytes it occupies in memory, whether it is an integer, a floating point number, or something else, etc.)

   :ivar dtype: Data-type of the array's elements
   :ivar ndim: Number of array dimensions
   :ivar shape: Tuple of array dimensions

今までに関数の引数に現れた，最初の属性 :attr:`dtype` は，配列の要素の型を指定します．
:class:`np.ndarray` は，基本的に配列の中の全要素の型は同じです [#]_ ．
二番目の属性 :attr:`ndim` は，次元数を表します．ベクトルでは 1 に，配列では 2 になります．
三番目の属性 :attr:`shape` は，各次元ごとの配列の大きさをまとめたタプルを返します．例えば，長さが 5 のベクトルは (5,) [#]_ となり， :math:`2 \times 3` の大きさの行列では (2, 3) となります．

.. index:: ! dtype

これらの属性のうち :attr:`dtype` について詳しく述べます．
よく使われる型は Python のビルトイン型の真理値型，整数型，浮動小数点型，複素数型に対応する :obj:`np.bool` ， :obj:`np.int` ， :obj:`np.float` ， :obj:`np.complex` です．
メモリのビット数を明示的に表す :obj:`np.int32` や :obj:`np.float64` などもありますが，メモリを特に節約したい場合や，C や Fortran で書いた関数とリンクするといった場合以外はあまり使わないでしょう．

文字列型については，ビルトイン型の :obj:`str` とは，少し異なります．
:class:`np.ndarray` では，要素の大きさが同じである必要があるため，文字列も固定長にする必要があります．
通常の文字列に対応する文字列は，NumPy の型を返す関数 :func:`np.dtype` を用いて， ``np.dtype('S<文字列長>')`` [#]_ のように指定します．
例えば，最大長が16である文字列を扱う場合は ``np.dtype("S16")`` のように指定します．
Unicode文字列の場合は，この ``S`` が ``U`` に置き換わります．

配列の :attr:`dtype` を指定するには，(1) :func:`np.array` などの配列生成関数の ``dtype`` 引数で指定する方法と， (2) :class:`np.ndarray` の :meth:`np.ndarray.astype` メソッドを使う方法とがあります．

まず，(1) の ``dtype`` 引数を指定する方法について述べます．
:func:`np.array` では要素が全て整数の場合は，要素の型は整数になりますが，それを浮動小数点にするには，次のように指定します．

.. code-block:: ipython

   In [1]: a = np.array([1, 2, 3])
   In [2]: a.dtype
   Out[2]: dtype('int64')
   In [3]: a = np.array([1, 2, 3], dtype=np.float)
   In [4]: a.dtype
   Out[4]: dtype('float64')

浮動小数点型の配列を複素数型で作り直す場合は，次のようになります．

.. code-block:: ipython

   In [5]: a = np.array([1.0, 1.5, 2.0])
   In [6]: a.dtype
   Out[6]: dtype('float64')
   In [7]: a = np.array(a, dtype=np.complex)
   In [8]: a.dtype
   Out[8]: dtype('complex128')
   In [9]: a
   Out[9]: array([ 1.0+0.j,  1.5+0.j,  2.0+0.j])

.. index::
   single: ndarray; astype

(2) の :meth:`np.ndarray.astype` も同様に利用できます．

.. code-block:: ipython

   In [10]: a = np.array([1, 2, 3])
   In [11]: a.dtype
   Out[11]: dtype('int64')
   In [12]: a = a.astype(np.float)
   In [13]: a.dtype
   Out[13]: dtype('float64')
   In [14]: a
   Out[14]: array([ 1.,  2.,  3.])

次は :class:`np.ndarray` の要素の参照方法について述べます．
非常に多様な要素の参照方法があるため，最も基本的な方法のみを述べ，他の方法については順次紹介することにします．
最も基本的な要素の参照方法とは，各次元ごとに何番目の要素を参照するかを指定します．
1次元配列であるベクトル :obj:`a` の要素 3 を ``a[3]`` 参照すると，次のような結果が得られます．

.. code-block:: ipython

   In [15]: a = np.array([1, 2, 3, 4, 5], dtype=float)
   In [16]: a[3]
   Out[16]: 4.0

ここで注意すべきは，添え字の範囲は，数学の規則である :math:`1,\ldots,5` ではなく，Python の規則に従って :math:`0,\ldots,4` となることです．
``a.shape[0]`` とすると，第1次元の要素の長さ，すなわちベクトルの長さとして 5 が得られますが，添え字の範囲はそれより 1 小さな 4 までとなります．
同様に， :math:`2 \times 3` の行列では，行は :math:`0,\ldots,1` の範囲で，列は :math:`0,\ldots,2` の範囲で指定します．

.. code-block:: ipython

   In [17]: a = np.array([[11, 12, 13], [21, 22, 23]])
   In [18]: a[1,2]
   Out[18]: 23
   In [19]: a.shape
   Out[19]: (2, 3)

最後に， :class:`np.ndarray` の1次元と2次元の配列と，数学の概念であるベクトルと行列との関係について補足します．
線形代数では，縦ベクトルや横ベクトルという区別がありますが，1次元の :class:`np.ndarray` 配列にはそのような区別はありません．
そのため，1次元配列を転置することができず，厳密には数学でいうところのベクトルとは厳密には異なります．

そこで，縦ベクトルや横ベクトルを区別して表現するには，それぞれ列数が1である2次元の配列と，行数が1である2次元配列を用います．
縦ベクトルは次のようになり:

.. code-block:: ipython

   In [20]: np.array([[1], [2], [3]])
   Out[20]: 
   array([[1],
          [2],
          [3]])

横ベクトルは次のようになります（リストが2重にネストしていることに注意）:

.. code-block:: ipython

   In [21]: np.array([[1, 2, 3]])
   Out[21]: array([[1, 2, 3]])

以上，NumPyの配列 :class:`np.ndarray` について基本的なことを述べました．
ここで紹介した基本事項を使い，NumPy / SciPy の他の機能を，機械学習のアルゴリズムの実装を通じて紹介してゆきます．

.. [#]
   オブジェクトを要素とする型 :obj:`np.object` や，行ごとに同じ構造である制限の下，いろいろな型を混在できる structured array があります．

.. [#]
   Python では， (5) と表記すると，スカラー量 5 を括弧でくくった数式とみなされるため，要素数が1個のタプルは (5,) となります．

.. [#]
   整数型や浮動小数点型にも同様の文字列を用いた指定方法があります．
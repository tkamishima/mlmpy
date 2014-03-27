.. _nbayes1-run:

実行
====

最後に，データをファイルから読み込み，そのデータに対して，実装した :class:`NaiveBayes1` クラスを用いて学習と予測を行います．

最初に， :class:`NaiveBayes1` クラスを :obj:`import` します．

.. code-block:: python

   from nbayes1 import NaiveBayes1

次にファイルからデータを読み込みます．
NumPy と SciPy にはいろいろな形式のファイルを読み込む関数があります [#]_ が，テキスト形式のファイルの読み込みをする :func:`np.genfromtxt` [#]_ を用います．

.. index:: genfromtxt

.. function:: np.genfromtxt(fname, dtype=<type 'float'>, comments='#', delimiter=None)

   Load data from a text file, with missing values handled as specified.

この関数は，カンマ区切り形式や，タブ区切り形式のテキストファイルを読み込み，それを NumPy 配列に格納します．
``fname`` は，読み込むファイルを，ファイル名を示す文字列か， :func:`open` 関数で得たファイルオブジェクトで指定します．
``dtype`` は，関数が返す NumPy 配列の :attr:`dtype` 属性を指定します．
``comments`` で指定した文字が，ファイル中の行の先頭にある場合，その行はコメント行として読み飛ばされます．
``delimiter`` は，列の区切りを指定します．
規定値では，タブを含むホワイトスペースの位置で区切ります．
カンマ区切り csv ファイルの場合は，カンマ ``","`` を区切り文字列として指定します．
区切り文字ではなく，数値や数値のタプルを指定することで，文字数で区切ることもできます．
引数の種類が非常に多い関数なので，ごく一部のみをここでは紹介しました．
その他の機能については `Importing data with genfromtxt <http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_ などを参照して下さい．

:class:`NaiveBayes1` のテスト用データとして， ``vote_filled.tsv`` を用意しました [#]_ ．

.. index:: sample; vote_filled.tsv

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/vote_filled.tsv

.. only:: html and not epub

  :download:`Congressional Voting Records Data Set : vote_filled.tsv <../source/vote_filled.tsv>`

このデータはタブ区切り形式です．
また， :class:`NaiveBayes1` クラスでは，入力訓練データの :attr:`dtype` 属性が整数であることを前提としています．
よって，次のようにファイルを読み込みます．

.. code-block:: python

   data = np.genfromtxt('vote_filled.tsv', dtype=np.int)

このファイルは，最終列がクラスラベル，それ以外に特徴量を格納しています．
このため，変数 :obj:`data` の最終列をクラスラベルの配列 :obj:`y` に，それ以外を特徴量の配列 :obj:`X` に格納します．

.. code-block:: python

   X = data[:, :-1]
   y = data[:, -1]

データが揃ったので，いよいよ :class:`NaiveBayes1` クラスを使うことができます．
設計どおり，コンストラクタで分類器を作り， :meth:`fit` メソッドに訓練データを与えてモデルパラメータを学習させます．

.. code-block:: python

   clr = NaiveBayes1()
   clr.fit(X, y)

テスト用のデータは， :obj:`X` の最初の10個分を再利用します．
予測クラスは，分類器の :meth:`predict` メソッドで得られます．
結果が正しいかどうかを調べるため，元のクラスと予測クラスを表示してみます．

.. code-block:: python

   predict_y = clr.predict(X[:10, :])
   for i in xrange(10):
       print i, y[i], predict_y[i]

結果を見ると，ほぼ正しく予測出来ていますが，6番のデータについては誤って予測しているようです．

実行可能な状態の :class:`NaiveBayes1` の実行スクリプトは，以下より取得できます．
実行時には ``nbayes1.py`` と ``vote_filled.tsv`` がカレントディレクトリに必要です．

.. index:: sample; run_nbayes1.py

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/run_nbayes1.py

.. only:: html and not epub

  :download:`NaiveBayes1 実行スクリプト：run_nbayes1.py <../source/run_nbayes1.py>`

.. only:: not latex

   .. rubric:: 注釈

.. [#]
   代表的な読み込み関数には，バイナリの npy 形式 :func:`np.load` ，matlab 形式 :func:`sp.io.loadmat` ，Weka の arff 形式 :func:`sp.io.loadarff` などがあります．
   ファイルの読み込みについては，Scipy.org にある `Cookbook / InputOutput <http://www.scipy.org/Cookbook/InputOutput>`_ が参考になります．

.. [#]
   :func:`np.loadtxt` という同様の機能をもつ関数もあります．
   :func:`np.genfromtxt` は， :func:`np.loadtxt` の機能に加えて，欠損値処理の機能が加えられているので，こちらを紹介します．

.. [#]
   ``vote_filled.tsv`` は UCI Repository の
   `Congressional Voting Records Data Set <http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records>`_
   をタブ区切り形式にしたファイルです．
   アメリカ議会での16種の議題に対する投票行動を特徴とし，議員が共和党 (0) と民主党 (1) のいずれであるかがクラスです．
   元データには欠損値が含まれていますが，各クラスの最頻値で補完しました．


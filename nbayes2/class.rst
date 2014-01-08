.. _nbayes2-class:

クラスの再編成
==============

.. todo::
   NaiveBayes1 クラスを BaseNaiveBayes のサブクラスに
   - predict メソッドの共有

この章では単純ベイズ法の学習のいろいろな実装を比較するのに便利になるように， :ref:`nbayes1` の :class:`NaiveBayes1` クラスを再編成します．
:class:`NaiveBayes1` クラスには，コンストラクタの他には，学習を行う :meth:`fit` メソッドと，予測を行う :meth:`predict` メソッドがありました．
学習メソッドの実装はいろいろ変えますが， :meth:`predict` メソッドはどの実装でも共通にする予定です．
そこで，予測メソッドなど共通部分含む抽象クラスを新たに作成し，各クラスで異なる部分は，その抽象クラスを継承した下位クラスに実装することにします．

.. _nbayes2-class-abstract:

.. index:: BaseBinaryNaiveBayes

二値単純ベイズの抽象クラス
--------------------------

.. index:: abstract class

二値単純ベイズの共通部分を含む抽象クラス :class:`BaseBinaryNaiveBayes` を作成します．
:mod:`abc` モジュールを利用して次のようにクラスを定義しておきます．

.. code-block:: python

    from abc import ABCMeta, abstractmethod

    class BaseBinaryNaiveBayes(object):
    """
    Abstract Class for Naive Bayes whose classes and features are binary.
    """

    __metaclass__ = ABCMeta

この抽象クラスでは実装しない :meth:`fit` メソッドは抽象メソッドとして次のように定義しておきます．
こう定義しておくことで，この抽象クラスの下位クラスで :meth:`fit` メソッドを定義し忘れると例外が発生するようになります．

.. code-block:: python

    @abstractmethod
    def fit(self, X, y):
        """
        Abstract method for fitting model
        """
        pass

最後に今後の単純ベイズの実装で共通して使われるコンストラクタと :meth:`predict` メソッドを，今までの :class:`NaiveBayes1` からコピーしておきます．
以上で，二値単純ベイズの抽象クラスは完成です．

.. _nbayes2-class-nbayes1:

.. index:: NaiveBayes1

新しい :class:`NaiveBayes1` クラス
----------------------------------

新しい :class:`NaiveBayes1` クラスを，上記の :class:`BaseBinaryNaiveBayes` の下位クラスとして次のように定義します．

.. code-block:: python

    class NaiveBayes1(BaseBinaryNaiveBayes):
        """
        Naive Bayes class (1)
        """

次に，このブクラスのコンストラクタを作成します．
ここでは単に上位クラスのコンストラクタを呼び出すように定義しておきます．

.. code-block:: python

    def __init__(self):
        super(NaiveBayes1, self).__init__()

最後にこのクラスに固有の :meth:`fit` メソッドを，以前の :class:`NaiveBayes1` クラスからコピーしておきます．
以上で， :class:`NaiveBayes1` クラスの再編成が完了しました．

.. _nbayes2-class-run:

実行
----

.. index:: sample; nbayes1b.py

新しい :class:`NaiveBayes1` クラスの実行可能な状態のファイルは，以下より取得できます．

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/nbayes1b.py

.. only:: html and not epub

  :download:`新 NaiveBayes1 クラス：nbayes1b.py <../source/nbayes1b.py>`

実行ファイルも， :class:`NaiveBayes1` クラスを読み込むファイルを変えるだけです．

.. index:: sample; run_nbayes1b.py

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/run_nbayes1b.py

.. only:: html and not epub

  :download:`新 NaiveBayes1 実行スクリプト：run_nbayes1b.py <../source/run_nbayes1b.py>`

データファイル ``vote_filled.tsv`` をカレントディレクトリに置いて実行すると，以前の ``run_nbayes1.py`` と同じ結果が得られます．

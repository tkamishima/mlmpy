.. _nbayes2-class:

クラスの再編成
==============

この章では単純ベイズ法の学習のいろいろな実装を比較するのに便利になるように， :ref:`nbayes1` の :class:`NaiveBayes1` クラスを再編成します．
:class:`NaiveBayes1` クラスには，コンストラクタの他には，学習を行う :meth:`fit` メソッドと，予測を行う :meth:`predict` メソッドがありました．
:meth:`predict` メソッドはどの実装でも共通にする予定ですが，学習メソッドのいろいろな実装をこれから試します．
そこで，予測メソッドなど共通部分含む抽象クラスを新たに作成し，各クラスで異なる学習メソッドは，その抽象クラスを継承した下位クラスに実装することにします．

.. index:: class; BaseBinaryNaiveBayes

.. _nbayes2-class-abstract:

二値単純ベイズの抽象クラス
--------------------------

.. index:: abstract class

二値単純ベイズの共通部分を含む抽象クラス :class:`BaseBinaryNaiveBayes` を作成します．
抽象クラスを作るための :mod:`abc` モジュールを利用して，次のようにクラスを定義しておきます [#]_ ．

.. code-block:: python

    from abc import ABCMeta, abstractmethod

    class BaseBinaryNaiveBayes(object, metaclass=ABCMeta):
    """
    Abstract Class for Naive Bayes whose classes and features are binary.
    """

この抽象クラスでは実装しない :meth:`fit` メソッドは，抽象メソッドとして次のように定義しておきます．
このように定義しておくと，この抽象クラスの下位クラスで :meth:`fit` メソッドが定義されていないときには例外が発生するので，定義し忘れたことが分かるようになります．

.. code-block:: python

    @abstractmethod
    def fit(self, X, y):
        """
        Abstract method for fitting model
        """
        pass

最後に，今後の単純ベイズの実装で共通して使うコンストラクタと :meth:`predict` メソッドを，今までの :class:`NaiveBayes1` からコピーしておきます．
以上で，二値単純ベイズの抽象クラスは完成です．

.. only:: not latex

   .. rubric:: 注釈

.. [#]
    抽象クラスの定義の記述は Python2 では異なっています．
    Python3 と 2 の両方で動作するようにするには :mod:`six` などのモジュールが必要になります．

.. index:: class; NaiveBayes1

.. _nbayes2-class-nbayes1:

新しい :class:`NaiveBayes1` クラス
----------------------------------

新しい :class:`NaiveBayes1` クラスを，上記の :class:`BaseBinaryNaiveBayes` の下位クラスとして次のように定義します．

.. code-block:: python

    class NaiveBayes1(BaseBinaryNaiveBayes):
        """
        Naive Bayes class (1)
        """

次に，このクラスのコンストラクタを作成します．
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

  `新 NaiveBayes1 クラス：nbayes1b.py <https://github.com/tkamishima/mlmpy/blob/master/source/nbayes1b.py>`_

実行ファイルも， :class:`NaiveBayes1` クラスを読み込むファイルを変えるだけです．

.. index:: sample; run_nbayes1b.py

.. only:: epub or latex

  https://github.com/tkamishima/mlmpy/blob/master/source/run_nbayes1b.py

.. only:: html and not epub

  `新 NaiveBayes1 実行スクリプト：run_nbayes1b.py <https://github.com/tkamishima/mlmpy/blob/master/source/run_nbayes1b.py>`_

データファイル ``vote_filled.tsv`` を作業ディレクトリに置いて実行すると，以前の ``run_nbayes1.py``
と同じ結果が得られます．

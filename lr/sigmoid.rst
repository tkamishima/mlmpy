.. _lr-sigmoid:

シグモイド関数
==============

.. index:: sigmoid function

ここでは :ref:`lr-lr` の式(1)で用いる次のシグモイド関数を実装します．

.. math::
    :label: eq-lr-sigmoid

    \mathrm{sig}(a) = \frac{1}{1 + \exp(-a)}

この関数の実装を通じ，数値演算エラーの扱い，ユニバーサル関数の作成方法，数学関数の実装に便利な関数などについて説明します．

.. _lr-sigmoid-simple:
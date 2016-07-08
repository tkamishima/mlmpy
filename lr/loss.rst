.. _lr-loss:

損失関数とその勾配
==================




:func:`np.zeros` ， :func:`np.ones` ，および :func:`np.empty` には，それぞれ今までに生成した配列と同じ大きさの配列を生成する関数 :func:`np.zeros_like` ， :func:`np.ones_like` ，および :func:`np.empty_like` があります．

.. index:: zeros_like

.. function:: np.zeros_like(a, dtype=None)

   Return an array of zeros with the same shape and type as a given array.

.. index:: ones_like

.. function:: np.ones_like(a, dtype=None)

   Return an array of ones with the same shape and type as a given array.

.. index:: empty_like

.. function:: np.empty_like(a, dtype=None)

   Return a new array with the same shape and type as a given array.


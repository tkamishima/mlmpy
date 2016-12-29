.. _lr-optimization:

非線形最適化関数
================

ロジスティック回帰を解くには， :ref:`lr-lr` の式(3)の非線形最適化問題を解く必要があります．
ここでは，この最適化問題を :mod:`scipy.optimize` モジュールに含まれる関数 :func:`minimize` を用いて実装します．
そこで，この節では :func:`minimize` などの最適化関数について俯瞰します．
そして，ロジスティック回帰モデルをあてはめるメソッドを，次の :ref:`lr-fit` で実装します．

.. _lr-optimization-func:

SciPy の非線形最適化関数
------------------------

.. index:: non-linear optimization, optimization

SciPy の非線形最適化関数には， :func:`minimize_scalar` と :func:`minimize` があります．
それぞれを簡単に紹介しておきます．

.. index:: minimize_scalar, Brent method

.. function:: sp.optimize.minimize_scalar(fun, args=(), method='brent')

    Minimization of scalar function of one variable.

:func:`minimize_scalar` は，入力パラメータと出力が共にスカラーである目的関数 ``fun`` の最小値と，そのときのパラメータの値を求めます．
関数 ``fun`` に，最小化するパラメータ以外の引数がある場合には ``args`` で指定します．
最適化の方法は ``method`` で指定します．
通常は，最小化するパラメータの範囲に制約がないときは ``brent`` を，制約がある場合は ``bounded`` を指定します．

.. index:: ! minimize

.. function:: sp.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, options=None)

    Minimization of scalar function of one or more variables.

:func:`minimize` は入力パラメータがベクトルで出力はスカラーである目的関数 ``fun`` の最小値と，そのときのパラメータの値を求めるという非線形計画問題を解きます．
``x0`` で指定したパラメータから解の探索を開始し，そこから到達できる局所解を見つけることができます [#]_ ．
``args`` にはパラメータ以外の関数 ``fun`` への入力，``method`` では次節で紹介する最適化手法を指定します．
以降の引数は，最適化手法によって指定の必要姓やその意味合いが異なります．
``jac`` は目的関数 ``fun`` の勾配ベクトル（ヤコビベクトル）すなわち，目的関数を入力パラメータの各変数での1次導関数を要素とするベクトル返す関数を与えます．
``hess`` や ``hessp`` は2次導関数を要素とするヘシアンを指定します [#]_ ．
制約付きの最適化手法で ``bounds`` や ``constraints`` はパラメータに対する制約条件を指定します．
``tol`` は終了条件の許容誤差で，必要な精度と計算時間のトレードオフを考慮して選びます [#]_ ．

:func:`minimize_scalar` と :func:`minimize` のいずれも，最適化の結果を次のクラスで返す

.. index:: OptimizeResult

.. class:: sp.optimize.OptimizeResult

    Represents the optimization result.

    :ivar fun: Values of objective function.
    :ivar x: The solution of the optimization.
    :ivar success: Whether or not the optimizer exited successfully.
    :ivar nit: Number of iterations performed by the optimizer.

:attr:`fun` と :attr:`x` は，それぞれ関数の最小値と，そのときのパラメータの値です．
:attr:`success` は最適化が成功したかどうか，:attr:`nit` は収束するまでの反復数です．

.. only:: not latex

   .. rubric:: 注釈

.. [#]

    .. index:: brute, basinhopping

    局所最適解を異なる初期値から探索することを何度も繰り返して大域最適解を求める関数として :func:`sp.optimize.basinhopping` や :func:`sp.optimize.brute` が用意されています．

.. [#]

    ``hess`` は通常のヘシアン，すなわち :math:`f(\mathbf{x})` の2次導関数が特定の値 :math:`\mathbf{a}` をとったときの行列 :math:`\mathbf{H}(\mathbf{a}) = {\left[ \frac{\partial^2 f}{\partial x_i \partial x_j} \right]}_{ij}\bigg|_{\mathbf{x}=\mathbf{a}}` を指定します．
    しかし，パラメータベクトル :math:`\mathbf{x}` の次元数が大きいときは，ヘシアンを保持するためには次元数の2乗という多くのメモリを必要としてしまいます．
    そのような場合に， ``hessp`` はヘシアンと特定のベクトル :math:`\mathbf{p}` との積 :math:`\mathbf{H}(\mathbf{a})\mathbf{p}` を計算する関数を指定することでメモリを節約することができます．

.. [#]

    非常に小さな値を指定すると，浮動小数点のまるめ誤差などの影響で最適化関数が停止しない場合があります．
    :math:`10^{-6}` より小さな値を指定するときは，このことを念頭においた方がよいでしょう．

.. _lr-optimization-methods:

各種の最適化手法
----------------

実装に移る前に， :func:`minimize` の ``method`` で指定できる最適化手法を一通り見ておきます．
最適化手法には，パラメータに制約がない場合とある場合に用いるものとがあります．

パラメータに制約がない手法には次のものがあります

1. 勾配ベクトルやヘシアンが不要
    * ``Nelder-Mead`` ：Nelder-Mead法
    * ``Powell`` ：Powell法
2. 勾配ベクトルのみが必要
    * ``CG`` ：共役勾配法 (conjugate gradient method)
    * ``BFGS`` ：BFGS法 (Broyden–Fletcher–Goldfarb–Shanno method)
3. 勾配ベクトルとヘシアンの両方が必要
    * ``Newton-CG`` ：ニュートン共役勾配法 (Newton conjugate gradient method)
    * ``trust-ncg`` ：信頼領域ニュートン共役勾配法 (Newton conjugate gradient trust-region method)
    * ``dogleg`` ：信頼領域dog-leg法 (dog-leg trust-region method)

1 から 3 になるにつれ，勾配やヘシアンなど引数として与える関数は増えますが，収束するまでの反復数は減ります．
1 の ``Nelder-Mead`` と ``Powell`` では，ほとんどの場合でPowell法が高速です．
大まかにいって，勾配を使う方法と比べて，1回の反復で必要になる目的関数の評価階数はパラメータ数倍になるため，これらの方法は遅いです．
勾配を解析的に計算出来ない場合にのみ使うべきでしょう．

勾配ベクトルのみを使う方法のうち， ``BFGS`` は近似計算したヘシアンを用いるニュートン法であるので，収束は ``CG`` に比べて速いです．
しかし，ヘシアンの大きさはパラメータ数の2乗であるため，パラメータ数が多いときには多くのメモリと計算量が必要となるため， ``CG`` の方が速くなることが多いです．

3 の方法はヘシアンも必要なので，ヘシアンの実装の手間や，その計算に必要な計算量やメモリを考慮して採用してください．

パラメータに制約のある方法には次のものがあります．

1. パラメータの範囲に制約がある場合
    * ``L-BFGS-B`` ：範囲制約付きメモリ制限BFGS法
    * ``TNC`` ：切断ニュートン共役勾配法
2. パラメータの範囲の制約に加えて，等式・不等式制約がある場合
    * ``COBYLA`` ：COBYLA法 (constrained optimization by linear approximation method)
    * ``SLSQP`` ：sequential least squares programming

パラメータの範囲は ``bounds`` に，パラメータそれぞれの値の最小値と最大値の対の系列を指定します．
等式・不等式制約は， ``type`` ， ``fun`` ， ``jac`` の要素を含む辞書の系列で指定します．
``type`` には，等式制約なら文字列定数 ``eq`` を，不等式制約なら ``ineq`` を指定します．
``fun`` には制約式の関数を， ``jac`` にはその勾配を指定します．

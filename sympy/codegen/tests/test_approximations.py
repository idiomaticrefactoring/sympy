import math
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.codegen.rewriting import optimize
from sympy.codegen.approximations import SumApprox, SeriesApprox


def test_SumApprox_trivial():
    x = symbols('x')
    expr1 , sum_approx  = 1 + x, SumApprox(bounds={x: (-1e-20, 1e-20)}, reltol=1e-16)
    apx1 = optimize(expr1, [sum_approx])
    assert not apx1 - 1


def test_SumApprox_monotone_terms():
    x, y, z = symbols('x y z')
    expr1 , bnds1  = exp(z) * (x ** 2 + y ** 2 + 1), {x: (0, 0.001), y: (100, 1000)}
    sum_approx_m2 , sum_approx_m5 , sum_approx_m11  = SumApprox(bounds=bnds1, reltol=0.01), SumApprox(bounds=bnds1, reltol=1e-05), SumApprox(bounds=bnds1, reltol=1e-11)
    assert not (optimize(expr1, [sum_approx_m2]) / exp(z) - y ** 2).simplify()
    assert not (optimize(expr1, [sum_approx_m5]) / exp(z) - (y ** 2 + 1)).simplify()
    assert not (optimize(expr1, [sum_approx_m11]) / exp(z) - (y ** 2 + 1 + x ** 2)).simplify()


def test_SeriesApprox_trivial():
    x, z = symbols('x z')
    for factor in [1, exp(z)]:
        x = symbols('x')
        expr1 , bnds1  = exp(x) * factor, {x: (-1, 1)}
        series_approx_50 , series_approx_10 , series_approx_05 , c  = SeriesApprox(bounds=bnds1, reltol=0.5), SeriesApprox(bounds=bnds1, reltol=0.1), SeriesApprox(bounds=bnds1, reltol=0.05), (bnds1[x][1] + bnds1[x][0]) / 2
        f0 = math.exp(c)  # 1.0

        ref_50 , ref_10 , ref_05 , res_50 , res_10 , res_05  = f0 + x + x ** 2 / 2, f0 + x + x ** 2 / 2 + x ** 3 / 6, f0 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24, optimize(expr1, [series_approx_50]), optimize(expr1, [series_approx_10]), optimize(expr1, [series_approx_05])


        assert not (res_50 / factor - ref_50).simplify()
        assert not (res_10 / factor - ref_10).simplify()
        assert not (res_05 / factor - ref_05).simplify()

        max_ord3 = SeriesApprox(bounds=bnds1, reltol=0.05, max_order=3)
        assert optimize(expr1, [max_ord3]) == expr1

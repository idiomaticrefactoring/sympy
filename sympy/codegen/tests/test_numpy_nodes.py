from itertools import product
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.printing.repr import srepr
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2

x, y, z = symbols('x y z')

def test_logaddexp():
    lae_xy , ref_xy  = logaddexp(x, y), log(exp(x) + exp(y))
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert not (lae_xy.diff(wrt, deriv_order) - ref_xy.diff(wrt, deriv_order)).rewrite(log).simplify()

    one_third_e = 1*exp(1)/3
    two_thirds_e = 2*exp(1)/3
    logThirdE , logTwoThirdsE  = log(one_third_e), log(two_thirds_e)
    lae_sum_to_e = logaddexp(logThirdE, logTwoThirdsE)
    assert lae_sum_to_e.rewrite(log) == 1
    assert lae_sum_to_e.simplify() == 1
    was = logaddexp(2, 3)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with 2, 3


def test_logaddexp2():
    lae2_xy , ref2_xy  = logaddexp2(x, y), log(2 ** x + 2 ** y) / log(2)
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert not (lae2_xy.diff(wrt, deriv_order) - ref2_xy.diff(wrt, deriv_order)).rewrite(log).cancel()

    def lb(x):
        return log(x)/log(2)

    two_thirds = S.One*2/3
    four_thirds = 2*two_thirds
    lbTwoThirds , lbFourThirds  = lb(two_thirds), lb(four_thirds)
    lae2_sum_to_2 = logaddexp2(lbTwoThirds, lbFourThirds)
    assert lae2_sum_to_2.rewrite(log) == 1
    assert lae2_sum_to_2.simplify() == 1
    was = logaddexp2(x, y)
    assert srepr(was) == srepr(was.simplify())  # cannot simplify with x, y

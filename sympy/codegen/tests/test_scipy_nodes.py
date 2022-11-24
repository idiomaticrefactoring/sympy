from itertools import product
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import cos
from sympy.core.numbers import pi
from sympy.codegen.scipy_nodes import cosm1

x, y, z = symbols('x y z')


def test_cosm1():
    cm1_xy , ref_xy  = cosm1(x * y), cos(x * y) - 1
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert not (cm1_xy.diff(wrt, deriv_order) - ref_xy.diff(wrt, deriv_order)).rewrite(cos).simplify()

    expr_minus2 = cosm1(pi)
    assert expr_minus2.rewrite(cos) == -2
    assert cosm1(3.14).simplify() == cosm1(3.14)  # cannot simplify with 3.14

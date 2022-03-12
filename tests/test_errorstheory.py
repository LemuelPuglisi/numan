import numpy as np

from numan import errorstheory as e


def test_absolute_error():
    assert e.absolute_error(992, 1001) == 9


def test_relative_error():
    assert e.relative_error(12500, 13000) == 0.04


def test_p_correct_decimals():
    x = 624.428731
    y = 624.428711
    assert e.p_correct_decimals(x, y, 4)


def test_machine_number_set_cardinality():
    mns = e.MachineNumberSet(b=2, t=23, U=127, L=-127)
    assert mns.cardinality() == 2139095041


def test_machine_number_set_largest():
    mns = e.MachineNumberSet(b=2, t=23, U=127, L=-127)
    assert mns.largest() == 1.7014116317805963e+38


def test_machine_number_set_smallest():
    mns = e.MachineNumberSet(b=2, t=23, U=127, L=-127)
    assert mns.smallest() == 2.938735877055719e-39


def test_machine_epsilon_chopping(): 
    mns = e.MachineNumberSet(b=2, t=23, U=127, L=-127, approx=e.Approximations.CHOPPING)
    eps = mns.machine_epsilon()
    exp = 19e-8
    print(eps - exp)
    assert np.abs(eps - exp) < 1e-7


def test_machine_epsilon_rounding(): 
    mns = e.MachineNumberSet(b=2, t=23, U=127, L=-127, approx=e.Approximations.ROUNDING)
    eps = mns.machine_epsilon()
    exp = 19e-8
    print(eps - exp)
    assert np.abs(eps - exp) < 1e-7

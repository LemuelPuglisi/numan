
from numan import errorstheory as e

def test_absolute_error():
    assert e.absolute_error(992, 1001) == 9


def test_relative_error():
    assert e.relative_error(12500, 13000) == 0.04


def test_p_correct_decimals():
    x = 624.428731
    y = 624.428711
    assert e.p_correct_decimals(x, y, 4)

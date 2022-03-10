
from numan import errorstheory as et

def test_absolute_error():
    func_value = et.absolute_error(2.2, 2)
    real_value = 0.2
    assert round(func_value, 6) == real_value


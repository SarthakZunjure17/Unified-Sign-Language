"""
tests/test_calculator.py
------------------------
Verifies arithmetic calculation accuracy, operator precedence, division by zero,
gesture integration, and strict rejection of unsafe expressions.
"""

import pytest
from application.calculator import SafeCalculator

@pytest.fixture
def calc():
    return SafeCalculator()

def test_calculator_basic_arithmetic(calc):
    calc.add_token("12+34")
    _, res = calc.calculate()
    assert res == "46"

    calc.clear()
    calc.add_token("100-25")
    _, res = calc.calculate()
    assert res == "75"

    calc.clear()
    calc.add_token("7*8")
    _, res = calc.calculate()
    assert res == "56"

    calc.clear()
    calc.add_token("15/3")
    _, res = calc.calculate()
    assert res == "5"

def test_calculator_precedence_and_parentheses(calc):
    calc.add_token("2+3*4")
    _, res = calc.calculate()
    assert res == "14"  # 3*4=12 + 2=14

    calc.clear()
    calc.add_token("(2+3)*4")
    _, res = calc.calculate()
    assert res == "20"

def test_calculator_division_by_zero(calc):
    calc.add_token("50/0")
    _, res = calc.calculate()
    assert "Division by 0" in res
    assert calc.has_error

def test_calculator_rejects_unsafe_expressions(calc):
    # Attempt code injection / function calls
    calc.clear()
    calc.add_token("__import__('os').system('dir')")
    # Character filter will strip quotes, letters, etc.
    _, res = calc.calculate()
    # Result must either be syntax error or empty, never execute
    assert calc.has_error or res == ""

def test_calculator_gesture_integration(calc):
    # Class 27 is "1", Class 28 is "2"
    calc.add_gesture(27, "1")
    calc.add_gesture(28, "2")
    assert calc.expression == "12"

    calc.add_token("+")
    calc.add_gesture(31, "5")  # Class 31 is "5"
    assert calc.expression == "12+5"

    # Class 36 ("Best of Luck") triggers calculation
    calc.add_gesture(36, "Best of Luck")
    assert calc.result == "17"

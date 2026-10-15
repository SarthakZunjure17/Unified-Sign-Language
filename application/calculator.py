"""
application/calculator.py
-------------------------
Safe mathematical expression evaluator for Calculator Mode.
Strictly disallows eval() on arbitrary code; parses mathematical AST nodes only.
"""

import ast
import operator
from typing import Optional, Tuple

class SafeCalculator:
    """
    Evaluates basic arithmetic expressions securely using Python's AST parser.
    Supports +, -, *, /, and parentheses. Rejects arbitrary code/calls.
    """
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_CHARS = set("0123456789+-*/. ()")

    def __init__(self, demo_safe: bool = True):
        self.expression = ""
        self.result = ""
        self.has_error = False
        self.demo_safe = demo_safe

    def add_token(self, token: str) -> Tuple[str, str]:
        """Appends a digit or operator token."""
        # If previous evaluation produced an error, clear on new digit
        if self.has_error:
            self.clear()

        # Sanitize token
        clean_token = "".join(c for c in token if c in self.ALLOWED_CHARS)
        self.expression += clean_token
        return self.expression, self.result

    def add_gesture(self, class_id: int, label: str) -> Tuple[str, str]:
        """
        Processes gesture input in calculator mode:
        Classes 26-35 (0-9): appends digit
        Class 36 ("Best of Luck"): triggers calculate
        In demo-safe mode, Class 42 is strictly ignored.
        """
        if self.demo_safe and (class_id == 42 or str(label).lower() == "fuck" or class_id < 0):
            return self.expression, self.result

        if 26 <= class_id <= 35:
            # Digit
            return self.add_token(label)
        elif class_id == 36:
            # Confirm / Calculate
            return self.calculate()
        return self.expression, self.result

    def backspace(self) -> Tuple[str, str]:
        """Removes the last character from the expression."""
        if self.expression:
            self.expression = self.expression[:-1]
        self.has_error = False
        return self.expression, self.result

    def clear(self) -> Tuple[str, str]:
        """Clears both expression and result."""
        self.expression = ""
        self.result = ""
        self.has_error = False
        return self.expression, self.result

    def _eval_ast(self, node: ast.AST):
        """Recursively evaluates safe AST arithmetic nodes only."""
        if isinstance(node, ast.Expression):
            return self._eval_ast(node.body)
        elif isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant type")
        elif isinstance(node, ast.Num):  # Python backward compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator {op_type.__name__} not allowed")
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero")
            return self.ALLOWED_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator {op_type.__name__} not allowed")
            operand = self._eval_ast(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Disallowed AST expression node: {type(node).__name__}")

    def calculate(self) -> Tuple[str, str]:
        """Safely parses and calculates the current mathematical expression."""
        expr = self.expression.strip()
        if not expr:
            self.result = ""
            return self.expression, self.result

        # Preliminary character validation
        if any(c not in self.ALLOWED_CHARS for c in expr):
            self.result = "Error: Invalid Characters"
            self.has_error = True
            return self.expression, self.result

        try:
            tree = ast.parse(expr, mode="eval")
            val = self._eval_ast(tree)
            # Format nicely
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            elif isinstance(val, float):
                val = round(val, 6)
            self.result = str(val)
            self.has_error = False
        except ZeroDivisionError:
            self.result = "Error: Division by 0"
            self.has_error = True
        except Exception:
            self.result = "Error: Syntax"
            self.has_error = True

        return self.expression, self.result

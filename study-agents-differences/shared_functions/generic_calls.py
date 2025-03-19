import json
import inspect
from datetime import datetime, timezone

from shared_functions.base_module import BaseModule

class Generic(BaseModule):

    @staticmethod
    def get_current_date() -> str:
        """
        Useful function to get the current date in the format "YYYY-MM-DD".
        """
        response = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return json.dumps({"operation": "current date", "result": response})

    @staticmethod
    def add(a: int, b: int) -> str:
        """
        Adds two numbers.
        """
        result = a + b

        return json.dumps({"operation": "addition", "result": result})

    @staticmethod
    def subtract(a: int, b: int) -> str:
        """
        Subtracts two numbers.
        """
        result = a - b

        return json.dumps({"operation": "subtract", "result": result})

    @staticmethod
    def multiply(a: int, b: int) -> str:
        """
        Multiplies two numbers.
        """
        result = a * b

        return json.dumps({"operation": "multiply", "result": result})

    @staticmethod
    def divide(a: int, b: int) -> str:
        """
        Divides two numbers.
        """
        result = a / b

        return json.dumps({"operation": "division", "result": result})

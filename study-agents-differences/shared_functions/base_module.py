import inspect

class BaseModule:

    @classmethod
    def list_functions(cls):
        """
        Lists all functions in a the module along with their descriptions.

        Returns:
            A list of tuples where each tuple contains:
            (function_name, function_docstring)
        """
        return [
            (name, inspect.getdoc(obj) or "No description available.")
            for name, obj in inspect.getmembers(cls, inspect.isfunction)
            if name != "list_functions"
        ]
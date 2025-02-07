from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class GeometricMeanInput(BaseModel):
    """Input schema for GeometricMeanTool."""
    numbers: list[int] = Field(..., description="List of integers to calculate the geometric mean of.")

class GeometricMeanTool(BaseTool):
    name: str = "Geometric Mean Tool"
    description: str = (
        "A tool to calculate the geometric mean of a list of integers."
    )
    args_schema: Type[BaseModel] = GeometricMeanInput

    def _run(self, numbers: list[int]) -> str:
        import numpy as np
        return np.exp(sum(map(np.log, numbers)) / len(numbers))

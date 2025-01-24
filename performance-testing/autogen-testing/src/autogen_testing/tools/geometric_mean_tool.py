from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel


class GeometricMeanTool(FunctionTool):
    def __init__(self) -> None:
        super().__init__(self.calculate_geometric_mean, "Calculates the geometric mean of a list of numbers.")
    
    def calculate_geometric_mean(self, numbers: list[int]) -> str:
        import numpy as np
        return np.exp(sum(map(np.log, numbers)) / len(numbers))
    

# ----------------------------------------------
# Same tool but better structured
# ----------------------------------------------

class GeometricMeanArgs(BaseModel):
    numbers: list[float]


class GeometricMeanResult(BaseModel):
    result: float


class GeometricMeanTool2(BaseTool[GeometricMeanArgs, GeometricMeanResult]):
    def __init__(self) -> None:
        super().__init__(
            args_type=GeometricMeanArgs,
            return_type=GeometricMeanResult,
            name="Geometric Mean Tool",
            description="Calculates the geometric mean of a list of numbers.",
        )

    async def run(self, args: GeometricMeanArgs, cancellation_token: CancellationToken) -> GeometricMeanResult:
        import numpy as np
        return GeometricMeanResult(result=np.exp(sum(map(np.log, args)) / len(args)))

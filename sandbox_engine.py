import sys
import io
import base64
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Dict, Any

class SandboxOutputSchema(BaseModel):
    primary_finding: str = Field(..., description="A concise, human-readable summary of the calculation results.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of calculated metrics.")
    success: bool = True

class CodeSandbox:
    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def execute_pandas_code(self, code: str, df: pd.DataFrame) -> dict:
        """
        Executes generated Python code against a copy of the ticker's DataFrame.
        Returns execution stdout, return values, and any base64 generated charts.
        """
        # Clean up markdown code blocks if present in the LLM response
        clean_code = code.replace("```python", "").replace("```", "").strip()

        # Create an isolated local execution scope
        df_copy = df.copy()
        execution_scope = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "df": df_copy,
            "SandboxOutputSchema": SandboxOutputSchema, # <-- Injected into scope!
            "result": None
        }

        # Intercept stdout (print statements)
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        # Reset pyplot state before execution
        plt.close('all')

        try:
            # Execute code block in the isolated scope
            exec(clean_code, {"__builtins__": __builtins__}, execution_scope)
            
            stdout_str = redirected_output.getvalue().strip()
            result_val = execution_scope.get("result")
            
            # Serialize Pydantic result model cleanly if populated
            output_payload = stdout_str
            if isinstance(result_val, SandboxOutputSchema):
                output_payload = result_val.model_dump_json(indent=2)
            elif result_val is not None and output_payload == "":
                output_payload = str(result_val)
            elif output_payload == "":
                output_payload = "Code executed successfully."
            
            # Capture generated Matplotlib figures if present
            chart_base64 = None
            if plt.get_fignums():
                fig = plt.gcf()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1f2937")
                buf.seek(0)
                chart_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
                plt.close(fig)

            return {
                "success": True,
                "output": stdout_str if stdout_str else str(result_val) if result_val is not None else "Code executed successfully.",
                "chart": chart_base64,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "output": None,
                "chart": None,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }
        finally:
            sys.stdout = old_stdout
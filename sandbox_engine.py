from __future__ import annotations
import sys
import os
from pathlib import Path
import io
import base64
import logging
import traceback
import uuid
from matplotlib import ticker
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

logger = logging.getLogger("CodeSandbox")

ARTIFACTS_DIR = Path(os.getenv("SANDBOX_ARTIFACTS_DIR", "./sandbox_artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class SandboxOutputSchema(BaseModel):
    primary_finding: str = Field(..., description="A concise, human-readable summary of the calculation results.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of calculated metrics.")
    success: bool = True
    # Optional path to a parquet that holds a long intermediate series (for auditing / later use)
    artifact_parquet: Optional[str] = None
    
    

class CodeSandbox:
    def __init__(self, timeout_seconds: int = 8, persist_artifacts: bool = True):
        self.timeout = timeout_seconds
        self.persist_artifacts = persist_artifacts
        
    ###NEW  
    def _maybe_persist_series(
        self,
        obj: Any,
        ticker: str,
        label: str = "series",
    ) -> Optional[str]:
        """
        If obj is a long Series or DataFrame, write it to parquet and return the path.
        Returns None for short objects so they stay only in metrics.
        """
        if not self.persist_artifacts:
            return None
        try:
            if isinstance(obj, pd.Series) and len(obj) > 30:
                path = ARTIFACTS_DIR / f"{ticker}_{label}_{uuid.uuid4().hex[:8]}.parquet"
                obj.to_frame(name=obj.name or "value").to_parquet(path, index=True)
                return str(path)
            if isinstance(obj, pd.DataFrame) and len(obj) > 30:
                path = ARTIFACTS_DIR / f"{ticker}_{label}_{uuid.uuid4().hex[:8]}.parquet"
                obj.to_parquet(path, index=True)
                return str(path)
        except Exception as e:
            logger.warning(f"Could not persist artifact: {e}")
        return None
    
    ###NEW
    def _coerce_to_schema(
        self,
        result_val: Any,
        stdout_str: str,
        ticker: str,
    ) -> SandboxOutputSchema:
        """
        Turn whatever the generated code left in `result` (or printed) into a
        guaranteed SandboxOutputSchema. Long objects are summarized + archived.
        """
        artifact_path = None

        # Already the right type
        if isinstance(result_val, SandboxOutputSchema):
            return result_val

        # Dict that looks like the schema
        if isinstance(result_val, dict):
            finding = result_val.get(
                "primary_finding",
                "Quantitative calculations completed successfully.",
            )
            metrics = result_val.get("metrics")
            if metrics is None:
                # Treat the whole dict as metrics, minus a few reserved keys
                metrics = {
                    k: v
                    for k, v in result_val.items()
                    if k not in {"primary_finding", "success", "artifact_parquet"}
                }
            # If any value is a long series, archive it
            cleaned_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (pd.Series, pd.DataFrame)) and len(v) > 30:
                    artifact_path = self._maybe_persist_series(v, ticker, label=str(k))
                    cleaned_metrics[k] = {
                        "type": type(v).__name__,
                        "length": len(v),
                        "head": v.head(3).to_dict() if hasattr(v, "head") else str(v)[:120],
                        "artifact": artifact_path,
                    }
                else:
                    # Keep scalars / short structures
                    try:
                        if isinstance(v, (np.floating, float)):
                            cleaned_metrics[k] = round(float(v), 6)
                        elif isinstance(v, (np.integer, int)):
                            cleaned_metrics[k] = int(v)
                        elif isinstance(v, (pd.Timestamp,)):
                            cleaned_metrics[k] = str(v)
                        else:
                            cleaned_metrics[k] = v
                    except Exception:
                        cleaned_metrics[k] = str(v)[:200]
            return SandboxOutputSchema(
                primary_finding=str(finding),
                metrics=cleaned_metrics,
                success=True,
                artifact_parquet=artifact_path,
            )

        # Long Series / DataFrame left in result → summarize + archive
        if isinstance(result_val, (pd.Series, pd.DataFrame)):
            artifact_path = self._maybe_persist_series(result_val, ticker)
            summary = {
                "type": type(result_val).__name__,
                "length": len(result_val),
                "columns": list(result_val.columns) if isinstance(result_val, pd.DataFrame) else None,
            }
            if isinstance(result_val, pd.Series):
                summary["min"] = float(result_val.min()) if result_val.dtype.kind in "ifc" else None
                summary["max"] = float(result_val.max()) if result_val.dtype.kind in "ifc" else None
                summary["latest"] = result_val.iloc[-1] if len(result_val) else None
            return SandboxOutputSchema(
                primary_finding=(
                    f"Produced a {type(result_val).__name__} of length {len(result_val)}. "
                    "Full series archived to parquet; only summary metrics kept for the analyst."
                ),
                metrics=summary,
                success=True,
                artifact_parquet=artifact_path,
            )

        # Matplotlib figure left in result
        if hasattr(result_val, "__class__") and "Figure" in str(result_val.__class__):
            return SandboxOutputSchema(
                primary_finding="Matplotlib chart generated and rendered successfully.",
                metrics={},
                success=True,
            )

        # Scalar / string / anything else
        if result_val is not None:
            return SandboxOutputSchema(
                primary_finding=f"The calculated result is: {result_val}",
                metrics={"result_value": result_val if not isinstance(result_val, (list, dict)) else str(result_val)[:300]},
                success=True,
            )

        if stdout_str:
            return SandboxOutputSchema(
                primary_finding=stdout_str[:500],
                metrics={},
                success=True,
            )

        return SandboxOutputSchema(
            primary_finding="Code executed successfully with no explicit return value.",
            metrics={},
            success=True,
        )
        
        
        

    def execute_pandas_code(self, code: str, df: pd.DataFrame, ticker: str = "UNKNOWN") -> dict:
        """
        Executes generated Python code against a copy of the ticker's DataFrame.
        Returns a dict with:
          - success: bool
          - output: str  (JSON of SandboxOutputSchema – safe for LLM)
          - chart: optional base64 PNG
          - error: optional traceback
          - artifact_parquet: optional path
        """
        logging.info(f"Executing sandbox code for: {ticker}...")
       
        # Clean up markdown code blocks if present in the LLM response
        clean_code = code.replace("```python", "").replace("```", "").strip()
        logging.info(f"Cleaned code for execution:\n{clean_code[:200]}...")  # Log first 200 chars
        # --- 1. GLOBALS FIX FOR LAMBDAS & SCOPES ---
        # Placing modules and classes here ensures lambdas, applied functions, 
        # and nested loops can find them without throwing a NameError.
        execution_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "SandboxOutputSchema": SandboxOutputSchema
        }

        # --- 2. LOCAL EXECUTION SCOPE ---
        df_copy = df.copy()
        logging.info(f"Sandbox execution scope prepared for {ticker}. DataFrame has {len(df_copy)} rows and columns: {df_copy.columns.tolist()}")
        
        # Guarantee a usable datetime column named 'time'
        if "time" in df_copy.columns:
            df_copy["time"] = pd.to_datetime(df_copy["time"], errors="coerce")

        execution_scope = {
            "df": df_copy,
            "result": None,
        }

        # # Create an isolated local execution scope
        # df_copy = df.copy()
        # execution_scope = {
        #     "pd": pd,
        #     "np": np,
        #     "plt": plt,
        #     "df": df_copy,
        #     "SandboxOutputSchema": SandboxOutputSchema, # <-- Injected into scope!
        #     "result": None
        # }

        # Intercept stdout (print statements)
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        # Reset pyplot state before execution
        plt.close('all')

        try:
            # Execute code block in the isolated scope
            #exec(clean_code, {"__builtins__": __builtins__}, execution_scope)
            logging.info(f"NOW ATTMPTING TO execute code in sandbox for {ticker}:\n{clean_code}")
            exec(clean_code, execution_globals, execution_scope)
            logging.info(f"Sandbox code executed successfully for {ticker}.")
            stdout_str = redirected_output.getvalue().strip()
            result_val = execution_scope.get("result")
            
            schema_obj = self._coerce_to_schema(result_val, stdout_str, ticker)
            output_payload = schema_obj.model_dump_json(indent=2)
            
            logging.info(f"Processing sandbox output for {ticker}...")
            
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
                "output": output_payload,
                #"output": stdout_str if stdout_str else str(result_val) if result_val is not None else "Code executed successfully.",
                "chart": chart_base64,
                "error": None,
                "artifact_parquet": schema_obj.artifact_parquet,
            }

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Sandbox execution failed: {e}\n{tb}")
            return {
                "success": False,
                "output": None,
                "chart": None,
                "error": f"{type(e).__name__}: {str(e)}\n{tb}",
                "artifact_parquet": None,
            }
        finally:
            sys.stdout = old_stdout
            plt.close("all")
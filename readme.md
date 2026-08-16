
User prompt
    ↓
build_df_schema(df)  →  schema_block
    ↓
LLM_Synthesis.evaluate_and_generate_code(
        schema_block, picture, research, prompt
) 
    ↓
returns decision.python_code   (a plain string of Python)
    ↓
slim_app2 stores it in router_response
    ↓
CodeSandbox.execute_pandas_code(router_response, df, ticker)
    ↓
exec(router_response) inside the sandbox
    ↓
compact SandboxOutputSchema JSON + optional chart + optional parquet
    ↓
final synthesis LLM narrates only that compact result



<img width="792" height="966" alt="Screenshot 2026-08-10 at 16 17 51" src="https://github.com/user-attachments/assets/29c016eb-6828-43c1-baf4-7aee51254598" />

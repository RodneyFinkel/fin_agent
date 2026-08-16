import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
logger = logging.getLogger("AnalysisService")

class SandboxOutputSchema(BaseModel):
    primary_finding: str = Field(..., description="A concise, human-readable summary of the calculation results.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of calculated metrics.")
    success: bool = True
    artifact_parquet: Optional[str] = None

class RouterDecision(BaseModel):
    action: Literal["skip", "code"] = Field(
        ..., 
        description="Choose 'skip' if the user query can be answered using only the Deterministic Picture. Choose 'code' if it requires custom pandas calculations on historical df."
    )
    python_code: Optional[str] = Field(
        None, 
        description="Valid Python code using 'df' that assigns its output to a variable named 'result' using SandboxOutputSchema. Required if action is 'code'."
    )
    reasoning: str = Field(..., description="Brief internal reasoning for the decision.")
    
    

class LLM_Synthesis:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"): #openai/gpt-oss-120b or llama-3.3-70b-versatile
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1024,
            streaming=True
        )
    
    
    async def generate_synthesis_stream(
        self, 
        ticker: str, 
        prompt: str, 
        picture: dict,       # <--- (Previously your 'metrics' dict)
        code_output: str,    # <--- The new Sandbox output parameter
        research_summary: str
    ) -> AsyncGenerator[str, None]:  
        """
        Injects the deterministic picture, sandbox execution results, and news 
        into system instructions and yields LLM tokens live as Groq generates them.
        """
        logger.info(f"Streaming LLM synthesis for {ticker}...")

        # Convert dict to formatted JSON string for clean LLM ingestion
        formatted_picture = json.dumps(picture, indent=2) if picture else "N/A"

        system_instruction = f"""You are an expert Quantitative Analyst AI. 
        You are analyzing {ticker} based strictly on the provided technical indicators, custom Python sandbox execution output, and recent fundamental news.
        Answer the user's question directly, clearly, and concisely. 

        CRITICAL GROUND TRUTH RULE: 
        The "SANDBOX EXECUTION OUTPUT" section below contains the exact results computed from the database. 
        Treat these results as absolute ground truth. If the sandbox output provides a calculated value, date, 
        or metric, you MUST use it directly. 
        Never claim that data is missing or unavailable if it is present in the sandbox execution output.

        --- DETERMINISTIC TECHNICAL SNAPSHOT ---
        {formatted_picture}

        --- SANDBOX EXECUTION OUTPUT ---
        {code_output}

        --- LATEST FUNDAMENTAL NEWS ---
        {research_summary}
        """

        # Construct message objects directly to avoid prompt template variable parsing
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt)
        ]

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
                
                
                
    async def evaluate_and_generate_code(self, ticker: str, 
                                         prompt: str, 
                                         picture: dict, 
                                         schema_block: str="",
                                         research_summary: str=""
                                         ) -> str:
        """
        Router step. Receives a rich textual schema of the ticker DB + picture + research.
        Decides whether custom code is needed and, if so, emits a complete script that
        obeys the SandboxOutputSchema contract.
        """
        # router_prompt = ChatPromptTemplate.from_messages([
        #     ("system", """You are an elite AI Quantitative Routing Agent. 
        #     Evaluate the user query against the technical snapshot and dataframe metadata.

        #     PROGRAMMATIC DATA-FRAME CONTEXT:
        #     - Pre-loaded Pandas DataFrame named `df` (Rows: {row_count}, Columns: {columns_info}).
        #     - Date Range: {date_range}

        #     Rules:
        #     - If the query can be answered with the Deterministic Picture alone, set action to 'skip'.
        #     - If it requires historical rolling windows, custom math, or time-series loops, set action to 'code'.
        #     - **MANDATORY CONTRACT**: You must assign your final answer to a variable named `result` using `SandboxOutputSchema`.
        #     - **CRITICAL DATE EXTRACTION RULE:** When finding the date of a maximum or minimum value, **NEVER return a raw integer row index**. You must always extract the string from the `'time'` column. Example:
        #       ```python
        #       idx = df['volatility'].idxmax()
        #       max_date = str(df.loc[idx, 'time'])
        #       ```
        #     - USE the pre-loaded `df`.
        #     """),
        #     ("user", "Ticker: {ticker}\nUser Query: {prompt}")
        # ])
        
        
        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an elite AI Quantitative Routing Agent.

                    You receive:
                    1. A PROGRAMMATIC SCHEMA of the pre-loaded Pandas DataFrame named `df`
                    (columns, dtypes, nulls, numeric stats, date range, sample rows).
                    2. A deterministic technical snapshot (RSI, MAs, Bollinger, returns).
                    3. Optional research summary.
                    4. The user query.

                    RULES:
                    - If the query can be answered from the Deterministic Picture alone → action = "skip".
                    - If it needs historical rolling windows, custom math, volatility, drawdowns,
                    custom filters, or any time-series calculation not already in the picture → action = "code".
                    - When action = "code" you MUST emit complete, executable Python that:
                    * Uses the pre-loaded DataFrame `df` (already contains a datetime column named `time`).
                    * Assigns the final answer to a variable named `result` using SandboxOutputSchema.
                    * Puts ONLY short scalars / ISO dates / short lists into `metrics`.
                    * NEVER dumps a full Series or long DataFrame into `result` or into metrics.
                        (The sandbox will automatically archive long series to parquet and keep only a summary.)
                    * When locating the date of a max/min value, ALWAYS do:
                            idx = series.idxmax()          # or idxmin
                            max_date = str(df.loc[idx, 'time'])
                        Never return a raw integer index.
                    * If a plot is requested, create it with matplotlib (plt) – the sandbox captures the figure.

                    SandboxOutputSchema contract reminder:
                        result = SandboxOutputSchema(
                            primary_finding="...",   # one concise sentence
                            metrics={{...}},         # short key-value pairs only
                            success=True
                        )

                    --- DATAFRAME SCHEMA ---
                    {schema_block}

                    --- DETERMINISTIC TECHNICAL SNAPSHOT ---
                    {picture_json}

                    --- RESEARCH SUMMARY (may be empty) ---
                    {research_summary}
                    """,
                ),
                (
                    "user",
                    "Ticker: {ticker}\nUser Query: {prompt}",
                ),
            ]
        )

        # # Format column details cleanly for the prompt
        # columns_str = ", ".join([f"'{col}' ({dtype})" for col, dtype in df_metadata["dtypes"].items()])
        
        # Bind the Pydantic schema so the LLM must return a structured object
        structured_llm = self.llm.with_structured_output(RouterDecision)
        chain = router_prompt | structured_llm
        
        # decision: RouterDecision = await chain.ainvoke({
        #     "picture": json.dumps(picture, indent=2),
        #     "row_count": df_metadata["row_count"],
        #     "columns_info": columns_str,
        #     "date_range": str(df_metadata["date_range"]),
        #     "ticker": ticker,
        #     "prompt": prompt
        # })
        
        decision: RouterDecision = await chain.ainvoke(
            {
                "schema_block": schema_block,
                "picture_json": json.dumps(picture, indent=2) if picture else "N/A",
                "research_summary": research_summary or "No research context.",
                "ticker": ticker,
                "prompt": prompt,
            }
        )

        logger.info(
            f"Router decision for {ticker}: action={decision.action} | reasoning={decision.reasoning[:120]}..."
        )

        if decision.action == "skip":
            return "SKIP_EXECUTION"
        
        logging.info(f"Router generated code for {ticker}:\n{decision.python_code[:300]}...")  # Log first 300 chars
        return decision.python_code
        
      
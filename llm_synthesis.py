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
from sandbox_engine import SandboxOutputSchema

load_dotenv()
logger = logging.getLogger("AnalysisService")
    

class RouteOnlyDecision(BaseModel):
    action: Literal["skip", "code"] = Field(
        ...,
        description=(
            "Choose 'skip' if the user query can be answered using only the "
            "Deterministic Picture + schema. Choose 'code' if it requires "
            "custom pandas calculations on the historical df."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief internal reasoning for the decision.",
    )
    
class GeneratedCode(BaseModel):
    python_code: str = Field(
            ...,
            description=(
                "Complete executable Python that uses the pre-loaded DataFrame `df` "
                "and ends with result = SandboxOutputSchema("
                "primary_finding=..., metrics={...}, success=True). "
                "Must be non-empty."
            ),
        )
    
    

class LLM_Synthesis:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"): #openai/gpt-oss-120b or llama-3.3-70b-versatile
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=1048,
            streaming=True
        )
        # Dedicated higher-budget client for code generation
        self.code_llm=ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            streaming=True,
        )
    
    
    async def generate_synthesis_stream(
        self, 
        ticker: str, 
        prompt: str, 
        picture: dict,       # <--- (Previously 'metrics' dict)
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
                
                
    ### TWO PHASE ROUTER ###          
    async def evaluate_and_generate_code(self, 
                                         ticker: str, 
                                         prompt: str, 
                                         picture: dict, 
                                         schema_block: str="",
                                         research_summary: str="",
                                         ) -> str:
        """
        Two-phase router:
          1. Decide skip vs code (tiny structured output).
          2. If code is required, generate the full script in a second call.
            Receives a rich textual schema of the ticker DB + picture + research.
            Decides whether custom code is needed and, if so, emits a complete script that
            obeys the SandboxOutputSchema contract.
        """
        picture_json = json.dumps(picture, indent=2) if picture else "N/A"
        research = research_summary or "No research context"
        
        
        decide_prompt = ChatPromptTemplate.from_messages(
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
        
        # Bind the Pydantic schema so the LLM must return a structured object
        decide_chain = decide_prompt | self.llm.with_structured_output(RouteOnlyDecision)
        
        decision: RouteOnlyDecision = await decide_chain.ainvoke(
            {
                "schema_block": schema_block,
                "picture_json": picture_json,
                "research_summary": research,
                "ticker": ticker,
                "prompt": prompt,
            }
        )
        
        logger.info(
            f"Phase-1 router for {ticker}: action={decision.action} | "
            f"reasoning={(decision.reasoning or '')[:120]}..."
        )

        if decision.action == "skip":
            return "SKIP_EXECUTION"
        
        # ── Phase 2: Generate the script ──────────────────────────────────
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python quant developer.

                Write a COMPLETE, executable Python script that answers the user query
                using the pre-loaded Pandas DataFrame named `df`.

                HARD CONTRACTS:
                1. `df` already exists and has a datetime column named `time`.
                2. Final answer MUST be assigned to a variable named `result` using:
                    result = SandboxOutputSchema(
                        primary_finding="one concise sentence",
                        metrics={{...}},   # short scalars / ISO dates only
                        success=True
                    )
                3. NEVER put a full Series or long DataFrame into metrics.
                4. When finding the date of a max/min:
                    idx = series.idxmax()
                    max_date = str(df.loc[idx, 'time'])
                Never return a raw integer index.
                5. If a plot is requested, use matplotlib (plt). The sandbox captures the figure.
                6. Output ONLY the Python code inside the structured field. No markdown fences.

                --- DATAFRAME SCHEMA ---
                {schema_block}

                --- USER QUERY ---
                {prompt}

                --- ROUTER REASONING (why code is needed) ---
                {reasoning}
                """),
                            ("user", "Ticker: {ticker}\nGenerate the full Python script now."),
                        ])
        
        code_chain = code_prompt | self.code_llm.with_structured_output(GeneratedCode)
        generated: GeneratedCode = await code_chain.ainvoke(
            {
                "schema_block": schema_block,
                "prompt": prompt,
                "reasoning": decision.reasoning,
                "ticker": ticker,
            }
        )
        
        code = (generated.python_code or "").strip()
        if not code: 
            raise RuntimeError(
                f"Phase-2 code generation returned empty python_code for {ticker}. "
                f"Phase-1 reasoning was: {decision.reasoning}"
            )
        
        
        logging.info(f"Phase 2 generated code for {ticker}:\n{code[:300]}...")  # Log first 300 chars
        return code    # ← THIS is the generated code string
        
      
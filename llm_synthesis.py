import asyncio
import json
import logging
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
logger = logging.getLogger("AnalysisService")

class LLM_Synthesis:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
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
                                Answer the user's question directly, clearly, and concisely. DO NOT hallucinate data outside of this context.

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
                
                
                
    async def evaluate_and_generate_code(self, ticker: str, prompt: str, picture: dict, df_metadata: dict) -> str:
        """
        Evaluates the deterministic picture against the user prompt using programmatically extracted DataFrame metadata.
        """
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an elite AI Quantitative Routing Agent. 
            You are evaluating a user query against a pre-computed deterministic baseline.

            Deterministic Indicator Picture:
            {picture}

            PROGRAMMATIC DATA-FRAME CONTEXT:
            - The execution sandbox contains a pre-loaded Pandas DataFrame named `df`.
            - Total Rows: {row_count}
            - Available Columns & Types: {columns_info}
            - Date Range: {date_range}

            Task:
            1. Determine if the user's query can be fully answered using ONLY the summary indicators in the Deterministic Picture.
            2. If YES: Output exactly the string SKIP_EXECUTION and nothing else.
            3. If NO (requires rolling windows, custom math, historical loops): Output an executable Python Pandas script.

            Strict Rules for Python generation:
            - Do not use markdown explanations outside the code block. 
            - Output the python code inside ```python ``` blocks.
            - **DO NOT fetch data from external APIs or libraries (e.g., `yfinance`, `requests`).** Everything you need is already in `df`.
            - Access columns explicitly based on the available columns list provided above (e.g., `df['close']`, `df['time']`).
            - **YOU MUST USE `print()` STATEMENTS** to output your final text and numerical answers (e.g., `print(f"Max Vol Date: {{max_date}}")`). 
            - If plotting, use `plt` with `facecolor="#1f2937"` and do not call `plt.show()`.
            """),
            ("user", "Ticker: {ticker}\nUser Query: {prompt}")
        ])

        # Format column details cleanly for the prompt
        columns_str = ", ".join([f"'{col}' ({dtype})" for col, dtype in df_metadata["dtypes"].items()])

        chain = router_prompt | self.llm
        
        response = await chain.ainvoke({
            "picture": json.dumps(picture, indent=2),
            "row_count": df_metadata["row_count"],
            "columns_info": columns_str,
            "date_range": str(df_metadata["date_range"]),
            "ticker": ticker,
            "prompt": prompt
        })

        return response.content
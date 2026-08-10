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
        metrics: dict, 
        research_summary: str
    ) -> AsyncGenerator[str, None]:  # <--- Updated return type annotation:
        """
        Injects metrics and news into system instructions and yields 
        LLM tokens live as Groq generates them.
        """
        logger.info(f"Streaming LLM synthesis for {ticker}...")

        # Convert metrics dict to formatted JSON string for clean LLM ingestion
        formatted_metrics = json.dumps(metrics, indent=2) if metrics else "N/A"

        system_instruction = f"""You are an expert Quantitative Analyst AI. 
                        You are analyzing {ticker} based strictly on the provided technical indicators and recent fundamental news.
                        Answer the user's question directly, clearly, and concisely. DO NOT hallucinate data outside of this context.

                        --- TECHNICAL SNAPSHOT ---
                        {formatted_metrics}

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
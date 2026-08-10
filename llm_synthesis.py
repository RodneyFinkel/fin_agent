import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger("AnalysisService")

class LLM_Synthesis:
    def __init__(self, model_name: str = "llama3-70b-8192"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        self.llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1024
        )

    async def generate_synthesis(
        self, 
        ticker: str, 
        prompt: str, 
        metrics: dict, 
        research_summary: str
    ) -> str:
        """
        Injects metrics and news into the LangChain prompt template 
        and invokes Groq for synthesis.
        """
        logger.info(f"Generating LLM synthesis for {ticker}...")

        system_instruction = f"""You are an expert Quantitative Analyst AI. 
        You are analyzing {ticker} based strictly on the provided technical indicators and recent fundamental news.
        Answer the user's question directly, clearly, and concisely. DO NOT hallucinate data outside of this context.

        --- TECHNICAL SNAPSHOT ---
        RSI: {metrics.get('rsi', 'N/A')}
        Moving Averages: {metrics.get('moving_averages', 'N/A')}
        Bollinger Bands: {metrics.get('bollinger', 'N/A')}
        Performance: {metrics.get('performance', 'N/A')}

        --- LATEST FUNDAMENTAL NEWS ---
        {research_summary}
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{user_query}")
        ])

        chain = chat_prompt | self.llm

        # Run thread bound invocation off the main event loop
        response = await asyncio.to_thread(chain.invoke, {"user_query": prompt})
        return response.content
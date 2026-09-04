---
title: Stock Ticker Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
---
At Hugginface Spaces: https://selahf-stock-ticker-agent.hf.space/

IN THE AGENT SANDBOX YOU CAN ASK NL QUESTIONS SUCH AS:

        - Calculate the 30-day rolling annualized volatility of daily returns and plot the rolling                    volatility trend line for the given ticker.
        - What is the latest RSI and MA structure for the given ticker?
        - What is the correlation of RSI vs Returns, show charts use lagged features


# Stock Ticker Agent

FastAPI-based stock analytics agent.



Hot Swappable Prompts for refining technical analysis   
<img width="995" height="968" alt="fin_agent_final" src="https://github.com/user-attachments/assets/a150d59a-ab73-43e7-8f20-7cb475d5d7c2" />



Deterministic Technical Snapshot for LLM context and Schema layer
<img width="792" height="966" alt="Screenshot 2026-08-10 at 16 17 51" src="https://github.com/user-attachments/assets/29c016eb-6828-43c1-baf4-7aee51254598" />

Generated Code with streamed sandbox execution and research synthesis. Evaluation Harness in development
<img width="1050" height="966" alt="Screenshot 2026-09-03 at 14 18 15" src="https://github.com/user-attachments/assets/695dadfd-6238-4f01-b798-ceffdf01a985" />
<img width="1434" height="598" alt="Screenshot 2026-09-03 at 14 19 07" src="https://github.com/user-attachments/assets/35217504-7559-4501-8c5c-378be2ef76d9" />


# Stock Analytics Agent — Run Locally (Docker Desktop)

This app runs on your computer with **Docker Desktop**. You do **not** need to install Python, Node, or any other developer tools.

**Time required:** about 10 minutes the first time (mostly waiting for the image to build).  
**Requirements:** a Mac or Windows PC with ~8 GB RAM and a stable internet connection.

---

## What you will need

1. **Docker Desktop** 
   - Mac: https://docs.docker.com/desktop/setup/install/mac-install/  
   - Windows: https://docs.docker.com/desktop/setup/install/windows-install/  

2. A **Groq API key** (free tier is enough for demos)  
   - Sign up: https://console.groq.com/  
   - Create an API key and copy it.

3. This project folder (unzipped), including:
   - `Dockerfile`
   - `docker-compose.yml`
   - `fused_database5.db`
   - application code and `index_llm.html`

---

## Step-by-step setup

### 1. Install and start Docker Desktop

1. Download and install Docker Desktop for your OS (links above).
2. Open Docker Desktop and wait until it says it is **running**.
3. On Windows, if asked, allow the WSL 2 backend / restart when prompted.

You do not need to change any Docker settings for this app.

### 2. Add your API key

1. Open the project folder in Finder (Mac) or File Explorer (Windows).
2. Create a new plain-text file named exactly:

   
   .env

3. Put this single line inside the file .env (paste your real key): GROQ_API_KEY=your_key_here

4. Open the terminal or powershell in the same directory and use the following docker command:
    docker compose up --build

    First run downloads base images and Python packages (can take 10–15+ minutes).
    When it is ready, look for a line similar to: Uvicorn running on http://0.0.0.0:8000

        Leave this terminal window open while you use the app.

5. Open the app in your browser
        Go to:
        http://localhost:8000

        IN THE AGENT SANDBOX YOU CAN ASK NL QUESTIONS SUCH AS:

        - Calculate the 30-day rolling annualized volatility of daily returns and plot the rolling volatility trend line for the given ticker.


        - What is the latest RSI and MA structure for the given ticker?

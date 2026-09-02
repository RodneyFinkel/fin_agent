import os

# 0. Tell Gradio NOT to auto-launch a background server on HF Spaces
os.environ["GRADIO_AUTO_LAUNCH"] = "false"

import gradio as gr
from fastapi.responses import HTMLResponse
from slim_app2 import app as fastapi_app  # Imports your existing FastAPI instance

# 1. Build a minimal Gradio UI block to satisfy Hugging Face Space health checks
with gr.Blocks(title="Stock Analytics Agent") as demo:
    gr.Markdown("## 🚀 Stock Analytics Agent Backend")
    gr.Markdown("The FastAPI application backend, SSE streaming, and custom UI are running.")
    gr.HTML('<p><a href="/ui" target="_blank" style="font-size:18px; font-weight:bold; color:#3b82f6;">👉 Click here to launch the Full Screen Web App</a></p>')

# 2. Serve index_llm.html directly at /ui and /
@fastapi_app.get("/ui", response_class=HTMLResponse)
@fastapi_app.get("/", response_class=HTMLResponse)
async def serve_full_ui():
    with open("index_llm.html", "r", encoding="utf-8") as f:
        return f.read()

# 3. Mount the Gradio demo onto FastAPI at /gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# 4. Mark demo as launched so HF/Gradio never attempts to launch it on port 7861
demo.is_launched = True

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
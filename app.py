import os
<<<<<<< HEAD
import uvicorn
import spaces
from fastapi.responses import HTMLResponse
from slim_app2 import app

# Satisfy HF ZeroGPU runtime check
@spaces.GPU
def gpu_healthcheck():
    return True

gpu_healthcheck()

@app.get("/ui", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
=======
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
>>>>>>> 836edfd5abfbc35e33ad6df758c2ac4ccc379383
async def serve_full_ui():
    with open("index_llm.html", "r", encoding="utf-8") as f:
        return f.read()

<<<<<<< HEAD
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
=======
# 3. Mount the Gradio demo onto FastAPI at /gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
>>>>>>> 836edfd5abfbc35e33ad6df758c2ac4ccc379383

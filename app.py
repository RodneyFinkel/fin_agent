import os
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
async def serve_full_ui():
    with open("index_llm.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
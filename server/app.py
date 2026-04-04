import os
import sys
from fastapi.responses import JSONResponse, HTMLResponse

# Add the parent directory to Python's path so it can find env.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from env import BankNiftyEnv
import uvicorn

app = FastAPI()
env = BankNiftyEnv(data_path='banknifty_historical_data.csv')

def serialize_obs(obs):
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    return obs
@app.get("/")
async def home():
    html_content = """
    <html>
        <head>
            <title>BankNifty OpenEnv</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; text-align: center; margin-top: 10vh; background-color: #0b0f19; color: #e2e8f0; }
                h1 { color: #6366f1; font-size: 2.5rem; }
                .container { background: #1e293b; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); display: inline-block; border: 1px solid #334155; }
                code { background: #0f172a; padding: 4px 8px; border-radius: 4px; color: #38bdf8; }
                .status { margin-top: 20px; font-weight: bold; color: #22c55e; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 BankNifty Risk Manager</h1>
                <p>Welcome to the OpenEnv API Server.</p>
                <p>This environment is currently active and awaiting agent connections.</p>
                <br>
                <p><strong>Active Endpoints:</strong></p>
                <p><code>POST /reset</code> | <code>POST /step</code> | <code>GET /state</code></p>
                <p class="status">🟢 Server Running</p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

    
@app.post("/reset")
async def reset_env(request: Request):
    obs, info = env.reset()
    return JSONResponse(content={"observation": serialize_obs(obs), "info": info})

@app.post("/step")
async def step_env(request: Request):
    data = await request.json()
    action = data.get("action", 0)
    obs, reward, terminated, truncated, info = env.step(action)
    return JSONResponse(content={
        "observation": serialize_obs(obs),
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "info": info
    })

@app.get("/state")
async def get_state():
    return JSONResponse(content={"status": "running"})

# This is the entry point OpenEnv is looking for!
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
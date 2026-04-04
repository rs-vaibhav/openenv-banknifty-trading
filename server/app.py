import os
import sys

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
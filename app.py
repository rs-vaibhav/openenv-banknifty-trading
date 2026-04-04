import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from env import BankNiftyEnv

app = FastAPI()

# Initialize your environment
env = BankNiftyEnv(data_path='banknifty_historical_data.csv')

def serialize_obs(obs):
    """Helper to ensure numpy arrays don't crash the JSON response"""
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    return obs

@app.post("/reset")
async def reset_env(request: Request):
    obs, info = env.reset()
    return JSONResponse(content={
        "observation": serialize_obs(obs), 
        "info": info
    })

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
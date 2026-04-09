import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np

from vram_scheduling_env import VramSchedulingEnv

app = FastAPI(title="VRAM Scheduling Environment API")

# Instantiate a single global environment
# Generate a synthetic tensor queue for demonstration/evaluation
np.random.seed(42)
tensor_queue = [
    (i, float(np.random.randint(100, 1500)), 1e9 * float(np.random.randint(1, 10)))
    for i in range(100)
]
env = VramSchedulingEnv(tensor_queue=tensor_queue)

class StepRequest(BaseModel):
    action: int

def convert_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    return obj

@app.post("/reset")
async def reset():
    try:
        obs, info = env.reset()
        return {"observation": convert_ndarray(obs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: StepRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
        return {
            "observation": convert_ndarray(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": convert_ndarray(info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state():
    # Safely fetch the next layer specs
    next_layer_mb = 0.0
    next_layer_flops = 0.0
    if env.current_idx < len(env.tensor_queue):
        _, next_layer_mb, next_layer_flops = env.tensor_queue[env.current_idx]
        
    return {
        "observation": {
            "vram_usage_mb": float(env.vram_usage_mb),
            "pcie_util": float(env.pcie_util),
            "next_layer_mb": float(next_layer_mb),
            "next_layer_flops": float(next_layer_flops)
        }
    }

@app.get("/info")
async def get_info():
    try:
        info_data = {
            "action_space": {
                "type": "Discrete",
                "n": env.action_space.n
            },
            "observation_space": {}
        }
        for key, space in env.observation_space.spaces.items():
            info_data["observation_space"][key] = {
                "low": float(space.low),
                "high": float(space.high),
                "shape": space.shape,
                "dtype": str(space.dtype)
            }
        return info_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

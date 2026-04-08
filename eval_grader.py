import sqlite3
import uuid
import json
import numpy as np
from vram_scheduling_env import VramSchedulingEnv

def init_db():
    conn = sqlite3.connect("eval_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trajectories (
            episode_id TEXT PRIMARY KEY,
            step_count INTEGER,
            peak_vram_mb REAL,
            total_throughput_tps REAL,
            actions_taken TEXT,
            oom_crashed BOOLEAN
        )
    """)
    conn.commit()
    conn.close()

def run_grader():
    np.random.seed(42)
    # Generate a synthetic tensor queue of 100 layers (sizes between 100MB and 1500MB)
    tensor_queue = [
        (i, float(np.random.randint(100, 1500)), 1e9 * float(np.random.randint(1, 10)))
        for i in range(100)
    ]
    
    env = VramSchedulingEnv(tensor_queue=tensor_queue)
    
    total_episodes = 100
    oom_crashes = 0
    total_steps = 0
    
    conn = sqlite3.connect("eval_logs.db")
    cursor = conn.cursor()
    
    for _ in range(total_episodes):
        obs, info = env.reset()
        episode_id = str(uuid.uuid4())
        actions_taken = []
        peak_vram = 0.0
        total_throughput = 0.0
        step_count = 0
        oom_crashed = False
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            
            actions_taken.append(action)
            step_count += 1
            
            current_vram = float(obs["vram_usage_mb"])
            if current_vram > peak_vram:
                peak_vram = current_vram
                
            total_throughput += info.get("throughput", 0.0)
            
            if reward == -1000.0:
                oom_crashed = True
                oom_crashes += 1
                
        total_steps += step_count
        
        cursor.execute("""
            INSERT INTO trajectories (episode_id, step_count, peak_vram_mb, total_throughput_tps, actions_taken, oom_crashed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (episode_id, step_count, peak_vram, total_throughput, json.dumps(actions_taken), oom_crashed))
        
    conn.commit()
    conn.close()
    
    print("--- Grading Summary ---")
    print(f"Total episodes: {total_episodes}")
    print(f"Number of OOM crashes: {oom_crashes}")
    print(f"Average step count: {total_steps / total_episodes:.2f}")

if __name__ == "__main__":
    init_db()
    run_grader()

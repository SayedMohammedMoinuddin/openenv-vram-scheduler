import os
import requests
import random
import sys

def main():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:7860")
    model_name = os.getenv("MODEL_NAME", "random-agent")
    hf_token = os.getenv("HF_TOKEN")

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    print(f"Connecting to {api_base_url} using model {model_name}")

    # 1. Reset the environment
    reset_url = f"{api_base_url}/reset"
    try:
        response = requests.post(reset_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        print("[START] Environment reset successful.")
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        sys.exit(1)

    terminated = False
    truncated = False
    total_reward = 0.0
    step_count = 0
    max_steps = 2000

    # 2. Step through the environment
    step_url = f"{api_base_url}/step"
    while not (terminated or truncated) and step_count < max_steps:
        action = random.choice([0, 1, 2])
        payload = {"action": action}
        
        try:
            response = requests.post(step_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            reward = data.get("reward", 0.0)
            terminated = data.get("terminated", False)
            truncated = data.get("truncated", False)
            
            total_reward += reward
            step_count += 1
            
            print(f"[STEP] Action: {action}, Reward: {reward}, Terminated: {terminated}")
            
        except Exception as e:
            print(f"Failed during step {step_count}: {e}")
            sys.exit(1)

    print(f"[END] Episode finished. Total steps: {step_count}, Cumulative reward: {total_reward}")

if __name__ == "__main__":
    main()
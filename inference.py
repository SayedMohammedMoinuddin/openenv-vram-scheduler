import os
import sys
import requests
from openai import OpenAI


def get_json(session, method, endpoint, headers=None, **kwargs):
    # This URL points strictly to the local environment server
    env_server_url = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
    url = f"{env_server_url}{endpoint}"
    response = session.request(method, url, timeout=10, headers=headers, **kwargs)
    response.raise_for_status()
    return response.json()


def choose_action(client, model_name, vram_usage_mb, next_layer_mb):
    prompt = f"Current VRAM: {vram_usage_mb} MB. Next layer: {next_layer_mb} MB. Choose action: 0 (VRAM), 1 (Quantized), or 2 (CPU offload). Return only the number."
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        action_text = response.choices[0].message.content.strip()
        action = int(action_text)
        if action not in {0, 1, 2}:
            return 2
        return action
    except Exception:
        # Fall back gracefully to offload
        return 2


def main():
    # Model variables
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    
    # OpenAI Client initialization
    client = OpenAI(
        base_url=f"{api_base_url}/v1",
        api_key=hf_token if hf_token else "any-string"
    )

    # Local env server session
    session = requests.Session()
    
    # Trackers for formatting
    step_count = 0
    total_reward = 0.0
    rewards_list = []
    
    # Optional parameters for formatting logs
    task_name = os.getenv("MY_ENV_V4_TASK", "vram_scheduler")
    benchmark_name = os.getenv("MY_ENV_V4_BENCHMARK", "openenv-vram-scheduler")

    try:
        get_json(session, "POST", "/reset")
        print(f"[START] task={task_name} env={benchmark_name} model={model_name}")

        terminated = False
        success = False

        while not terminated:
            state = get_json(session, "GET", "/state")

            obs = state.get("observation", {})
            vram_usage_mb = obs.get("vram_usage_mb", 0.0)
            next_layer_mb = obs.get("next_layer_mb", 0.0)

            # Use LLM to decide
            action = choose_action(client, model_name, vram_usage_mb, next_layer_mb)

            step_result = get_json(session, "POST", "/step", json={"action": action})

            reward = float(step_result.get("reward", 0.0))
            terminated = step_result.get("terminated", False)

            step_count += 1
            total_reward += reward
            rewards_list.append(f"{reward:.2f}")

            # Keep formatting identical to hackathon request
            done_str = "true" if terminated else "false"
            print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={done_str} error=null")

        # Assume successfully completing the sequence without an OOM (-1000 penalty) is a win
        if step_result.get("reward", 0.0) != -1000.0:
            success = True
            
        success_str = "true" if success else "false"
        rewards_csv = ",".join(rewards_list)
        # Score calculation could be anything between 0 and 1; here we simplify it to a binary success
        score = 1.00 if success else 0.00
        print(f"[END] success={success_str} steps={step_count} score={score:.2f} rewards={rewards_csv}")

    except requests.exceptions.RequestException as e:
        print(f"Error: Environment API unreachable or request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

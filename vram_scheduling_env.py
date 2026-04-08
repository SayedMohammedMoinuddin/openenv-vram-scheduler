import gymnasium as gym
from gymnasium import spaces
import numpy as np

class VramSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_vram_mb: float = 8192.0,
        pcie_bandwidth_mb_s: float = 16000.0,
        tensor_queue: list[tuple[int, float, float]] | None = None
    ) -> None:
        super().__init__()
        self.max_vram_mb = max_vram_mb
        self.pcie_bandwidth_mb_s = pcie_bandwidth_mb_s
        self.tensor_queue = tensor_queue if tensor_queue is not None else []

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Dict(
            {
                "vram_usage_mb": spaces.Box(low=0.0, high=self.max_vram_mb, shape=(), dtype=np.float32),
                "pcie_util": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "next_layer_mb": spaces.Box(low=0.0, high=2048.0, shape=(), dtype=np.float32),
                "next_layer_flops": spaces.Box(low=0.0, high=1e12, shape=(), dtype=np.float32),
            }
        )

        self.vram_usage_mb: float = 0.0
        self.pcie_util: float = 0.0
        self.current_idx: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed, options=options)
        self.vram_usage_mb = 0.0
        self.pcie_util = 0.0
        self.current_idx = 0
        self.terminated = False
        self.truncated = False

        next_layer_mb = 0.0
        next_layer_flops = 0.0
        if self.current_idx < len(self.tensor_queue):
            _, next_layer_mb, next_layer_flops = self.tensor_queue[self.current_idx]

        observation = {
            "vram_usage_mb": np.array(self.vram_usage_mb, dtype=np.float32),
            "pcie_util": np.array(self.pcie_util, dtype=np.float32),
            "next_layer_mb": np.array(next_layer_mb, dtype=np.float32),
            "next_layer_flops": np.array(next_layer_flops, dtype=np.float32),
        }
        info: dict = {}

        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if action not in {0, 1, 2}:
            print(f"Warning: Invalid action {action}. Clamping to 2 (offload to CPU).")
            action = 2

        if self.terminated or self.current_idx >= len(self.tensor_queue):
            self.terminated = True
            observation = {
                "vram_usage_mb": np.array(self.vram_usage_mb, dtype=np.float32),
                "pcie_util": np.array(self.pcie_util, dtype=np.float32),
                "next_layer_mb": np.array(0.0, dtype=np.float32),
                "next_layer_flops": np.array(0.0, dtype=np.float32),
            }
            return observation, 0.0, self.terminated, self.truncated, {}

        layer_id, memory_mb, flops = self.tensor_queue[self.current_idx]
        
        self.pcie_util = 0.0
        latency_ms = 0.0
        reward = 0.0
        oom = False

        if action == 0:
            if self.vram_usage_mb + memory_mb <= self.max_vram_mb:
                self.vram_usage_mb += memory_mb
                latency_ms = 0.0
                reward = 10.0
            else:
                self.terminated = True
                reward = -1000.0
                oom = True
                self.current_idx += 1
        elif action == 1:
            memory_mb_quant = memory_mb * 0.25
            if self.vram_usage_mb + memory_mb_quant <= self.max_vram_mb:
                self.vram_usage_mb += memory_mb_quant
                latency_ms = 5.0
                reward = 5.0
            else:
                self.terminated = True
                reward = -1000.0
                oom = True
                self.current_idx += 1
        elif action == 2:
            self.pcie_util = 0.9
            latency_ms = 50.0
            reward = -2.0

        if not oom:
            self.current_idx += 1
            if self.current_idx >= len(self.tensor_queue):
                self.terminated = True

        next_layer_mb = 0.0
        next_layer_flops = 0.0
        if self.current_idx < len(self.tensor_queue):
            _, next_layer_mb, next_layer_flops = self.tensor_queue[self.current_idx]

        observation = {
            "vram_usage_mb": np.array(self.vram_usage_mb, dtype=np.float32),
            "pcie_util": np.array(self.pcie_util, dtype=np.float32),
            "next_layer_mb": np.array(next_layer_mb, dtype=np.float32),
            "next_layer_flops": np.array(next_layer_flops, dtype=np.float32),
        }

        throughput = (flops / latency_ms) if latency_ms > 0 else flops
        info: dict = {
            "latency_ms": latency_ms,
            "throughput": throughput
        }

        return observation, reward, self.terminated, self.truncated, info
"""PPO settings; the defaults are those of the report's runs where they apply."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PPOConfig:
    # Rollouts: each simulates ``worlds`` scenes for a full episode. One update uses
    # ``rollouts_per_update`` rollouts, on every GPU. On a small GPU, use e.g.
    # --worlds 256 --rollouts-per-update 8 --store-on-host.
    worlds: int = 2048
    rollouts_per_update: int = 1
    epochs: int = 20  # passes over the training scenes
    # Training world modes as weights (log replay, reactive, self-play); see
    # training/mixing.py. The default is pure non-reactive training. Self-play worlds
    # add up to self_play_agents policy vehicles each, so lower ``worlds`` with them.
    log_worlds: float = 1.0
    reactive_worlds: float = 0.0
    self_play_worlds: float = 0.0
    self_play_agents: int = 10
    # Domain randomization of training episodes (training/randomization.py): reward
    # weights (observed: needs --model reward_conditions=true), blind actors, sizes, and
    # initial states of policy vehicles.
    domain_randomization: bool = False
    blind_probability: float = 0.1
    # Optimization: cosine learning-rate decay to min_learning_rate_fraction.
    learning_rate: float = 3e-4
    min_learning_rate_fraction: float = 0.05
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    optimization_epochs: int = 4
    minibatch_size: int = 20480  # transitions per GPU
    microbatch_size: int = 20480  # gradients accumulate over these within a minibatch
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    # Runtime.
    seed: int = 0
    amp: bool = True  # bf16 autocast for the network
    compile: bool = True
    # Keep collected samples in host memory (half precision) instead of on the GPU.
    store_on_host: bool = False
    # Load every scene into host memory once at startup; batches are then padded on the
    # GPU without file reads (mini: 9 GB, ~1 min). Turn off on machines with little RAM.
    cache_scenes: bool = True
    # Without the cache, worker processes load the next batch's scenes while the GPU
    # simulates. Ignored with the cache (workers would copy every scene).
    loader_workers: int = 4
    # Build the next training batch on a side CUDA stream during this one's rollout and
    # update (one more padded batch in GPU memory: about 16 GB at 2048 worlds).
    prefetch: bool = True
    # Host threads for PyTorch ops (0 keeps PyTorch's default, one per visible core,
    # which oversubscribes containers with a CPU quota below the visible core count).
    cpu_threads: int = 8
    # Evaluation batch per set: at most this many scenes (a smaller set runs in one batch).
    eval_worlds: int = 2048
    # Evaluation action: the Beta "mode" (the report) or "mean".
    eval_action: str = "mode"
    # Evaluate (and keep a checkpoint) every this many epochs, and after the last one.
    eval_interval: int = 1
    # If positive, evaluate every this many updates instead (checkpoint update_NNNNN.pt),
    # and after the last epoch: long epochs (e.g. self-play's 512 worlds) still get
    # evaluated as often as other runs.
    eval_updates: int = 0
    # Also score every evaluation set in reactive mode (IDM vehicles, rule-based
    # pedestrians), logged as NAME-reactive.
    eval_reactive: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

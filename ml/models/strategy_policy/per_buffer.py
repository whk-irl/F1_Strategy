"""Prioritized Experience Replay buffer for Stable-Baselines3 DQN.

Implements the proportional variant from Schaul et al. 2016
(arXiv:1511.05952).  Designed as a drop-in for SB3's ``ReplayBuffer``
via the ``replay_buffer_class`` parameter on ``DQN.__init__``.

Usage::

    from ml.models.strategy_policy.per_buffer import PrioritizedReplayBuffer

    model = DQN(
        ...,
        replay_buffer_class=PrioritizedReplayBuffer,
        replay_buffer_kwargs={"alpha": 0.6, "beta": 0.4},
    )
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


# ---------------------------------------------------------------------------
# Namedtuple returned by PrioritizedReplayBuffer.sample()
# ---------------------------------------------------------------------------


class PrioritizedReplayBufferSamples(NamedTuple):
    """Standard SB3 replay fields plus PER-specific metadata."""

    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor   # importance-sampling correction weights
    indices: np.ndarray  # buffer positions — required for update_priorities()


# ---------------------------------------------------------------------------
# Sum tree
# ---------------------------------------------------------------------------


class _SumTree:
    """Binary heap sum tree for O(log n) proportional priority sampling.

    Leaf i stores ``priority[i] ** alpha``; internal nodes store subtree sums.
    Index convention: root at 1, children of node k at 2k and 2k+1.
    Leaves occupy positions [capacity, 2*capacity).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # Tree array: index 0 unused; root at 1; leaves at [capacity, 2*capacity)
        self._tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Set priority for leaf ``leaf_idx`` and propagate the delta upward."""
        tree_idx = leaf_idx + self._capacity
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagate — walk up until root (idx=1)
        tree_idx >>= 1
        while tree_idx >= 1:
            self._tree[tree_idx] += delta
            tree_idx >>= 1

    def sample_leaf(self, value: float) -> int:
        """Return the leaf index whose prefix sum range contains ``value``."""
        idx = 1  # start at root
        while idx < self._capacity:
            left = 2 * idx
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        return idx - self._capacity

    def get_leaf_priority(self, leaf_idx: int) -> float:
        return float(self._tree[leaf_idx + self._capacity])

    @property
    def total(self) -> float:
        return float(self._tree[1])


# ---------------------------------------------------------------------------
# Prioritized replay buffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with proportional priority sampling (Schaul et al. 2016).

    New transitions enter with the maximum observed priority so they are
    always sampled at least once before their TD error is known.

    Args:
        buffer_size: Maximum number of transitions to store.
        observation_space: Env observation space.
        action_space: Env action space.
        device: PyTorch device for returned tensors.
        n_envs: Number of parallel environments.
        optimize_memory_usage: Passed through to SB3 base class.
        alpha: Priority exponent — 0 = uniform, 1 = full prioritization.
        beta: Initial IS-weight exponent.  Annealed linearly to 1.0 over
            ``beta_annealing_steps`` calls to :meth:`sample`.
        beta_annealing_steps: Steps over which *beta* is annealed to 1.0.
        epsilon: Small constant added to raw TD errors before priority
            assignment to prevent any transition from having zero priority.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 1_000_000,
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self._alpha = alpha
        self._beta_start = beta
        self._beta = beta
        self._beta_annealing_steps = beta_annealing_steps
        self._epsilon = epsilon
        self._max_priority: float = 1.0
        self._sample_count: int = 0
        self._sum_tree = _SumTree(self.buffer_size)

    # ------------------------------------------------------------------
    # SB3 ReplayBuffer overrides
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store transition and assign maximum priority so it is sampled soon."""
        idx = self.pos  # capture current write position before super() advances it
        super().add(obs, next_obs, action, reward, done, infos)
        self._sum_tree.update(idx, self._max_priority**self._alpha)

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
    ) -> PrioritizedReplayBufferSamples:  # type: ignore[override]
        """Sample a batch proportional to priorities.

        Returns :class:`PrioritizedReplayBufferSamples` instead of the
        standard :class:`~stable_baselines3.common.type_aliases.ReplayBufferSamples`
        so the caller can apply IS weights to the loss and push TD errors back
        via :meth:`update_priorities`.

        Args:
            batch_size: Number of transitions to sample.
            env: Optional VecNormalize wrapper for observation rescaling.

        Returns:
            Named tuple with standard fields plus ``weights`` and ``indices``.
        """
        assert self.full or self.pos > 0, "Cannot sample from an empty buffer."

        valid_size = self.buffer_size if self.full else self.pos
        total = self._sum_tree.total
        # Stratified sampling: divide [0, total) into batch_size equal segments.
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        raw_priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            idx = self._sum_tree.sample_leaf(value)
            idx = min(max(idx, 0), valid_size - 1)
            indices[i] = idx
            raw_priorities[i] = self._sum_tree.get_leaf_priority(idx)

        # Anneal beta linearly from beta_start → 1.0
        self._sample_count += 1
        self._beta = float(
            min(
                1.0,
                self._beta_start
                + (1.0 - self._beta_start) * self._sample_count / self._beta_annealing_steps,
            )
        )

        # Importance-sampling weights: w_i = (N * P(i))^{-beta}, normalised by max.
        probs = raw_priorities / total
        probs = np.clip(probs, 1e-10, None)  # guard against zero division
        weights = (valid_size * probs) ** (-self._beta)
        weights /= weights.max()

        data: ReplayBufferSamples = self._get_samples(indices, env=env)
        return PrioritizedReplayBufferSamples(
            observations=data.observations,
            actions=data.actions,
            next_observations=data.next_observations,
            dones=data.dones,
            rewards=data.rewards,
            weights=th.as_tensor(weights, dtype=th.float32, device=self.device),
            indices=indices,
        )

    # ------------------------------------------------------------------
    # Priority update (called by DQNPER after each gradient step)
    # ------------------------------------------------------------------

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities from fresh TD errors.

        Args:
            indices: Buffer positions returned by the last :meth:`sample` call.
            td_errors: Absolute TD errors (no gradient) for those positions,
                shape ``(batch_size,)``.
        """
        for idx, err in zip(indices, td_errors):
            priority = float(np.abs(err)) + self._epsilon
            self._sum_tree.update(int(idx), priority**self._alpha)
            if priority > self._max_priority:
                self._max_priority = priority

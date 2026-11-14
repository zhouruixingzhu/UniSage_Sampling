"""Self-contained Sifter trace sampler.

Sifter is the baseline introduced by Las-Casas, Papakerashvili, Anand and Mace
in "Sifter: Scalable Sampling for Distributed Traces, without Feature
Engineering" (SoCC 2019). The authors describe a TensorFlow reference
implementation but did not publish source code. This module reimplements
the algorithm directly from the paper using only numpy so that it can run
on the same macOS arm64 / Python 3.9 environment used for the rest of the
UniSage baselines and so that its behaviour stays auditable.

Defaults follow paper Section 6: N=5, P=10, k=50, learning_rate=0.01.
Per-trace loss is the mean cross-entropy of predicting the middle label of
each N-length path; sampling probability follows Section 5.3 verbatim.

The implementation is deliberately small. It exposes:

    Sifter.sample(labels) -> (decision, loss, probability)
    Sifter.score(labels)  -> loss (no update, no decision)

`labels` is the ordered list of N-length label sequences extracted from a
trace. A helper `extract_paths` builds those sequences from a UniSage
normalized span frame, walking root-to-leaf simple paths and padding short
paths with a dedicated PAD label, which matches the paper's "happened-before
path" semantics for span trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PAD_LABEL = "__PAD__"


def _root_id(span_ids: set, parent_ids: pd.Series) -> int | None:
    mask = parent_ids.str.lower().isin({"root", "nan", "none", ""}) | ~parent_ids.isin(span_ids)
    if mask.any():
        return int(parent_ids.index[mask][0])
    return None


def _label_for(service: str, operation: str) -> str:
    return f"{service}::{operation}"


def extract_paths(
    spans: pd.DataFrame,
    path_length: int = 5,
    pad_label: str = PAD_LABEL,
) -> List[List[str]]:
    """Return all root-to-leaf simple paths of length `path_length`.

    `spans` is the normalized per-trace span frame produced by
    `sampling_labels.normalize_trace_frame`. Each output path is a list
    of `path_length` label strings ordered along the happened-before chain
    from the trace root toward a descendant. Trees shallower than
    `path_length` are right-padded with `pad_label`.
    """
    if spans.empty or path_length <= 0:
        return []

    spans = spans.sort_values("Start")
    span_ids = set(spans["SpanID"].astype(str))
    parent_ids = spans["ParentID"].astype(str)

    children: Dict[str, List[str]] = {}
    by_id: Dict[str, dict] = {}
    for row in spans.itertuples(index=False):
        sid = str(row.SpanID)
        pid = str(row.ParentID)
        label = _label_for(str(row.Service), str(row.Operation))
        by_id[sid] = {"label": label}
        if pid in span_ids and pid != sid:
            children.setdefault(pid, []).append(sid)

    root_mask = parent_ids.str.lower().isin({"root", "nan", "none", ""}) | ~parent_ids.isin(span_ids)
    if root_mask.any():
        root_indices = spans.index[root_mask].tolist()
        roots = [str(spans.at[i, "SpanID"]) for i in root_indices]
    else:
        roots = [str(spans.iloc[0]["SpanID"])]

    paths: List[List[str]] = []

    def walk(node: str, stack: List[str]) -> None:
        stack = stack + [by_id[node]["label"]]
        kids = children.get(node, [])
        # Emit a path each time the window first reaches `path_length`,
        # and continue walking so deeper trees yield overlapping windows
        # along the happened-before chain.
        if len(stack) >= path_length:
            paths.append(list(stack[-path_length:]))
        if not kids:
            if len(stack) < path_length:
                padded = stack + [pad_label] * (path_length - len(stack))
                paths.append(padded)
            return
        for child in kids:
            walk(child, stack)

    for root in roots:
        if root in by_id:
            walk(root, [])

    return paths


@dataclass
class SifterConfig:
    path_length: int = 5
    embedding_dim: int = 10
    window_size: int = 50
    learning_rate: float = 0.01
    pad_label: str = PAD_LABEL
    seed: int = 1

    @property
    def context_size(self) -> int:
        # Sifter predicts the middle label given the (N-1) surrounding ones.
        return self.path_length - 1


class Sifter:
    """Per-paper Sifter implementation: 2-layer label-prediction model + biased sampling."""

    def __init__(self, alpha: float, config: SifterConfig | None = None):
        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if config is None:
            config = SifterConfig()
        if config.path_length < 2:
            raise ValueError("path_length must be >= 2 so a middle label exists")
        self.alpha = float(alpha)
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._label_to_idx: Dict[str, int] = {}
        # Embedding: |Σ| × P. Output projection: ((N-1)·P) × |Σ|.
        self._embedding = np.empty((0, config.embedding_dim), dtype=np.float64)
        self._output_w = np.empty((config.context_size * config.embedding_dim, 0), dtype=np.float64)
        self._loss_window: List[float] = []

    # ----- vocabulary -----

    def _ensure_label(self, label: str) -> int:
        idx = self._label_to_idx.get(label)
        if idx is not None:
            return idx
        idx = len(self._label_to_idx)
        self._label_to_idx[label] = idx
        # Grow embedding and output rows/cols.
        new_emb_row = self.rng.standard_normal(self.cfg.embedding_dim) * 0.05
        self._embedding = np.vstack([self._embedding, new_emb_row])
        new_out_col = self.rng.standard_normal(self.cfg.context_size * self.cfg.embedding_dim) * 0.05
        if self._output_w.size == 0:
            self._output_w = new_out_col.reshape(-1, 1)
        else:
            self._output_w = np.hstack([self._output_w, new_out_col.reshape(-1, 1)])
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self._label_to_idx)

    # ----- per-path forward/backward -----

    def _path_to_indices(self, labels: Sequence[str]) -> Tuple[np.ndarray, int]:
        """Return (context_indices [N-1], target_index) for a single path."""
        if len(labels) != self.cfg.path_length:
            raise ValueError(
                f"path length {len(labels)} != configured N={self.cfg.path_length}"
            )
        mid = self.cfg.path_length // 2
        target = self._ensure_label(labels[mid])
        context_labels = [labels[i] for i in range(self.cfg.path_length) if i != mid]
        context = np.fromiter(
            (self._ensure_label(label) for label in context_labels),
            dtype=np.int64,
            count=self.cfg.context_size,
        )
        return context, target

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        ez = np.exp(z)
        return ez / ez.sum()

    def _path_forward(self, context: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray, float]:
        ctx_emb = self._embedding[context]  # shape (N-1, P)
        flat = ctx_emb.reshape(-1)  # shape ((N-1)·P,)
        logits = flat @ self._output_w  # shape (|Σ|,)
        probs = self._softmax(logits)
        loss = -float(np.log(probs[target] + 1e-12))
        return flat, probs, loss

    def _path_backward(self, context: np.ndarray, target: int, flat: np.ndarray, probs: np.ndarray) -> None:
        # Cross-entropy gradient at logits = probs - one_hot(target)
        d_logits = probs.copy()
        d_logits[target] -= 1.0
        # Gradient on output_w: outer(flat, d_logits)
        grad_out = np.outer(flat, d_logits)
        # Gradient on flat: output_w @ d_logits
        grad_flat = self._output_w @ d_logits
        grad_emb = grad_flat.reshape(self.cfg.context_size, self.cfg.embedding_dim)
        lr = self.cfg.learning_rate
        self._output_w -= lr * grad_out
        # Accumulate updates onto context embeddings (an index can repeat).
        for i, idx in enumerate(context):
            self._embedding[idx] -= lr * grad_emb[i]

    # ----- trace-level API -----

    def _register_trace_labels(self, paths: Sequence[Sequence[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-register every label that appears in `paths` and return the
        index tensors used by the batched forward/backward.

        Returns (contexts, targets) with shapes (P_n, N-1) and (P_n,) where
        P_n = number of paths. Vocabulary growth happens entirely before any
        matrix op, so the embedding / output_w shapes stay consistent for the
        rest of the trace.
        """
        n_paths = len(paths)
        mid = self.cfg.path_length // 2
        contexts = np.empty((n_paths, self.cfg.context_size), dtype=np.int64)
        targets = np.empty((n_paths,), dtype=np.int64)
        for pi, path in enumerate(paths):
            if len(path) != self.cfg.path_length:
                raise ValueError(
                    f"path length {len(path)} != configured N={self.cfg.path_length}"
                )
            targets[pi] = self._ensure_label(path[mid])
            ci = 0
            for li, label in enumerate(path):
                if li == mid:
                    continue
                contexts[pi, ci] = self._ensure_label(label)
                ci += 1
        return contexts, targets

    def _trace_loss(self, paths: Sequence[Sequence[str]], update: bool) -> float:
        """Per-trace batched forward (and optional backward).

        Vectorized over all paths within a single trace. The paper §5.4
        describes one backpropagation pass per trace; this implementation
        matches that semantics: all paths' contributions are pooled into one
        gradient and applied once. Vocabulary grows before any matmul.
        """
        if not paths:
            return 0.0
        contexts, targets = self._register_trace_labels(paths)
        n_paths = contexts.shape[0]
        # Forward: (P_n, N-1, D) → (P_n, (N-1)·D) → (P_n, V).
        ctx_emb = self._embedding[contexts]
        flat = ctx_emb.reshape(n_paths, -1)
        logits = flat @ self._output_w
        # Guard against non-finite values from extreme path counts.
        if not np.isfinite(logits).all():
            return float(np.log(self.vocab_size + 1e-12))
        # Numerically stable row-wise softmax.
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        probs = logits  # (P_n, V)
        # Per-path cross-entropy on the target column.
        target_probs = probs[np.arange(n_paths), targets]
        losses = -np.log(target_probs + 1e-12)
        loss = float(losses.mean())

        if update:
            # d_logits = probs - one_hot(targets); shape (P_n, V).
            probs[np.arange(n_paths), targets] -= 1.0
            d_logits = probs
            # Gradients (sum over paths so one update absorbs the full trace).
            grad_output = flat.T @ d_logits  # ((N-1)·D, V)
            grad_flat = d_logits @ self._output_w.T  # (P_n, (N-1)·D)
            grad_emb_per_pos = grad_flat.reshape(n_paths, self.cfg.context_size, self.cfg.embedding_dim)
            lr = self.cfg.learning_rate
            self._output_w -= lr * grad_output
            # Embedding update: an index can repeat across (path, position).
            flat_idx = contexts.reshape(-1)
            flat_grad = grad_emb_per_pos.reshape(-1, self.cfg.embedding_dim)
            np.add.at(self._embedding, flat_idx, -lr * flat_grad)
        return loss

    def score(self, paths: Sequence[Sequence[str]]) -> float:
        """Compute mean per-path cross-entropy without updating the model."""
        return self._trace_loss(paths, update=False)

    def sample(self, paths: Sequence[Sequence[str]]) -> Tuple[bool, float, float]:
        """Run Sifter's per-trace step.

        Returns (decision, loss, probability). The model is updated and the
        loss window is advanced on every call, exactly as in the paper —
        regardless of whether the trace is sampled.
        """
        loss = self._trace_loss(paths, update=True)
        prob = self._sampling_probability(loss)
        decision = bool(self.rng.random() < prob)
        # Window bookkeeping after probability calc (paper §5.4).
        self._loss_window.append(loss)
        if len(self._loss_window) > self.cfg.window_size:
            self._loss_window.pop(0)
        return decision, loss, prob

    def _sampling_probability(self, current_loss: float) -> float:
        window = self._loss_window
        if not window:
            # Bootstrap: until we have any history Sifter samples uniformly at α.
            return self.alpha
        losses = window + [current_loss]
        n = len(losses)
        min_loss = min(losses)
        weights = [l - min_loss for l in losses]
        total = sum(weights)
        if total <= 0:
            return self.alpha
        return float(weights[-1] / total * n * self.alpha)


# ----- public helpers -----


def trace_order_index(spans: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per TraceID sorted by its earliest Start.

    Matches the implicit assumption in Sifter that traces arrive in temporal
    order at the tracing backend.
    """
    if spans.empty:
        return pd.DataFrame(columns=["TraceID", "Start"])
    order = (
        spans.groupby("TraceID", sort=False)["Start"].min().reset_index().sort_values(["Start", "TraceID"]).reset_index(drop=True)
    )
    return order


def run_sifter_over_trace_frame(
    spans: pd.DataFrame,
    alpha: float,
    config: SifterConfig | None = None,
) -> pd.DataFrame:
    """Run Sifter across all traces in `spans` and return per-trace results.

    `spans` is the normalized span frame for one case (one row per span,
    columns include TraceID, SpanID, ParentID, Service, Operation, Start).
    Output columns: TraceID, SifterLoss, SifterProbability, SifterDecision,
    SifterNumPaths.
    """
    if spans.empty:
        return pd.DataFrame(
            columns=["TraceID", "SifterLoss", "SifterProbability", "SifterDecision", "SifterNumPaths"]
        )

    cfg = config or SifterConfig()
    sifter = Sifter(alpha=alpha, config=cfg)
    order = trace_order_index(spans)
    rows = []
    for trace_id in order["TraceID"]:
        trace_spans = spans[spans["TraceID"] == trace_id]
        paths = extract_paths(trace_spans, path_length=cfg.path_length, pad_label=cfg.pad_label)
        if paths:
            decision, loss, prob = sifter.sample(paths)
            n_paths = len(paths)
        else:
            decision, loss, prob, n_paths = False, float("nan"), float("nan"), 0
        rows.append(
            {
                "TraceID": str(trace_id),
                "SifterLoss": loss,
                "SifterProbability": prob,
                "SifterDecision": decision,
                "SifterNumPaths": n_paths,
            }
        )
    return pd.DataFrame(rows)


def top_k_decision_by_loss(
    frame: pd.DataFrame,
    target_count: int,
    score_column: str = "SifterLoss",
) -> pd.Series:
    """Pick the top `target_count` TraceIDs by `score_column` (NaN-aware)."""
    if target_count <= 0 or frame.empty:
        return pd.Series(False, index=frame.index)
    ranked = frame.sort_values([score_column, "TraceID"], ascending=[False, True], na_position="last")
    keep = ranked.head(min(target_count, len(ranked))).index
    decision = pd.Series(False, index=frame.index)
    decision.loc[keep] = True
    return decision


def smoke_self_test() -> bool:
    """Tiny synthetic self-test: outlier path label must rank above common ones."""
    rng = np.random.default_rng(0)
    common_paths = [["A::x", "B::y", "C::z", "B::y", "A::x"] for _ in range(80)]
    outlier_paths = [["A::x", "B::y", "Q::rare", "B::y", "A::x"] for _ in range(5)]

    cfg = SifterConfig(path_length=5, embedding_dim=4, window_size=30, seed=0)
    sifter = Sifter(alpha=0.1, config=cfg)
    common_losses = []
    outlier_losses = []
    # Interleave to mimic a streaming workload.
    schedule = []
    for i in range(80):
        schedule.append((common_paths[i], "common"))
        if i in {20, 40, 60}:
            schedule.append((outlier_paths[i % len(outlier_paths)], "outlier"))
    rng.shuffle(schedule)
    for paths, tag in schedule:
        _, loss, _ = sifter.sample([paths])
        (outlier_losses if tag == "outlier" else common_losses).append(loss)
    return float(np.mean(outlier_losses[-1:])) > float(np.mean(common_losses[-10:]))


if __name__ == "__main__":
    ok = smoke_self_test()
    print("smoke_self_test:", "PASS" if ok else "FAIL")

"""GPU VRAM management & model offloading hooks for the Autonomous Remix Studio.

The studio chains several heavy models (BS-RoFormer, WhisperX, MusicGen,
Stable Audio, Seed-VC) that together far exceed a single consumer GPU.
The discipline is sequential load/unload: one model resident at a time,
``torch.cuda.empty_cache()`` between swaps, and a VRAM budget check before
any load so a 16 GB card (Colab T4 / RTX 4060 class) never OOMs mid-run.

Everything in this module is CPU-safe: ``torch`` is an optional import.
On a machine without torch (or without CUDA) the helpers degrade to
logged no-ops instead of crashing the import.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

__all__ = [
    "MODEL_VRAM_GB",
    "get_device",
    "get_vram_budget_gb",
    "model_session",
    "ensure_fits",
    "sequential_load_plan",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torch import — the whole module must import cleanly on CPU-only
# machines (CI, laptops without CUDA, the studio's DSP-only test venv).
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import-time environment branch
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time environment branch
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


#: Honest peak-VRAM estimates (weights + a working activation buffer) for
#: the models the studio orchestrates, in GiB. These are planning numbers
#: for a 16 GB consumer GPU, not vendor spec-sheet figures:
#:   * BS-RoFormer (vocal/drum/bass separation) is the hungriest stem model.
#:   * WhisperX large-v3 with alignment runs ~2 GiB in fp16.
#:   * MusicGen-Melody (1.5 B) and Stable Audio Open (1.1 B) are ~3-4 GiB.
#:   * Seed-VC / RVC voice conversion is a small UNet + vocoder, ~2 GiB.
#:   * Basic Pitch is a tiny onsets/frames model, ~0.5 GiB.
#:   * Matchering is pure CPU DSP (no GPU memory at all).
MODEL_VRAM_GB: dict[str, float] = {
    "BS-RoFormer": 4.0,
    "WhisperX": 2.0,
    "MusicGen-Melody": 3.0,
    "Stable Audio Open": 4.0,
    "Seed-VC": 2.0,
    "Basic Pitch": 0.5,
    "Matchering": 0.0,
}

#: Reserve this much VRAM for the framework / activations / fragmentation so
#: a plan that exactly fills the card still has headroom.
VRAM_HEADROOM_GB: float = 1.0


def get_device() -> str:
    """Return ``"cuda"`` when a CUDA GPU is usable, else ``"cpu"``.

    Never raises on a torch-less install: no torch means no CUDA.
    """
    if not _TORCH_AVAILABLE or torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - defensive: CUDA init can fail oddly
        logger.warning("torch.cuda.is_available() raised; falling back to CPU", exc_info=True)
        return "cpu"


def get_vram_budget_gb() -> float:
    """Total GPU memory in GiB, or ``0.0`` when there is no usable GPU.

    Uses ``torch.cuda.get_device_properties`` so the number is the card's
    real total memory, not a free-memory snapshot (free memory shrinks as
    other processes allocate).
    """
    if get_device() != "cuda":
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:  # pragma: no cover - defensive
        logger.warning("could not read GPU memory; assuming 0 GiB budget", exc_info=True)
        return 0.0


def _check_model_names(model_names: list[str]) -> None:
    """Raise ``ValueError`` naming any model absent from the VRAM registry."""
    unknown = [name for name in model_names if name not in MODEL_VRAM_GB]
    if unknown:
        known = ", ".join(sorted(MODEL_VRAM_GB))
        raise ValueError(f"unknown model(s): {unknown}; known models: {known}")


@contextmanager
def model_session(name: str) -> Iterator[None]:
    """Context manager for one heavy model's resident window.

    Logs the load, yields control to the caller (which does the actual
    model load + inference inside the block), then on exit logs the unload
    and calls ``torch.cuda.empty_cache()`` when CUDA is present so the next
    model starts from a clean slate. On CPU it is a logged no-op.

    Usage::

        with model_session("BS-RoFormer"):
            stems = separate_stems(audio)   # loads + runs + frees the model
    """
    _check_model_names([name])
    logger.info("loading model %s (est. %.1f GiB VRAM)", name, MODEL_VRAM_GB[name])
    try:
        yield
    finally:
        if get_device() == "cuda" and torch is not None:
            try:
                torch.cuda.empty_cache()
                logger.info("unloaded model %s; CUDA cache emptied", name)
            except Exception:  # pragma: no cover - defensive
                logger.warning("empty_cache() failed after %s", name, exc_info=True)
        else:
            logger.info("unloaded model %s (CPU session, nothing to free)", name)


def ensure_fits(*model_names: str) -> bool:
    """Return ``True`` when the named models fit in the VRAM budget together.

    The check is ``sum(estimates) + headroom <= budget``. On a CPU-only
    machine there is no VRAM to exceed, so it always returns ``True`` and
    logs that the run will be slow instead. Unknown model names raise
    ``ValueError``.
    """
    names = list(model_names)
    _check_model_names(names)
    needed = sum(MODEL_VRAM_GB[name] for name in names)
    if get_device() != "cuda":
        logger.info(
            "CPU-only run: no VRAM limit for %s (%.1f GiB est.); "
            "expect heavy models to be slow",
            names,
            needed,
        )
        return True
    budget = get_vram_budget_gb()
    fits = needed + VRAM_HEADROOM_GB <= budget
    logger.info(
        "VRAM check for %s: need %.1f GiB + %.1f GiB headroom vs %.1f GiB budget -> %s",
        names,
        needed,
        VRAM_HEADROOM_GB,
        budget,
        "fits" if fits else "DOES NOT FIT",
    )
    return fits


def sequential_load_plan(model_names: list[str]) -> list[str]:
    """Return the sequential load order for the named models.

    Sorted by VRAM estimate descending (largest first). With the
    load-one/unload-one discipline, peak usage is the single largest
    model, so running the largest first also surfaces an OOM on the
    first step instead of halfway through a pipeline. Ties keep the
    caller's original order (stable sort).
    """
    _check_model_names(model_names)
    return sorted(model_names, key=lambda name: MODEL_VRAM_GB[name], reverse=True)

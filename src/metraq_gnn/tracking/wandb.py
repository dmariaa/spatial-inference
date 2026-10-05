from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import wandb


class WandbLogger:
    """Small adapter exposing only the W&B operations used by ``GNNTrainer``."""

    def __init__(
        self,
        *,
        project: str,
        config: Mapping[str, Any],
        entity: str | None = None,
        name: str | None = None,
        tags: Sequence[str] | None = None,
        mode: Literal["online", "offline", "disabled"] | None = None,
    ) -> None:
        if not project.strip():
            raise ValueError("project must be a non-empty string")
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=list(tags) if tags is not None else None,
            config=dict(config),
            mode=mode,
        )
        if self.run is None:
            raise RuntimeError("wandb.init() did not create a run")

    @property
    def url(self) -> str | None:
        return self.run.url

    def log_epoch(self, metrics: Mapping[str, float], *, step: int) -> None:
        self.run.log(dict(metrics), step=step)

    def log_summary(self, summary: Mapping[str, float | int]) -> None:
        self.run.summary.update(dict(summary))

    def log_checkpoint(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        artifact = wandb.Artifact(name=f"{self.run.id}-best-model", type="model")
        artifact.add_file(str(checkpoint))
        self.run.log_artifact(artifact, aliases=["best"])

    def finish(self, *, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish(exit_code=1 if exception_type is not None else 0)


__all__ = ["WandbLogger"]


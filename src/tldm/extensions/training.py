from collections.abc import Iterable, Iterator
from operator import length_hint
from typing import Any, Generic, Self, TypeVar

from tldm.std import TldmTypeError, tldm

T = TypeVar("T")


class TrainingTldm(Generic[T]):
    """Training-oriented wrapper around nested epoch and step tldm bars."""

    def __init__(
        self,
        epochs: int,
        steps_per_epoch: int | float | None = None,
        *,
        desc: str = "train",
        epoch_desc: str = "epoch",
        step_desc: str = "step",
        position: int = 0,
        tldm_class: type[tldm] = tldm,
        **tldm_kwargs: Any,
    ) -> None:
        self.epochs_total = epochs
        self.steps_per_epoch = steps_per_epoch
        self.desc = desc
        self.epoch_desc = epoch_desc
        self.step_desc = step_desc
        self.position = position
        self.tldm_class = tldm_class
        self.tldm_kwargs = dict(tldm_kwargs)
        self.epoch_bar: tldm[int] | None = None
        self.step_bar: tldm[T] | None = None
        self.current_epoch = 0

    def __enter__(self) -> Self:
        epoch_kwargs = dict(self.tldm_kwargs)
        epoch_kwargs.setdefault("position", self.position)
        epoch_kwargs.setdefault("leave", True)
        self.epoch_bar = self.tldm_class(
            total=self.epochs_total,
            desc=self.desc,
            **epoch_kwargs,
        )
        self.epoch_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.step_bar is not None:
            self.step_bar.__exit__(exc_type, exc_value, traceback)
            self.step_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.__exit__(exc_type, exc_value, traceback)
            self.epoch_bar = None

    def _require_epoch_bar(self) -> tldm[int]:
        if self.epoch_bar is None:
            raise TldmTypeError("training_tldm must be used as a context manager")
        return self.epoch_bar

    def _active_bar(self) -> tldm[Any]:
        if self.step_bar is not None:
            return self.step_bar
        return self._require_epoch_bar()

    def epochs(self) -> Iterator[int]:
        epoch_bar = self._require_epoch_bar()
        for epoch_index in range(self.epochs_total):
            self.current_epoch = epoch_index + 1
            epoch_bar.set_description_str(f"{self.desc} {self.current_epoch}/{self.epochs_total}")
            yield epoch_index
            epoch_bar.update()

    def steps(self, iterable: Iterable[T], **kwargs: Any) -> Iterator[T]:
        self._require_epoch_bar()
        if self.step_bar is not None:
            self.step_bar.close()
            self.step_bar = None

        step_kwargs = dict(self.tldm_kwargs)
        step_kwargs.update(kwargs)
        step_kwargs.setdefault("position", self.position + 1)
        step_kwargs.setdefault("leave", False)
        step_kwargs.setdefault("total", self.steps_per_epoch)

        if step_kwargs.get("total") is None:
            try:
                step_kwargs["total"] = length_hint(iterable)
            except TypeError:
                step_kwargs["total"] = None

        default_desc = self.step_desc
        if self.current_epoch:
            default_desc = f"{self.epoch_desc} {self.current_epoch}/{self.epochs_total}"
        step_kwargs.setdefault("desc", default_desc)

        self.step_bar = self.tldm_class(iterable, **step_kwargs)
        step_bar = self.step_bar

        def _iterate_steps() -> Iterator[T]:
            try:
                with step_bar as active_step_bar:
                    yield from active_step_bar
            finally:
                self.step_bar = None

        return _iterate_steps()

    def set_metrics(self, *args: Any, **kwargs: Any) -> None:
        self._active_bar().set_metrics(*args, **kwargs)

    def set_throughput(self, *args: Any, **kwargs: Any) -> None:
        self._active_bar().set_throughput(*args, **kwargs)

    def mark(self, *args: Any, **kwargs: Any) -> None:
        self._active_bar().mark(*args, **kwargs)

    def section(self, *args: Any, **kwargs: Any):
        return self._active_bar().section(*args, **kwargs)


def training_tldm(
    epochs: int,
    steps_per_epoch: int | float | None = None,
    **kwargs: Any,
) -> TrainingTldm[Any]:
    """Create a training-oriented nested progress helper."""
    return TrainingTldm(epochs, steps_per_epoch=steps_per_epoch, **kwargs)

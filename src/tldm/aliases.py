import itertools
import sys
from collections.abc import Callable, Iterable, Iterator
from operator import length_hint
from typing import Any, TypeVar

from .std import tldm

__all__ = [
    "tenumerate",
    "tzip",
    "tmap",
    "tproduct",
    "trange",
    "auto_tldm",
]


T = TypeVar("T")
R = TypeVar("R")


# Auto-detection of notebook/IPython environment
def _get_auto_tldm() -> type[tldm]:
    """
    Automatically choose between `tldm.notebook` and `tldm.std`.

    Returns
    -------
    type[tldm]
        The appropriate tldm class based on the current environment.
    """
    try:
        # Try to detect IPython/Jupyter notebook environment
        get_ipython = sys.modules["IPython"].get_ipython  # type: ignore[attr-defined]
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")

        # Check if ipywidgets is available
        from warnings import warn

        from .notebook import WARN_NOIPYW
        from .std import TldmWarning

        try:
            from .notebook import IProgress  # type: ignore[attr-defined]
        except ImportError:
            IProgress = None

        if IProgress is None:
            warn(WARN_NOIPYW, TldmWarning, stacklevel=2)
            raise ImportError("ipywidgets")

        # Use notebook version
        from .notebook import tldm as notebook_tldm

        return notebook_tldm  # type: ignore[return-value]
    except Exception:
        # Fallback to standard tldm
        return tldm


# Create auto alias - automatically selects notebook or standard tldm
auto_tldm = _get_auto_tldm()


def tenumerate(
    iterable: Iterable[T],
    start: int = 0,
    total: int | float | None = None,
    tldm_class: type[tldm] | None = None,
    **tldm_kwargs: Any,
) -> Iterator[tuple[int, T]]:
    """
    Equivalent of builtin `enumerate`.

    Parameters
    ----------
    tldm_class  : [default: auto_tldm (automatically detected)].
    """
    if tldm_class is None:
        tldm_class = auto_tldm
    return enumerate(tldm_class(iterable, total=total, **tldm_kwargs), start)


def tzip(
    iter1: Iterable[T], *iter2plus: Iterable[Any], **tldm_kwargs: Any
) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tldm_class  : [default: auto_tldm (automatically detected)].
    """
    kwargs = tldm_kwargs.copy()
    tldm_class = kwargs.pop("tldm_class", None)
    if tldm_class is None:
        tldm_class = auto_tldm
    yield from zip(tldm_class(iter1, **kwargs), *iter2plus)


def tmap(function: Callable[..., R], *sequences: Iterable[Any], **tldm_kwargs: Any) -> Iterator[R]:
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tldm_class  : [default: auto_tldm (automatically detected)].
    """
    for i in tzip(*sequences, **tldm_kwargs):
        yield function(*i)


def tproduct(*iterables: Iterable[T], **tldm_kwargs: Any) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of `itertools.product`.

    Parameters
    ----------
    tldm_class  : [default: auto_tldm (automatically detected)].
    """
    kwargs = tldm_kwargs.copy()
    repeat = kwargs.pop("repeat", 1)
    tldm_class = kwargs.pop("tldm_class", None)
    if tldm_class is None:
        tldm_class = auto_tldm
    try:
        lens = list(map(length_hint, iterables))
    except TypeError:
        total = None
    else:
        total = 1
        for i in lens:
            total *= i
        total = total**repeat
        kwargs.setdefault("total", total)
    with tldm_class(**kwargs) as t:
        it = itertools.product(*iterables, repeat=repeat)
        for val in it:
            yield val
            t.update()


def trange(*args: int, **kwargs: Any) -> tldm:
    """Shortcut for tldm(range(*args), **kwargs)."""
    return auto_tldm(range(*args), **kwargs)

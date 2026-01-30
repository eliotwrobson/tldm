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
    "tbatched",
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
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")

        # Check if ipywidgets is available
        from warnings import warn

        from .notebook import WARN_NOIPYW
        from .std import TldmWarning

        try:
            from .notebook import IProgress
        except ImportError:
            IProgress = None

        if IProgress is None:
            warn(WARN_NOIPYW, TldmWarning, stacklevel=2)
            raise ImportError("ipywidgets")

        # Use notebook version
        from .notebook import tldm as notebook_tldm

        return notebook_tldm
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


def tbatched(
    iterable: Iterable[T],
    n: int,
    *,
    strict: bool = False,
    total: int | float | None = None,
    tldm_class: type[tldm] | None = None,
    **tldm_kwargs: Any,
) -> Iterator[tuple[T, ...]]:
    """
    Equivalent of `itertools.batched` (Python 3.12+) with progress bar.

    Yields tuples of up to `n` elements from the iterable, with progress
    tracked per batch rather than per item.

    Parameters
    ----------
    iterable : Iterable[T]
        The iterable to batch.
    n : int
        The batch size (maximum number of elements per batch).
    strict : bool, optional
        If True, raises ValueError if the final batch has fewer than n elements.
        Requires Python 3.13+. Default: False.
    total : int | float | None, optional
        Total number of items in the iterable (used to calculate number of batches).
        If not provided and iterable has __len__, it will be used.
    tldm_class : type[tldm] | None, optional
        The tldm class to use. Default: auto_tldm (automatically detected).
    **tldm_kwargs : Any
        Additional keyword arguments to pass to tldm.

    Yields
    ------
    tuple[T, ...]
        Tuples of up to n elements from the iterable.

    Examples
    --------
    >>> from tldm import tbatched
    >>> for batch in tbatched(range(10), 3):
    ...     print(batch)
    (0, 1, 2)
    (3, 4, 5)
    (6, 7, 8)
    (9,)

    See Also
    --------
    itertools.batched : The standard library equivalent (Python 3.12+).
    https://github.com/tqdm/tqdm/issues/1615 : Original feature request.
    """
    import sys

    if sys.version_info < (3, 12):
        raise ImportError("tbatched requires Python 3.12+ (itertools.batched)")

    tldm_class = tldm_class or auto_tldm

    # Calculate total number of batches
    if total is None:
        try:
            total_items = length_hint(iterable)
            total_batches = (total_items + n - 1) // n if total_items else None
        except TypeError:
            total_batches = None
    else:
        total_batches = (int(total) + n - 1) // n

    tldm_kwargs.setdefault("total", total_batches)

    # Use itertools.batched with appropriate arguments
    if sys.version_info >= (3, 13):
        batched_iter = itertools.batched(iterable, n, strict=strict)
    elif strict:
        raise ValueError("strict parameter requires Python 3.13+")
    else:
        batched_iter = itertools.batched(iterable, n)

    with tldm_class(**tldm_kwargs) as t:
        for batch in batched_iter:
            yield batch
            t.update()


def trange(*args: int, **kwargs: Any) -> tldm:
    """Shortcut for tldm(range(*args), **kwargs)."""
    return auto_tldm(range(*args), **kwargs)

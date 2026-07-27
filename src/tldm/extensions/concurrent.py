"""
Thin wrappers around `concurrent.futures`.
"""

from contextlib import contextmanager, nullcontext
from operator import length_hint

from ..utils import TldmWarning


@contextmanager
def ensure_lock(tldm_class, lock_name=""):
    """get (create if necessary) and then restore `tldm_class`'s lock"""
    old_lock = getattr(tldm_class, "_lock", None)  # don't create a new lock
    lock = old_lock or tldm_class.get_lock()  # maybe create a new lock
    lock = getattr(lock, lock_name, lock)  # maybe subtype
    tldm_class.set_lock(lock)
    yield lock
    if old_lock is None:
        del tldm_class._lock
    else:
        tldm_class.set_lock(old_lock)


def _min_map_len(iterables) -> int | None:
    """Return the shortest known iterable length, or None if unknown."""
    lengths = [n for it in iterables if (n := length_hint(it, -1)) >= 0]
    return min(lengths) if lengths else None


def _executor_map(PoolExecutor, fn, *iterables, **tldm_kwargs):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    tldm_class  : [default: tldm.auto.tldm].
    max_workers  : [default: None].
    timeout  : [default: None].
    chunksize  : [default: 1].
    buffersize  : [default: None]. Requires Python>=3.14.
    thread_name_prefix  : str
    max_tasks_per_child  : int
    mp_context  : object
    lock_name  : [default: "":str].
    """
    kwargs = tldm_kwargs.copy()
    if "total" not in kwargs:
        min_len = _min_map_len(iterables)
        if min_len is not None:
            kwargs["total"] = min_len
    tldm_class = kwargs.pop("tldm_class", None)
    if tldm_class is None:
        from ..std import tldm as tldm_class
    max_workers = kwargs.pop("max_workers", None)
    timeout = kwargs.pop("timeout", None)
    chunksize = kwargs.pop("chunksize", 1)
    lock_name = kwargs.pop("lock_name", "")
    map_kwargs = {}
    if "buffersize" in kwargs:
        map_kwargs["buffersize"] = kwargs.pop("buffersize")
    pool_kwargs = {}
    for key in ("thread_name_prefix", "max_tasks_per_child", "mp_context"):
        if key in kwargs:
            pool_kwargs[key] = kwargs.pop(key)
    with ensure_lock(tldm_class, lock_name=lock_name) as lk:
        # share lock in case workers are already using `tldm`
        with PoolExecutor(
            max_workers=max_workers,
            initializer=tldm_class.set_lock,
            initargs=(lk,),
            **pool_kwargs,
        ) as ex:
            pbar = tldm_class(**kwargs)
            cm = (
                pbar
                if hasattr(pbar, "__enter__") and hasattr(pbar, "__exit__")
                else nullcontext(pbar)
            )
            with cm as pbar:
                if hasattr(pbar, "update"):
                    orig_submit = ex.submit

                    def patch_submit(*args, **inner_kwargs):
                        fut = orig_submit(*args, **inner_kwargs)
                        fut.add_done_callback(lambda _: pbar.update())
                        return fut

                    ex.submit = patch_submit
                    return list(
                        ex.map(
                            fn,
                            *iterables,
                            timeout=timeout,
                            chunksize=chunksize,
                            **map_kwargs,
                        )
                    )

                # Backward-compatible path for lightweight wrappers that only
                # accept/iterate an iterable and do not expose `update`.
                return list(
                    tldm_class(
                        ex.map(
                            fn,
                            *iterables,
                            timeout=timeout,
                            chunksize=chunksize,
                            **map_kwargs,
                        ),
                        **kwargs,
                    )
                )


def thread_map(fn, *iterables, **tldm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ThreadPoolExecutor.__init__`.
    thread_name_prefix : str, optional
        Passed to `concurrent.futures.ThreadPoolExecutor.__init__`.
    timeout  : int or float, optional
        The iterator raises a TimeoutError if __next()__ is called and the
        result isn't available within the timeout specified from the
        original call to thread_map. [default: None].
    buffersize  : int, optional
        Maximum number of submitted tasks whose results are not yet yielded.
        Requires Python>=3.14.
    tldm_class  : optional
        `tldm` class to use for bars [default: tldm.auto.tldm].
    """
    from concurrent.futures import ThreadPoolExecutor

    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tldm_kwargs)


def process_map(fn, *iterables, **tldm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
    timeout  : int or float, optional
        The iterator raises a TimeoutError if __next()__ is called and the
        result isn't available within the timeout specified from the
        original call to process_map. [default: None].
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    buffersize  : int, optional
        Maximum number of submitted tasks whose results are not yet yielded.
        Requires Python>=3.14.
    max_tasks_per_child  : int, optional
        Maximum number of tasks a worker can process before replacement.
    mp_context  : multiprocessing.BaseContext, optional
        Multiprocessing context used by ProcessPoolExecutor.
    lock_name  : str, optional
        Member of `tldm_class.get_lock()` to use [default: mp_lock].
    tldm_class  : optional
        `tldm` class to use for bars [default: tldm.auto.tldm].
    """
    from concurrent.futures import ProcessPoolExecutor

    if iterables and "chunksize" not in tldm_kwargs:
        # default `chunksize=1` has poor performance for large iterables
        # (most time spent dispatching items to workers).
        shortest_iterable_len = _min_map_len(iterables)
        if shortest_iterable_len is not None and shortest_iterable_len > 1000:
            from warnings import warn

            warn(
                "Iterable length %d > 1000 but `chunksize` is not set."
                " This may seriously degrade multiprocess performance."
                " Set `chunksize=1` or more." % shortest_iterable_len,
                TldmWarning,
                stacklevel=2,
            )
    if "lock_name" not in tldm_kwargs:
        tldm_kwargs = tldm_kwargs.copy()
        tldm_kwargs["lock_name"] = "mp_lock"
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tldm_kwargs)

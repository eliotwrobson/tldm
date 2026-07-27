"""
Tests for `tldm.contrib.concurrent`.
"""

from contextlib import closing
from io import StringIO
from unittest.mock import patch

from pytest import importorskip, mark, skip, warns

from tldm.extensions.concurrent import process_map, thread_map
from tldm.utils import TldmWarning


def incr(x):
    """Dummy function"""
    return x + 1


class MockTldm:
    _lock = object()

    def __init__(self, **_kwargs):
        self.n = 0

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        self.n += n


def test_thread_map():
    """Test contrib.concurrent.thread_map"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        try:
            assert thread_map(lambda x: x + 1, a, file=our_file) == b
        except ImportError as err:
            skip(str(err))
        assert thread_map(incr, a, file=our_file) == b


def test_process_map():
    """Test contrib.concurrent.process_map"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        try:
            assert process_map(incr, a, file=our_file) == b
        except ImportError as err:
            skip(str(err))


@mark.parametrize(
    "iterables,should_warn",
    [
        ([], False),
        (["x"], False),
        ([()], False),
        (["x", ()], False),
        (["x" * 1001], True),
        (["x" * 100, ("x",) * 1001], False),
    ],
)
def test_chunksize_warning(iterables, should_warn):
    """Test extensions.concurrent.process_map chunksize warnings"""
    patch = importorskip("unittest.mock").patch
    with patch("tldm.extensions.concurrent._executor_map"):
        if should_warn:
            warns(TldmWarning, process_map, incr, *iterables)
        else:
            process_map(incr, *iterables)


def test_thread_map_unknown_length_iterable():
    """Test thread_map handles iterables without known length."""
    with closing(StringIO()) as our_file:
        assert thread_map(incr, (i for i in range(4)), file=our_file, tldm_class=MockTldm) == [
            1,
            2,
            3,
            4,
        ]


def test_thread_map_forwards_executor_kwargs():
    """Test thread_map forwards thread_name_prefix and buffersize."""
    seen = {}

    class DummyFuture:
        def add_done_callback(self, callback):
            callback(self)

    class DummyThreadPoolExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=(), **kwargs):
            seen["init"] = {
                "max_workers": max_workers,
                "initializer": initializer,
                "initargs": initargs,
                **kwargs,
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *_args, **_kwargs):
            return DummyFuture()

        def map(self, fn, *iterables, timeout=None, chunksize=1, **kwargs):
            seen["map"] = {
                "timeout": timeout,
                "chunksize": chunksize,
                **kwargs,
            }
            for args in zip(*iterables):
                self.submit(fn, *args)
                yield fn(*args)

    with patch("concurrent.futures.ThreadPoolExecutor", DummyThreadPoolExecutor):
        with closing(StringIO()) as our_file:
            result = thread_map(
                incr,
                range(3),
                file=our_file,
                thread_name_prefix="tldm-test",
                buffersize=7,
                tldm_class=MockTldm,
            )

    assert result == [1, 2, 3]
    assert seen["init"]["thread_name_prefix"] == "tldm-test"
    assert seen["map"]["buffersize"] == 7


def test_process_map_forwards_pool_kwargs():
    """Test process_map forwards max_tasks_per_child and mp_context."""
    seen = {}

    class DummyFuture:
        def add_done_callback(self, callback):
            callback(self)

    class DummyProcessPoolExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=(), **kwargs):
            seen["init"] = {
                "max_workers": max_workers,
                "initializer": initializer,
                "initargs": initargs,
                **kwargs,
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *_args, **_kwargs):
            return DummyFuture()

        def map(self, fn, *iterables, timeout=None, chunksize=1, **kwargs):
            seen["map"] = {
                "timeout": timeout,
                "chunksize": chunksize,
                **kwargs,
            }
            for args in zip(*iterables):
                self.submit(fn, *args)
                yield fn(*args)

    with patch("concurrent.futures.ProcessPoolExecutor", DummyProcessPoolExecutor):
        with closing(StringIO()) as our_file:
            result = process_map(
                incr,
                range(3),
                file=our_file,
                max_tasks_per_child=5,
                mp_context="dummy-context",
                buffersize=3,
                tldm_class=MockTldm,
            )

    assert result == [1, 2, 3]
    assert seen["init"]["max_tasks_per_child"] == 5
    assert seen["init"]["mp_context"] == "dummy-context"
    assert seen["map"]["buffersize"] == 3

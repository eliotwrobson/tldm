"""
Tests for `tldm.contrib.itertools`.
"""

import itertools as it
import sys
from contextlib import closing
from io import StringIO

import pytest

from tldm.aliases import tbatched, tenumerate, tmap, tproduct, tzip
from tldm.std import tldm


class NoLenIter:
    def __init__(self, iterable):
        self._it = iterable

    def __iter__(self):
        yield from self._it


def test_product():
    """Test contrib.itertools.product"""
    with closing(StringIO()) as our_file:
        a = range(9)
        assert list(tproduct(a, a[::-1], file=our_file)) == list(it.product(a, a[::-1]))

        assert list(tproduct(a, NoLenIter(a), file=our_file)) == list(it.product(a, NoLenIter(a)))


def test_product_with_repeat():
    """Test the case where a repeat argument has been set"""
    with closing(StringIO()) as our_file:
        a = range(9)
        assert list(tproduct(a, repeat=2, file=our_file)) == list(it.product(a, repeat=2))


@pytest.mark.parametrize("tldm_kwargs", [{}, {"tldm_class": tldm}])
def test_enumerate(tldm_kwargs):
    """Test contrib.tenumerate"""
    a = range(9)

    with closing(StringIO()) as our_file:
        assert list(tenumerate(a, file=our_file, **tldm_kwargs)) == list(enumerate(a))
        assert list(tenumerate(a, 42, file=our_file, **tldm_kwargs)) == list(enumerate(a, 42))
    with closing(StringIO()) as our_file:
        _ = tenumerate(iter(a), file=our_file, **tldm_kwargs)
        assert "100%" not in our_file.getvalue()
    with closing(StringIO()) as our_file:
        _ = list(tenumerate(iter(a), file=our_file, **tldm_kwargs))
        assert "100%" in our_file.getvalue()


@pytest.mark.parametrize("tldm_kwargs", [{}, {"tldm_class": tldm}])
def test_zip(tldm_kwargs):
    """Test contrib.tzip"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        gen = tzip(a, b, file=our_file, **tldm_kwargs)
        assert gen != list(zip(a, b))
        assert list(gen) == list(zip(a, b))


@pytest.mark.parametrize("tldm_kwargs", [{}, {"tldm_class": tldm}])
def test_map(tldm_kwargs):
    """Test contrib.tmap"""
    with closing(StringIO()) as our_file:
        a = range(9)
        b = [i + 1 for i in a]
        gen = tmap(lambda x: x + 1, a, file=our_file, **tldm_kwargs)
        assert gen != b
        assert list(gen) == b


# Only test batched on Python 3.12+ where itertools.batched exists
@pytest.mark.skipif(sys.version_info < (3, 12), reason="itertools.batched requires Python 3.12+")
@pytest.mark.parametrize("tldm_kwargs", [{}, {"tldm_class": tldm}])
def test_batched(tldm_kwargs):
    """Test contrib.tbatched - basic functionality, progress display, generators, and total"""
    # Basic functionality
    with closing(StringIO()) as our_file:
        a = range(10)
        result = list(tbatched(a, 3, file=our_file, **tldm_kwargs))
        expected = list(it.batched(a, 3))
        assert result == expected
        assert len(result) == 4  # (0,1,2), (3,4,5), (6,7,8), (9,)

    # Progress display shows batch count, not item count
    with closing(StringIO()) as our_file:
        result = list(
            tbatched(range(100), 10, file=our_file, leave=True, ascii=True, **tldm_kwargs)
        )
        assert len(result) == 10
        output = our_file.getvalue()
        assert "10/10" in output and "100%" in output

    # Generator without __len__
    def gen():
        yield from range(10)

    with closing(StringIO()) as our_file:
        result = list(tbatched(gen(), 3, file=our_file, **tldm_kwargs))
        assert result == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]

    # Generator with explicit total
    with closing(StringIO()) as our_file:
        result = list(
            tbatched(gen(), 3, total=10, file=our_file, leave=True, ascii=True, **tldm_kwargs)
        )
        assert result == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]
        assert "4/4" in our_file.getvalue()


@pytest.mark.skipif(sys.version_info < (3, 13), reason="strict parameter requires Python 3.13+")
@pytest.mark.parametrize("tldm_kwargs", [{}, {"tldm_class": tldm}])
def test_batched_strict(tldm_kwargs):
    """Test contrib.tbatched with strict=True (Python 3.13+)"""
    with closing(StringIO()) as our_file:
        # strict=True should raise if last batch is incomplete
        with pytest.raises(ValueError):
            list(tbatched(range(10), 3, strict=True, file=our_file, **tldm_kwargs))

        # strict=True should work if batches are complete
        result = list(tbatched(range(9), 3, strict=True, file=our_file, **tldm_kwargs))
        expected = list(it.batched(range(9), 3, strict=True))
        assert result == expected

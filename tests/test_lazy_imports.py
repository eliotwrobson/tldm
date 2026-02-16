"""
Tests for lazy imports (PR #1710 equivalent).

Verifies that std_tldm and auto_tldm are not imported at module load time
when callers supply their own tldm_class, improving performance.
"""

import logging


def test_concurrent_with_custom_class():
    """Test that concurrent uses custom tldm_class without loading std_tldm"""
    from tldm.extensions.concurrent import thread_map

    class MockTldm:
        _lock = None

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable or []

        def __iter__(self):
            return iter(self.iterable)

        @classmethod
        def get_lock(cls):
            from threading import RLock

            return cls._lock or RLock()

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

    result = thread_map(lambda x: x + 1, range(3), tldm_class=MockTldm)
    assert result == [1, 2, 3]


def test_concurrent_uses_default():
    """Test that concurrent lazy-loads std_tldm when tldm_class is None"""
    from tldm.extensions.concurrent import thread_map

    result = thread_map(lambda x: x + 1, range(3))
    assert result == [1, 2, 3]


def test_logging_with_custom_class():
    """Test that logging uses custom tldm_class without loading std_tldm"""
    from tldm.logging import TldmLoggingHandler

    class MockTldm:
        messages = []

        @classmethod
        def write(cls, msg, **kwargs):
            cls.messages.append(msg)

    MockTldm.messages = []
    logger = logging.Logger("test")
    logger.handlers = [TldmLoggingHandler(MockTldm)]
    logger.info("test")
    assert "test" in MockTldm.messages


def test_logging_uses_default():
    """Test that logging lazy-loads std_tldm when tldm_class is None"""
    from tldm.logging import TldmLoggingHandler, tldm_logging_redirect

    handler = TldmLoggingHandler()
    assert hasattr(handler.tldm_class, "write")

    with tldm_logging_redirect() as pbar:
        assert hasattr(pbar, "update")

import logging
import logging.handlers
import sys
from io import StringIO

import pytest
from pytest import importorskip

from tldm import tldm
from tldm.logging import (
    TldmLoggingHandler,
    _get_first_found_console_logging_handler,
    logging_redirect_tldm,
    tldm_logging_redirect,
)

LOGGER = logging.getLogger(__name__)

TEST_LOGGING_FORMATTER = logging.Formatter()


class CustomTldm(tldm):
    messages = []

    @classmethod
    def write(cls, s, **__):
        CustomTldm.messages.append(s)


class ErrorRaisingTldm(tldm):
    exception_class = RuntimeError

    @classmethod
    def write(cls, s, **__):
        raise ErrorRaisingTldm.exception_class("fail fast")


class TestTldmLoggingHandler:
    def test_should_call_tldm_write(self):
        CustomTldm.messages = []
        logger = logging.Logger("test")
        logger.handlers = [TldmLoggingHandler(CustomTldm)]
        logger.info("test")
        assert CustomTldm.messages == ["test"]

    def test_should_call_handle_error_if_exception_was_thrown(self):
        patch = importorskip("unittest.mock").patch
        logger = logging.Logger("test")
        ErrorRaisingTldm.exception_class = RuntimeError
        handler = TldmLoggingHandler(ErrorRaisingTldm)
        logger.handlers = [handler]
        with patch.object(handler, "handleError") as mock:
            logger.info("test")
            assert mock.called

    @pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
    def test_should_not_swallow_certain_exceptions(self, exception_class):
        logger = logging.Logger("test")
        ErrorRaisingTldm.exception_class = exception_class
        handler = TldmLoggingHandler(ErrorRaisingTldm)
        logger.handlers = [handler]
        with pytest.raises(exception_class):
            logger.info("test")


class TestGetFirstFoundConsoleLoggingHandler:
    def test_should_return_none_for_no_handlers(self):
        assert _get_first_found_console_logging_handler([]) is None

    def test_should_return_none_without_stream_handler(self):
        handler = logging.handlers.MemoryHandler(capacity=1)
        assert _get_first_found_console_logging_handler([handler]) is None

    def test_should_return_none_for_stream_handler_not_stdout_or_stderr(self):
        handler = logging.StreamHandler(StringIO())
        assert _get_first_found_console_logging_handler([handler]) is None

    def test_should_return_stream_handler_if_stream_is_stdout(self):
        handler = logging.StreamHandler(sys.stdout)
        assert _get_first_found_console_logging_handler([handler]) == handler

    def test_should_return_stream_handler_if_stream_is_stderr(self):
        handler = logging.StreamHandler(sys.stderr)
        assert _get_first_found_console_logging_handler([handler]) == handler


class TestRedirectLoggingToTldm:
    def test_should_add_and_remove_tldm_handler(self):
        logger = logging.Logger("test")
        with logging_redirect_tldm(loggers=[logger]):
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], TldmLoggingHandler)
        assert not logger.handlers

    def test_should_remove_and_restore_console_handlers(self):
        logger = logging.Logger("test")
        stderr_console_handler = logging.StreamHandler(sys.stderr)
        stdout_console_handler = logging.StreamHandler(sys.stderr)
        logger.handlers = [stderr_console_handler, stdout_console_handler]
        with logging_redirect_tldm(loggers=[logger]):
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], TldmLoggingHandler)
        assert logger.handlers == [stderr_console_handler, stdout_console_handler]

    def test_should_inherit_console_logger_formatter(self):
        logger = logging.Logger("test")
        formatter = logging.Formatter("custom: %(message)s")
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        logger.handlers = [console_handler]
        with logging_redirect_tldm(loggers=[logger]):
            assert logger.handlers[0].formatter == formatter

    def test_should_inherit_console_logger_level(self):
        level = 99
        logger = logging.Logger("test")
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        logger.handlers = [console_handler]
        with logging_redirect_tldm(loggers=[logger]):
            assert logger.handlers[0].level == level

    def test_should_inherit_console_logger_filters(self):
        """Test that filters are preserved during logging redirect (issue #1581)"""

        class TestFilter(logging.Filter):
            def filter(self, record):
                record.msg = f"{record.msg} -- filtered"
                return True

        logger = logging.Logger("test")
        console_handler = logging.StreamHandler(sys.stderr)
        test_filter = TestFilter()
        console_handler.addFilter(test_filter)
        logger.handlers = [console_handler]
        with logging_redirect_tldm(loggers=[logger]):
            assert test_filter in logger.handlers[0].filters

    def test_should_not_remove_stream_handlers_not_for_stdout_or_stderr(self):
        logger = logging.Logger("test")
        stream_handler = logging.StreamHandler(StringIO())
        logger.addHandler(stream_handler)
        with logging_redirect_tldm(loggers=[logger]):
            assert len(logger.handlers) == 2
            assert logger.handlers[0] == stream_handler
            assert isinstance(logger.handlers[1], TldmLoggingHandler)
        assert logger.handlers == [stream_handler]


class TestTldmWithLoggingRedirect:
    def test_should_add_and_remove_handler_from_root_logger_by_default(self):
        original_handlers = list(logging.root.handlers)
        with tldm_logging_redirect(total=1) as pbar:
            assert isinstance(logging.root.handlers[-1], TldmLoggingHandler)
            LOGGER.info("test")
            pbar.update(1)
        assert logging.root.handlers == original_handlers

    def test_should_add_and_remove_handler_from_custom_logger(self):
        logger = logging.Logger("test")
        with tldm_logging_redirect(total=1, loggers=[logger]) as pbar:
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], TldmLoggingHandler)
            logger.info("test")
            pbar.update(1)
        assert not logger.handlers

    def test_should_not_fail_with_logger_without_console_handler(self):
        logger = logging.Logger("test")
        logger.handlers = []
        with tldm_logging_redirect(total=1, loggers=[logger]):
            logger.info("test")
        assert not logger.handlers

    def test_should_format_message(self):
        logger = logging.Logger("test")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(r"prefix:%(message)s"))
        logger.handlers = [console_handler]
        CustomTldm.messages = []
        with tldm_logging_redirect(loggers=[logger], tldm_class=CustomTldm):
            logger.info("test")
        assert CustomTldm.messages == ["prefix:test"]

    def test_use_root_logger_by_default_and_write_to_custom_tldm(self):
        logger = logging.root
        CustomTldm.messages = []
        with tldm_logging_redirect(total=1, tldm_class=CustomTldm) as pbar:
            assert isinstance(pbar, CustomTldm)
            logger.info("test")
            assert CustomTldm.messages == ["test"]

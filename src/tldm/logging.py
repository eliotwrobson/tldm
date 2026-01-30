"""
Helper functionality for interoperability with stdlib `logging`.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .std import tldm as std_tldm


class TldmLoggingHandler(logging.StreamHandler):
    def __init__(
        self,
        tldm_class: "type[std_tldm]" = std_tldm,
    ) -> None:
        super().__init__()
        self.tldm_class = tldm_class

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.tldm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa: E722  # pylint: disable=bare-except
            self.handleError(record)


def _is_console_logging_handler(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.StreamHandler) and handler.stream in {
        sys.stdout,
        sys.stderr,
    }


def _get_first_found_console_logging_handler(
    handlers: list[logging.Handler],
) -> logging.StreamHandler | None:
    for handler in handlers:
        if _is_console_logging_handler(handler):
            return handler  # type: ignore[return-value]
    return None


@contextmanager
def logging_redirect_tldm(
    loggers: list[logging.Logger] | None = None,
    tldm_class: "type[std_tldm]" = std_tldm,
) -> Iterator[None]:
    """
    Context manager redirecting console logging to `tldm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tldm_class  : optional

    Example
    -------
    ```python
    import logging
    from tldm import trange
    from tldm.logging import logging_redirect_tldm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tldm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tldm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tldm_handler = TldmLoggingHandler(tldm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tldm_handler.setFormatter(orig_handler.formatter)
                tldm_handler.setLevel(orig_handler.level)
                tldm_handler.stream = orig_handler.stream
                # Copy filters from original handler (issue #1581)
                for f in orig_handler.filters:
                    tldm_handler.addFilter(f)
            logger.handlers = [
                handler for handler in logger.handlers if not _is_console_logging_handler(handler)
            ] + [tldm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers


@contextmanager
def tldm_logging_redirect(
    *args: Any,
    **kwargs: Any,
) -> Iterator[std_tldm]:
    """
    Convenience shortcut for:
    ```python
    with tldm_class(*args, **tldm_kwargs) as pbar:
        with logging_redirect_tldm(loggers=loggers, tldm_class=tldm_class):
            yield pbar
    ```

    Parameters
    ----------
    tldm_class  : optional, (default: tldm.std.tldm).
    loggers  : optional, list.
    **tldm_kwargs  : passed to `tldm_class`.
    """
    tldm_kwargs = dict(kwargs)
    loggers = tldm_kwargs.pop("loggers", None)
    tldm_class = tldm_kwargs.pop("tldm_class", std_tldm)
    with tldm_class(*args, **tldm_kwargs) as pbar:
        with logging_redirect_tldm(loggers=loggers, tldm_class=tldm_class):
            yield pbar

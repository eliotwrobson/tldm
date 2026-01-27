"""
Helper functionality for interoperability with stdlib `logging`.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .std import tldm as std_tldm


class _TqdmLoggingHandler(logging.StreamHandler):
    def __init__(
        self,
        tqdm_class: "type[std_tldm]" = std_tldm,
    ) -> None:
        super().__init__()
        self.tqdm_class = tqdm_class

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
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
def logging_redirect_tqdm(
    loggers: list[logging.Logger] | None = None,
    tqdm_class: "type[std_tldm]" = std_tldm,
) -> Iterator[None]:
    """
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    if loggers is None:
        loggers = [logging.root]
    original_handlers_list = [logger.handlers for logger in loggers]
    try:
        for logger in loggers:
            tqdm_handler = _TqdmLoggingHandler(tqdm_class)
            orig_handler = _get_first_found_console_logging_handler(logger.handlers)
            if orig_handler is not None:
                tqdm_handler.setFormatter(orig_handler.formatter)
                tqdm_handler.setLevel(orig_handler.level)
                tqdm_handler.stream = orig_handler.stream
            logger.handlers = [
                handler for handler in logger.handlers if not _is_console_logging_handler(handler)
            ] + [tqdm_handler]
        yield
    finally:
        for logger, original_handlers in zip(loggers, original_handlers_list):
            logger.handlers = original_handlers


@contextmanager
def tqdm_logging_redirect(
    *args: Any,
    **kwargs: Any,
) -> Iterator[std_tldm]:
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    tqdm_kwargs = dict(kwargs)
    loggers = tqdm_kwargs.pop("loggers", None)
    tqdm_class = tqdm_kwargs.pop("tqdm_class", std_tldm)
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar

"""
Registration for `tldm` to provide `pandas` progress indicators.
"""

import contextlib
from typing import Any

from ..aliases import auto_tldm


def tldm_pandas(**tldm_kwargs: dict[str, Any]) -> None:
    """
    Registers the current `tldm` class with
        pandas.core.
        ( frame.DataFrame
        | series.Series
        | groupby.(generic.)DataFrameGroupBy
        | groupby.(generic.)SeriesGroupBy
        ).progress_apply

    A new instance will be created every time `progress_apply` is called,
    and each instance will automatically `close()` upon completion.

    Parameters
    ----------
    tldm_kwargs  : arguments for the tldm instance

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tldm import tldm
    >>>
    >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
    >>>
    >>> # Recommended: Use syntactic sugar
    >>> tldm.pandas(ncols=50)
    >>>
    >>> # Alternative: Direct import
    >>> from tldm.extensions.pandas import tldm_pandas
    >>> tldm_pandas(ncols=50)
    >>>
    >>> # Now you can use `progress_apply` instead of `apply`
    >>> df.groupby(0).progress_apply(lambda x: x**2)

    References
    ----------
    <https://stackoverflow.com/questions/18603270/\
    progress-indicator-during-pandas-operations-python>
    """
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
    from pandas.core.groupby.groupby import GroupBy
    from pandas.core.series import Series
    from pandas.core.window.expanding import Expanding
    from pandas.core.window.rolling import Rolling

    _Rolling_and_Expanding = (Rolling, Expanding)

    tldm_kwargs = tldm_kwargs.copy()

    def inner_generator(df_function="apply"):
        def inner(df, func, **kwargs):
            """
            Parameters
            ----------
            df  : (DataFrame|Series)[GroupBy]
                Data (may be grouped).
            func  : function
                To be applied on the (grouped) data.
            **kwargs  : optional
                Transmitted to `df.apply()`.
            """

            # Precompute total iterations
            total = tldm_kwargs.pop("total", getattr(df, "ngroups", None))
            if total is None:  # not grouped
                if df_function == "applymap":
                    total = df.size
                elif isinstance(df, Series):
                    total = len(df)
                elif _Rolling_and_Expanding is None or not isinstance(df, _Rolling_and_Expanding):
                    # DataFrame or Panel
                    axis = kwargs.get("axis", 0)
                    if axis == "index":
                        axis = 0
                    elif axis == "columns":
                        axis = 1
                    # when axis=0, total is shape[axis1]
                    total = df.size // df.shape[axis]

            # Init bar
            t = auto_tldm(total=total, **tldm_kwargs)

            # Try to use pandas' is_builtin_func if available (optimization)
            # This was removed in pandas 3.0, so we need to handle both cases
            try:
                from pandas.core.common import is_builtin_func

                with contextlib.suppress(TypeError):
                    func = is_builtin_func(func)
            except ImportError:
                # pandas >= 3.0 removed is_builtin_func
                # We can safely skip this optimization
                pass

            # Define bar updating wrapper
            def wrapper(*args, **kwargs):
                # update tbar correctly
                # it seems `pandas apply` calls `func` twice
                # on the first column/row to decide whether it can
                # take a fast or slow code path; so stop when t.total==t.n
                t.update(n=1 if not t.total or t.n < t.total else 0)
                return func(*args, **kwargs)

            # Apply the provided function (in **kwargs)
            # on the df using our wrapper (which provides bar updating)
            try:
                return getattr(df, df_function)(wrapper, **kwargs)
            finally:
                t.close()

        return inner

    # Monkeypatch pandas to provide easy methods
    # Enable custom tldm progress in pandas!
    Series.progress_apply = inner_generator()
    SeriesGroupBy.progress_apply = inner_generator()
    Series.progress_map = inner_generator("map")
    SeriesGroupBy.progress_map = inner_generator("map")

    DataFrame.progress_apply = inner_generator()
    DataFrameGroupBy.progress_apply = inner_generator()
    DataFrame.progress_applymap = inner_generator("applymap")
    DataFrame.progress_map = inner_generator("map")
    DataFrameGroupBy.progress_map = inner_generator("map")

    GroupBy.progress_apply = inner_generator()
    GroupBy.progress_aggregate = inner_generator("aggregate")
    GroupBy.progress_transform = inner_generator("transform")

    Rolling.progress_apply = inner_generator()
    Expanding.progress_apply = inner_generator()

from importlib.metadata import version

from ._monitor import TMonitor as TMonitor
from .aliases import auto_tldm as auto_tldm
from .aliases import tbatched as tbatched
from .aliases import tenumerate as tenumerate
from .aliases import tmap as tmap
from .aliases import tproduct as tproduct
from .aliases import trange as trange
from .aliases import tzip as tzip
from .std import tldm as tldm

__version__ = version("tldm")


# Syntactic sugar for pandas integration
def pandas(**tldm_kwargs) -> None:
    """
    Register tldm with pandas to enable progress_apply, progress_map, etc.

    This is a convenience function that imports and calls tldm_pandas.
    Automatically uses the appropriate progress bar for your environment
    (terminal or Jupyter notebook).

    Parameters
    ----------
    **tldm_kwargs : dict
        Keyword arguments to pass to tldm instances created by pandas operations.
        Common options include: desc, total, leave, ncols, etc.

    Examples
    --------
    Using tldm.pandas():

    >>> import pandas as pd
    >>> from tldm import tldm
    >>> tldm.pandas(desc="Processing")  # Register with options
    >>> df.progress_apply(lambda x: x**2)

    Alternative import style:

    >>> from tldm import pandas
    >>> pandas()  # Register with default settings
    >>> df.progress_apply(lambda x: x**2)
    """
    from .extensions.pandas import tldm_pandas

    tldm_pandas(**tldm_kwargs)

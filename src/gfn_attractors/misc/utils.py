import sys
import inspect
from collections.abc import Iterable
import itertools
import numpy as np
import pandas as pd


_use_torch = 'torch' in sys.modules
if _use_torch:
    import torch


def kv_str(_delim=" | ", _digits=3, **kwargs):
    s = []
    for k, v in kwargs.items():
        if _use_torch and isinstance(v, torch.Tensor):
            if len(v.shape) == 0:
                v = v.item()
            else:
                v = v.detach().cpu()
        if isinstance(v, float):
            v = round(v, _digits)
        s.append("{}: {}".format(k, v))
    s = _delim.join(s)
    return s


def kv_print(_delim=" | ", _digits=3, **kwargs):
    """
    Pretty-prints kwargs

    :param _delim: Delimiter to separate kwargs
    :param _digits: number of decimal digits to round to
    :param kwargs: stuff to print
    :return:
    """
    print(kv_str(_delim, _digits=_digits, **kwargs))


def is_iterable(obj, allow_str=False):
    if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        return len(obj.shape) > 0
    if allow_str:
        return isinstance(obj, Iterable)
    else:
        return isinstance(obj, Iterable) and not isinstance(obj, str)


def extract_args(args):
    """
    Use when *args is used as a function parameter.
    Allows both an iterable and a sequence of parameters to be passed in.
    For example, if f([1, 2, 3]) and f(1, 2, 3) will be valid inputs to the following function
        def f(*args):
            args = extract_args(args)
            ...
    @param args:
    @return:
    """
    if len(args) == 1 and is_iterable(args[0]):
        return args[0]
    return args


def filter_kwargs(fn, kwargs):
    """
    Filters out kwargs that are not in the signature of fn
    """
    return {k: v for k, v in kwargs.items() if k in inspect.signature(fn).parameters}


def to_long_df(array, dim_names, value_name='value', **kwargs):
    """
    Given a multi-dimensional array or tensor, returns a long-form DataFrame with the dimensions as columns.
    Can also specify additional columns with the kwargs, as long as they have the same shape as the first k dimensions of array,
    where k is the number of dimensions in the kwarg value..
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    shape = array.shape
    array = array.flatten()
    index = pd.MultiIndex.from_product([range(i) for i in shape], names=dim_names)
    df = pd.DataFrame(array, columns=[value_name], index=index).reset_index()
    for k, v in kwargs.items():
        i = len(v.shape)
        v = v.flatten()
        while len(v) < len(df):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.repeat(v, shape[i])
            i += 1
        df[k] = v
    return df


def cycle(iterator, n):
    """
    draws from an iterator n times, reinstantiating the iterator each time, rather than
    caching the results like itertools.cycle
    this is useful for DataLoader when using shuffle=True

    :param iterator:
    :param n:
    :return:
    """
    i = 0
    while i < n:
        for x in iterator:
            yield x
            i += 1
            if i >= n:
                return


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    # Added to itertools in Python 3.12
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def chunk(lst, n):
    """
    Splits the list into n chunks of approximately equal size
    """
    if n < 1:
        raise ValueError('n must be at least one')
    k = len(lst) // n
    m = len(lst) % n
    return (lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))


def get_partition_sizes(n, k):
    """
    Splits n into k approximately equal integers
    >>> get_partition_sizes(11, 3)
    [4, 4, 3]
    """
    # Calculate the approximate size of each integer
    size = n // k
    
    # Calculate the remaining difference
    remaining = n - (size * k)
    
    # Generate the list of integers
    integers = [size] * k
    
    # Distribute the remaining difference evenly among the integers
    for i in range(remaining):
        integers[i] += 1
    
    return integers

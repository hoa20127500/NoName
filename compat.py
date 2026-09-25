"""Python 3.10+ aliases needed by older DGL (Colab currently ships 3.13).

DGL still does ``from collections import Mapping, Iterable``. Those names
moved to ``collections.abc`` and were removed from ``collections`` in 3.10.
"""
import collections
import collections.abc

for _name in (
        'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence',
        'Iterable', 'Iterator', 'Callable', 'Set', 'MutableSet',
        'Hashable', 'Container', 'ValuesView', 'KeysView', 'ItemsView',
):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

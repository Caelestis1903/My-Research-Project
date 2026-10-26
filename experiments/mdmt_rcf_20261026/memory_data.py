"""Explicit, lossless mappings of row data; never model or optimizer transport.

Call sites are restricted to H/Z, pilot activations, predictions, function
outputs, and label-access masks. No object-graph or tensor serialization exists.
"""
from __future__ import annotations
import gc
import hashlib
import mmap
from pathlib import Path
import weakref
import numpy as np

_MIN_BYTES = 256 * 1024
_CHUNK_BYTES = 8 * 1024 * 1024


def data_hash(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    rows = max(1, _CHUNK_BYTES // max(1, a.dtype.itemsize * int(np.prod(a.shape[1:]))))
    if a.ndim == 0:
        h.update(a.tobytes())
    else:
        for i in range(0, len(a), rows):
            h.update(np.ascontiguousarray(a[i:i+rows]).tobytes())
    return h.hexdigest()


class RowDataStore:
    def __init__(self, root):
        self.root = Path(root).resolve() / 'row_data_cache'
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.maps = weakref.WeakValueDictionary()

    def persist(self, label, value):
        if not isinstance(value, np.ndarray):
            raise TypeError('Only explicit NumPy row-data arrays may be mapped')
        if value.dtype.hasobject:
            raise TypeError('Object arrays are prohibited')
        if value.nbytes < _MIN_BYTES:
            return value
        if isinstance(value, np.memmap):
            return value
        hv = data_hash(value)
        key = hashlib.sha256(str(label).encode()).hexdigest()[:20] + '_' + hv
        path = self.root / (key + '.npy')
        if path.exists():
            mapped = np.load(path, mmap_mode='r', allow_pickle=False)
            if mapped.dtype != value.dtype or mapped.shape != value.shape or data_hash(mapped) != hv:
                raise ValueError('Existing row-data mapping differs: ' + str(path))
        else:
            tmp = path.with_suffix('.npy.partial')
            mapped = np.lib.format.open_memmap(tmp, mode='w+', dtype=value.dtype, shape=value.shape)
            rows = max(1, _CHUNK_BYTES // max(1, value.dtype.itemsize * int(np.prod(value.shape[1:]))))
            for i in range(0, len(value), rows):
                mapped[i:i+rows] = value[i:i+rows]
            mapped.flush()
            if data_hash(mapped) != hv:
                raise ValueError('Row-data mapping changed values')
            del mapped
            tmp.replace(path)
            mapped = np.load(path, mmap_mode='r', allow_pickle=False)
        self.maps[key] = mapped
        self.records[key] = dict(label=str(label), path=str(path), shape=list(value.shape),
                                 dtype=str(value.dtype), array_hash=hv, bytes=int(value.nbytes),
                                 content_kind='ROW_DATA_ONLY')
        drop_pages(mapped)
        return mapped

    def release_pages(self):
        for value in list(self.maps.values()):
            drop_pages(value)
        gc.collect()
        # Return freed large temporary allocations, without touching live tensors.
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass


def drop_pages(value):
    if isinstance(value, np.memmap):
        value.flush()
        backing = getattr(value, '_mmap', None)
        if backing is not None and hasattr(backing, 'madvise'):
            backing.madvise(mmap.MADV_DONTNEED)


def store_for(bundle):
    store = getattr(bundle, '_row_data_store', None)
    root = getattr(bundle, 'output_root', None)
    if store is None and root is not None:
        store = RowDataStore(root)
        bundle._row_data_store = store
    return store


def persist(bundle, label, value):
    store = store_for(bundle)
    return value if store is None else store.persist(label, value)


def map_prediction(bundle, prediction):
    # Preserve the actual dictionaries captured by validation closures.
    for kind in ('probabilities', 'terminal_probabilities'):
        values = getattr(prediction, kind)
        for role in values:
            values[role] = persist(bundle, (prediction.method, kind, role), values[role])
    return prediction


def release_data_pages(bundle):
    store = store_for(bundle)
    if store is not None:
        store.release_pages()
    ledger = getattr(bundle, 'ledger', None)
    for value in getattr(ledger, 'unique', {}).values():
        drop_pages(value)


def data_cache_receipt(bundle):
    store = store_for(bundle)
    ledger = getattr(bundle, 'ledger', None)
    label_masks = [dict(path=str(Path(v.filename).resolve()),shape=list(v.shape),
                       dtype=str(v.dtype),bytes=int(v.nbytes),content_kind='MUTABLE_LABEL_ACCESS_DATA')
                   for v in getattr(ledger, 'unique', {}).values() if isinstance(v,np.memmap)]
    return dict(model_files_written=0, mappings=[] if store is None else list(store.records.values()),
                label_access_masks=label_masks,
                storage='NUMPY_ROW_DATA_MMAP_ONLY')

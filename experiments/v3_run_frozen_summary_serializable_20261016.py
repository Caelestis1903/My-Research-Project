"""Execute unchanged frozen summary with numpy-scalar JSON output compatibility."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import v3_test_summary_20261016 as frozen

def normalize(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value

def main():
    original_dump = frozen.base.dump
    source = Path(frozen.__file__)
    assert frozen.base.sha(source) == '149430001d72e57f5de6e745afb7614462b7c9c5f6a8c5b19482fef3ab58c2aa'
    frozen.base.dump = lambda path, value: original_dump(path, normalize(value))
    frozen.main()
    import sys
    output = Path(sys.argv[sys.argv.index('--out')+1])
    original_dump(output/'serialization_wrapper_provenance.json', dict(status='COMPLETE', wrapper_source_sha=frozen.base.sha(__file__), unchanged_frozen_summary_source_sha=frozen.base.sha(source), scope='numpy scalar to native scalar conversion only at JSON output boundary', time=time.time()))

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Complete requested package entrypoint, using the frozen shared protocol."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/reproduction_20261021')]
from experiments.reproduction_20261021.run import parser,run
if __name__=='__main__':
    p=parser(); p.set_defaults(arm="G")
    run(p.parse_args())

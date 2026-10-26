"""Mechanical queue inventory; emits configurations without starting training."""
from .bootstrap import ART,dump,sources,dependency_inventory,population,activate,actual_module_paths
from .v2 import configurations as v2_configurations

def write():
    population()
    activate('v2')
    import v2_20261016  # archived producer identity; no Runner or optimizer instantiated
    configs=v2_configurations()
    dump(ART/'extensions_manifest.json',dict(status='IMPLEMENTED_REVIEW_REQUIRED',
        source_inventory='experiments/natural_ml_extensions/source_archive/inventory.json',
        source_hashes=sources('v2'),dependency_inventory=dependency_inventory('v2'),
        actual_module_paths=actual_module_paths('v2'),
        configurations=configs,training_started=False,exploratory_replication=True))
    return configs
if __name__=='__main__':write()

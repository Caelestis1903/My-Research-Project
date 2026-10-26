"""Protocol trainer adapter. Original API is archived in source snapshots.

This port accepts the explicit run argparse namespace and runs the same audited
training/evaluation/checkpoint code as the executable package entrypoint.
"""
class CTRTrainer:
    def __init__(self, config):
        self.config=config
    def fit(self):
        from experiments.reproduction_20261021.run import run
        return run(self.config)

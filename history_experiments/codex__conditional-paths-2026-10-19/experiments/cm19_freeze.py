"""Bind complete training/configuration/selection before opening test metrics."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from cm19_train import ROOT,OUT,dump

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text());records={}
    for c in manifest['runs']:
        out=OUT/'runs'/c['run_id'];r=json.loads((out/'receipt.json').read_text());sel=json.loads((out/'selection_frozen.json').read_text());cfg=json.loads((out/'config.json').read_text())
        assert r['status']=='COMPLETE_TRAINING_TEST_DEFERRED';assert not (out/'test_receipt.json').exists()
        if c.get('max_steps'):assert r['steps']==c['max_steps']
        else:assert r['epochs']==c['epochs']
        assert all(cfg[k]==v for k,v in c.items())
        records[c['run_id']]=dict(selection_sha256=sha(out/'selection_frozen.json'),config_sha256=sha(out/'config.json'),receipt_sha256=sha(out/'receipt.json'),selected_epoch=sel['selected_epoch'],steps=r['steps'],checkpoint_sha256={label:sha(Path(cfg['state_directory'])/file) for label,file in [('selected','selected.pt'),('final','last.pt')]})
    assert (OUT/'DEVELOPMENT_DISPOSITION.md').exists()
    files=['cm19_train.py','cm19_models.py','cm19_legacy_data.py','cm19_legacy_models.py','cm19_hamur_reference.py']
    # Every run must bind to the current unchanged training graph; auxiliary scripts may have been added later.
    for c in manifest['runs']:
        hashes=json.loads((OUT/'runs'/c['run_id']/'source_hashes.json').read_text())
        for name in files:assert hashes[name]==sha(ROOT/'experiments'/name),(c['run_id'],name)
    dump(OUT/'TEST_FREEZE.json',dict(status='FROZEN_BEFORE_TEST',created_utc=datetime.now(timezone.utc).isoformat(),runs=records,training_source_hashes={name:sha(ROOT/'experiments'/name) for name in files},manifest_sha256=sha(OUT/'run_manifest.json'),development_sha256=sha(OUT/'DEVELOPMENT_DISPOSITION.md'),aliases=manifest['aliases'],new_population_confirmation=False))
    print(json.dumps(dict(status='FROZEN',runs=len(records),test_read=False)))
if __name__=='__main__':main()

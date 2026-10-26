"""Visualize already completed, immutable finite-K full-coordinate spectra."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_plans/2026-10-16_v3/execution_20261016'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    frame=pd.read_csv(OUT/'analysis/V1_full_coordinate_geometry.csv')
    assert frame.run.nunique()==16 and len(frame)==160 and set(frame.k)=={512}
    rows=[];sources=[];spectra=[]
    for name in frame.run.unique():
        directory=OUT/'audits'/name
        receipt=json.loads((directory/'receipt.json').read_text())
        assert receipt['status']=='PASS' and receipt['exact_identity_pass'] and receipt['k']==512
        assert sha(directory/'geometry.json')==receipt['artifact_hashes']['geometry.json']
        values=json.loads((directory/'geometry.json').read_text())
        assert len(values)==10
        for item in values:
            eigen=np.asarray(item['eigenvalues'],float)
            assert len(eigen)==512 and np.isfinite(eigen).all() and (eigen>=0).all()
            assert np.all(eigen[1:]<=eigen[:-1])
            assert abs(eigen.sum()-item['covariance_trace'])<1e-18+1e-9*abs(item['covariance_trace'])
            for i,value in enumerate(eigen):
                rows.append(dict(run=name,optimizer=item['optimizer'],seed=item['seed'],epoch=item['epoch'],arm=item['arm'],block=item['block'],eigen_rank=i+1,eigenvalue=float(value),K=512,population_spectrum_claim=False))
            if item['block']=='ALL': spectra.append(item)
        sources.append(dict(run=name,receipt_sha=sha(directory/'receipt.json'),geometry_sha=sha(directory/'geometry.json')))
    assert len(rows)==81920 and len(spectra)==32
    path=OUT/'archive/formal/final_V1_all_frozen_eigenvalues_81920.csv'
    assert not path.exists()
    pd.DataFrame(rows).to_csv(path,index=False)
    files=[path]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for index,optimizer in enumerate(('adam','sgd')):
        for item in spectra:
            if item['optimizer']!=optimizer: continue
            eigen=np.asarray(item['eigenvalues'])
            rank=np.arange(1,513);color='#2364aa' if item['arm']=='SN' else '#da641c'
            positive=eigen>0
            axes[index,0].plot(rank[positive],eigen[positive],color=color,alpha=.45,linewidth=.9)
            axes[index,1].plot(rank,np.cumsum(eigen)/eigen.sum(),color=color,alpha=.45,linewidth=.9)
        axes[index,0].set_yscale('log')
        for j in range(2):
            axes[index,j].set_xscale('log')
            axes[index,j].set_xlabel('Eigenvalue rank (all 512 stored eigenvalues)')
            axes[index,j].grid(alpha=.2)
            axes[index,j].set_title(optimizer.upper()+': individual spectra at all 8 audited theta points')
        axes[index,0].set_ylabel('Empirical covariance eigenvalue')
        axes[index,1].set_ylabel('Cumulative eigenvalue / trace')
        axes[index,1].set_ylim(0,1.02)
    fig.legend(handles=[Line2D([0],[0],color='#2364aa',label='SN'),Line2D([0],[0],color='#da641c',label='SB')],loc='upper right')
    fig.suptitle('Frozen full-coordinate ALL-block covariance spectra | K=512 independent batches per q\nIndividual conditional theta curves; no pooled seed mean, no population-spectrum or mediation claim')
    fig.tight_layout(rect=(0,0,1,.94))
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):
        path=directory/('final_frozen_covariance_spectra.'+suffix)
        assert not path.exists()
        fig.savefig(path,dpi=170,bbox_inches='tight');files.append(path)
    plt.close(fig)
    path=OUT/'final_covariance_spectrum_figure_receipt.json';assert not path.exists()
    path.write_text(json.dumps(dict(status='PASS',time=time.time(),independent_review=False,source_sha=sha(Path(__file__)),source_runs=sources,spectra_plotted=32,all_block_eigenvalue_rows=81920,scope='Onlyexistingfrozen16theta/fullK512spectra. All8theta/optimizer individuallines, not8trainingseedsornewexperiments. Zeroeigenvaluesomittedonlyonlogyrawpanel; all512incumulativepanel/CSV. Geometryassociationnotnecessarymediator.',hashes={str(p.relative_to(OUT)):sha(p) for p in files}),indent=2)+'\n')
if __name__=='__main__':main()

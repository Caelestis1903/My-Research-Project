"""Standalone reproduction of the fixed-prediction N0 figure."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import n17_common as c

def main():
    out=c.OUT/'N0';cells=pd.read_csv(out/'calAdam_E50_cell_contributions.csv');curve=pd.read_csv(out/'evaluation_weight_curve.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));means=cells.groupby('domain').difference.mean()
    axes[0].bar(['d0','d1','d2'],means,color=['#b44a4a','#367a57','#b44a4a']);axes[0].axhline(0,color='black',lw=.7)
    axes[0].set(ylabel='SB - SN AUC (two-task mean)',title='Fixed calAdam E50 predictions')
    for seed,part in curve.groupby('seed'):axes[1].plot(part.alpha,part.difference,alpha=.35,lw=1,label=str(seed))
    average=curve.groupby('alpha').difference.mean()
    axes[1].plot(average.index,average.to_numpy(),color='black',lw=2,label='4-seed mean');axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set(xlabel='alpha: uniform domain weights -> test proportions',ylabel='Weighted cell AUC difference',title='Evaluation weights change the sign');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'evaluation_decomposition.png',dpi=180);fig.savefig(out/'evaluation_decomposition.pdf');plt.close(fig)

if __name__=='__main__':main()

"""Reproducible dataset profile, outlier flags, and feature histograms."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.core.config import ROOT, RAW

OUT=ROOT/'reports'/'generated'; OUT.mkdir(parents=True,exist_ok=True)
def run():
    report={}
    for stem in ('parkinsons','parkinsons_updrs'):
        df=pd.read_csv(RAW/f'{stem}.data')
        num=df.select_dtypes(include='number')
        outliers={}
        for col in num:
            q1,q3=num[col].quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
            count=int(((num[col]<lo)|(num[col]>hi)).sum())
            if count: outliers[col]={"iqr_lower":float(lo),"iqr_upper":float(hi),"flagged_records":count}
        fig,axes=plt.subplots(4,4,figsize=(15,11)); axes=axes.flatten()
        for ax,col in zip(axes,num.columns):
            num[col].hist(ax=ax,bins=24,color='#248f87',edgecolor='white'); ax.set_title(col,fontsize=8); ax.tick_params(labelsize=7)
        for ax in axes[len(num.columns):]: ax.remove()
        fig.tight_layout(); fig.savefig(OUT/f'{stem}_distributions.png',dpi=150); plt.close(fig)
        report[stem]={"rows":len(df),"columns":len(df.columns),"types":{c:str(t) for c,t in df.dtypes.items()},"missing_by_column":df.isna().sum().astype(int).to_dict(),"duplicate_rows":int(df.duplicated().sum()),"numeric_summary":num.describe().round(4).to_dict(),"iqr_outlier_flags_not_removed":outliers}
        if stem=='parkinsons': report[stem]["class_counts"]=df.status.value_counts().sort_index().astype(int).to_dict(); report[stem]["recording_groups"]=int(df.name.str.rsplit('_',n=1).str[0].nunique())
        else: report[stem]["subjects"]=int(df['subject#'].nunique()); report[stem]["rows_per_subject"]=df.groupby('subject#').size().describe().to_dict(); report[stem]["time_range"]=[float(df.test_time.min()),float(df.test_time.max())]
    (OUT/'data_quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
if __name__=='__main__': print(json.dumps(run(),indent=2))

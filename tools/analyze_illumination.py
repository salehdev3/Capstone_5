"""Create a full-test illumination review template or aggregate completed human labels."""
import argparse
from pathlib import Path
import pandas as pd


def analyze(predictions, review, output):
    predictions, review, output=map(Path,(predictions,review,output))
    p=pd.read_csv(predictions,dtype={'sample_id':str,'true':str,'prediction':str})
    if p.empty or p.sample_id.isna().any() or p.sample_id.duplicated().any():
        raise ValueError('Predictions must contain unique sample IDs')
    if not review.exists():
        review.parent.mkdir(parents=True,exist_ok=True)
        p[['sample_id','camera']].assign(illumination='',reviewer='',notes='').to_csv(review,index=False)
        return 'Review template created; assign day/night/uncertain to every sample'
    r=pd.read_csv(review,dtype=str,keep_default_na=False)
    if r.sample_id.duplicated().any() or set(r.sample_id)!=set(p.sample_id):
        raise ValueError('Review must cover exactly the full test sample IDs')
    if not r.illumination.isin(['day','night','uncertain']).all() or not r.reviewer.str.strip().ne('').all():
        raise ValueError('Each sample needs a named reviewer and day/night/uncertain label')
    joined=p.merge(r[['sample_id','illumination']],on='sample_id',validate='one_to_one')
    joined['error']=joined.true.ne(joined.prediction)
    summary=joined.groupby('illumination').agg(samples=('sample_id','size'),errors=('error','sum'))
    summary['error_rate']=summary.errors/summary.samples
    patterns=joined[joined.error].groupby(['illumination','true','prediction']).size().reset_index(name='count')
    output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/'illumination_metrics.csv');patterns.to_csv(output/'illumination_error_patterns.csv',index=False)
    return 'Full-test illumination metrics exported; uncertain labels are retained'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','review','output'):p.add_argument('--'+name,required=True)
    print(analyze(**vars(p.parse_args())))

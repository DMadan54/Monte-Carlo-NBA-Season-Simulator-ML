"""Frozen chronological Platt/isotonic calibration evaluation."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

def score(y, p):
    p=np.clip(p,1e-6,1-1e-6); bins=np.minimum((p*10).astype(int),9)
    ece=sum((bins==b).sum()*abs(p[bins==b].mean()-y[bins==b].mean()) for b in range(10) if (bins==b).any())/len(y)
    return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),'ece':float(ece)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); args=ap.parse_args()
    data=pd.read_csv(args.run_dir/'game_probabilities.csv')
    data=data[(data.model=='hybrid_development_ml') & (data.learned_uncertainty==False)].copy()
    data['year']=data.season.str[:4].astype(int)
    train=data[data.year<=2022]; val=data[data.year==2023]; test=data[data.year==2024]
    methods={'raw':lambda p:p}
    platt=LogisticRegression(C=1,max_iter=1000).fit(train[['probability']],train.target)
    iso=IsotonicRegression(out_of_bounds='clip').fit(train.probability,train.target)
    methods['platt']=lambda p:platt.predict_proba(np.asarray(p).reshape(-1,1))[:,1]
    methods['isotonic']=lambda p:iso.predict(np.asarray(p))
    validation={name:score(val.target.to_numpy(),fn(val.probability.to_numpy())) for name,fn in methods.items()}
    chosen=min(validation,key=lambda k:validation[k]['log_loss'])
    result={'protocol':'Fit on 2018-19 to 2022-23 cutoffs; select on 2023-24; test once on 2024-25.',
            'validation':validation,'selected':chosen,
            'test':score(test.target.to_numpy(),methods[chosen](test.probability.to_numpy())),
            'raw_test':score(test.target.to_numpy(),test.probability.to_numpy())}
    (args.run_dir/'calibration_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()

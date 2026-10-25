"""Generate synthetic input and execute the full benchmark CLI for wiring checks."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'finshield'))
from benchmark import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(42)
    frame=pd.DataFrame({'Time':np.arange(3000),'signal':rng.normal(size=3000),'noise':rng.normal(size=3000)})
    frame['Class']=((frame.signal+rng.normal(size=3000))>2).astype(int)
    csv=output/'synthetic.csv'
    if csv.exists(): raise ValueError('Use a fresh output directory')
    frame.to_csv(csv,index=False)
    result=run(csv,output/'experiment',['signal','noise'],
        'Synthetic Gaussian fixture, seed 42. Wiring check only; NOT real fraud evidence.')
    print('Synthetic smoke check selected:',result['selected_by_validation_cost'])
    print('Report:',output/'experiment/report.json')

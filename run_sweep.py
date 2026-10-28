"""Standalone AngleFair controlled feature-angle sweep and masked-GCN baselines."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from src.synthetic import make_graph

p=argparse.ArgumentParser()
p.add_argument('--angles',nargs='+',type=int,default=[0,30,60,90])
p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
p.add_argument('--stage',type=int,choices=[1,2,3,4],default=4)
p.add_argument('--epochs',type=int,default=150)
p.add_argument('--generate-only',action='store_true')
p.add_argument('--output',default='results/csbm')
args=p.parse_args()
root=Path(args.output)
for angle in args.angles:
    for seed in args.seeds:
        work=root/f'angle_{angle}'/f'seed_{seed}'
        data=work/'data'
        data.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(data/'bail.npz',**make_graph(angle=angle,seed=seed))
        if not args.generate_only:
            subprocess.run([sys.executable,'train.py','--data-dir',str(data),'--dataset','bail',
                '--stage',str(args.stage),'--seeds',str(seed),'--epochs',str(args.epochs),
                '--output',str(work/'train')],check=True)
        print(f'angle={angle}, seed={seed}: {work}')

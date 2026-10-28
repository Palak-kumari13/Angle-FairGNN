import argparse
from pathlib import Path
import numpy as np
from src.synthetic import make_graph
p=argparse.ArgumentParser()
p.add_argument('--angles',type=float,nargs='+',default=[0,30,60,90]);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]);p.add_argument('--output',default='data/synthetic')
a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
for angle in a.angles:
    for seed in a.seeds:
        dst=root/f'angle{int(angle)}_seed{seed}.npz'
        np.savez_compressed(dst, **make_graph(angle=angle, seed=seed))
        print(dst)

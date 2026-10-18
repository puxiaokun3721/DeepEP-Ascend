"""Reproducible byte data for CPU golden and future A5 snapshot harness."""
import argparse
from pathlib import Path
import numpy as np
from golden import golden


def generate(batch=33, hidden=128, k=8, experts=64, ranks=8, fp8=False):
    rng = np.random.default_rng(20261016)
    topk = np.stack([rng.choice(experts,k,replace=False) for _ in range(batch)]) if batch else np.empty((0,k),np.int64)
    x = rng.integers(0,256,(batch,hidden if fp8 else 2*hidden),dtype=np.uint8)
    # Encodings are intentionally not converted through float: NaN payloads survive.
    weights = rng.integers(0,2**32,(batch,k),dtype=np.uint32)
    sf = rng.integers(0,256,(batch,hidden//32 if fp8 else 0),dtype=np.uint8)
    mask = np.arange(batch)%3 != 1
    golden(topk,experts,ranks,mask)
    return dict(x_bytes=x,sf_bytes=sf,topk=topk.astype('<i8'),weight_bits=weights,mask=mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--batch',type=int,default=33)
    parser.add_argument('--fp8',action='store_true')
    args = parser.parse_args()
    np.savez(args.output,**generate(batch=args.batch,fp8=args.fp8))

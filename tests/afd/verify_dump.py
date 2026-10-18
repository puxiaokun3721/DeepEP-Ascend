"""Validate a real F snapshot dump against independent CPU data (no NPU launch).

Requires one round's sliced input .npz from gen_data schema, and an F snapshot
.npz containing payload(uint8), route_refs, counts, prefixes, rank, num_experts,
num_ranks. Take snapshot BEFORE sending Consumed, after observing InputReady.
"""
import argparse
import json
import numpy as np
from golden import verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs')
    parser.add_argument('snapshot')
    args = parser.parse_args()
    with np.load(args.inputs,allow_pickle=False) as inputs, np.load(args.snapshot,allow_pickle=False) as snapshot:
        result = verify_snapshot(snapshot['payload'].tobytes(),snapshot['route_refs'],
            snapshot['counts'],snapshot['prefixes'],int(snapshot['rank']),
            inputs['x_bytes'],inputs['sf_bytes'],inputs['topk'],inputs['weight_bits'],
            int(snapshot['num_experts']),int(snapshot['num_ranks']),inputs['mask'])
    print(json.dumps({'status':'PASS','scope':'provided snapshot only',**result}))


if __name__ == '__main__':
    main()

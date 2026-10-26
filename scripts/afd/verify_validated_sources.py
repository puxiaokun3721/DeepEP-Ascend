#!/usr/bin/env python3
"""Check delivered source bytes; this is not a compiler or NPU validation."""

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / 'docs/afd/deployment/validated-source-manifest.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            failures.append(f"MISSING {entry['path']}")
            continue
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            failures.append(f"MISMATCH {entry['path']}")
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        print('Preserve local changes; do not rebuild or claim a matching snapshot.', file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} source files against the delivery manifest.")
    print('Source identity only. Rebuild and run the documented tests on the target machine.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

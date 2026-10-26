#!/usr/bin/env python3
"""Apply the three pinned, reviewed DeepJIT patches without replacing local edits."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dependency = root / 'third-party/deep_jit'
    patches = Path(__file__).resolve().parent / 'deepjit-patches'
    manifest = json.loads((patches / 'manifest.json').read_text(encoding='utf-8'))
    head = subprocess.check_output(
        ['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip()
    if head != manifest['base_commit']:
        raise SystemExit('DeepJIT HEAD differs from the validated base; no files changed.')

    for entry in manifest["executable_files"]:
        if digest(dependency / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"Unexpected content in {entry['path']}; no files changed.")

    pending = []
    for entry in manifest['files']:
        actual = digest(dependency / entry['path'])
        if actual == entry['after_sha256']:
            continue
        if actual != entry['before_sha256']:
            raise SystemExit(f"Unexpected content in {entry['path']}; preserve and review local edits.")
        pending.append(entry)

    # Check every patch before applying any; never reset or overwrite the dependency.
    for entry in pending:
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(dependency), 'apply', '--check',
                        '--directory=include/deep_jit', str(patches / entry['patch'])], check=True)
    for entry in pending:
        subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(dependency), 'apply',
                        '--directory=include/deep_jit', str(patches / entry['patch'])], check=True)
    for entry in manifest['files']:
        if digest(dependency / entry['path']) != entry['after_sha256']:
            raise SystemExit(f"Post-apply hash mismatch: {entry['path']}")
    # Match the site test-script executable bit; Windows verifies content only.
    if os.name == "posix":
        for entry in manifest["executable_files"]:
            (dependency / entry["path"]).chmod(int(entry["mode"], 8))
    print(f'DeepJIT verified: {len(manifest["files"])} files; applied {len(pending)} patches.')


if __name__ == '__main__':
    main()

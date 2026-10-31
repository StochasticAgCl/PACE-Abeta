"""Local experiment records and content hashes."""
from pathlib import Path
import hashlib
import json
import os


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hashes(paths, root):
    return {str(Path(p).relative_to(root)): digest(p) for p in paths}


def verify(record, root):
    for relative, expected in record['hashes'].items():
        path = Path(root) / relative
        if digest(path) != expected:
            raise RuntimeError(f'Frozen file changed: {path}')

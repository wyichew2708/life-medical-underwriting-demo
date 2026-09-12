"""Verified-evidence registry: the hash list that lets a case go straight through.

The demo sends the ML adapter document hashes only, never document bytes. Certifying
evidence is therefore a human act recorded here, not something the model can infer. A
freshly uploaded report is unknown until someone verifies it, so it blocks acceptance.

This file is a local stand-in for an evidence-management system with real identities,
timestamps and an immutable audit store. It has none of those.
"""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT = Path(__file__).parent / 'artifacts' / 'evidence_registry.json'
WARNING = ('Local demonstration registry. Entries are not authenticated, signed or audited. '
           'Adding a hash asserts that a human verified that exact file.')


def load(path=DEFAULT):
    if not Path(path).exists():
        return {'verified': {}, 'warning': WARNING}
    data = json.loads(Path(path).read_text())
    data.setdefault('verified', {})
    return data


def save(data, path=DEFAULT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data['warning'] = WARNING
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=('add', 'add-hash', 'list', 'remove', 'clear'))
    ap.add_argument('values', nargs='*', help='file paths for add, hex digests for add-hash/remove')
    ap.add_argument('--by', default='local reviewer', help='who verified the evidence')
    ap.add_argument('--note', default='', help='what was checked')
    ap.add_argument('--registry', type=Path, default=DEFAULT)
    args = ap.parse_args()

    data = load(args.registry)
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if args.command in ('add', 'add-hash'):
        for value in args.values:
            digest = sha256(value) if args.command == 'add' else value.strip().lower()
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise SystemExit(f'Not a SHA-256 digest: {value}')
            data['verified'][digest] = {'verified_by': args.by, 'verified_at': now,
                                        'note': args.note or f'Verified from {value}'}
            print(f'verified {digest}')
    elif args.command == 'remove':
        for value in args.values:
            data['verified'].pop(value.strip().lower(), None)
            print(f'removed {value}')
    elif args.command == 'clear':
        data['verified'] = {}
        print('registry cleared')
    else:
        for digest, record in sorted(data['verified'].items()):
            print(f"{digest}  {record.get('verified_at', '?')}  {record.get('verified_by', '?')}")
        print(f"{len(data['verified'])} verified document(s)")
        return
    save(data, args.registry)


if __name__ == '__main__':
    main()

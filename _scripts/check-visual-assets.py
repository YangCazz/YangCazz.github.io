#!/usr/bin/env python3
"""Validate versioned visual manifests without third-party dependencies."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'id', 'path', 'kind', 'title', 'alt', 'source_url', 'verified_at', 'rights', 'used_by'}

def validate(root=ROOT):
    errors, ids, count = [], set(), 0
    manifests = sorted((root / '_data/visual_assets').glob('*.json'))
    if not manifests:
        return ['No visual asset manifests found'], 0
    for manifest in manifests:
        try:
            records = json.loads(manifest.read_text())
            if not isinstance(records, list):
                raise ValueError('root must be an array')
        except (ValueError, OSError) as exc:
            errors.append(f'{manifest.name}: {exc}')
            continue
        for asset in records:
            count += 1
            if not isinstance(asset, dict):
                errors.append(f'{manifest.name}: record must be an object')
                continue
            label = asset.get('id', manifest.name)
            missing = REQUIRED - asset.keys()
            if missing:
                errors.append(f'{label}: missing {sorted(missing)}')
                continue
            if label in ids:
                errors.append(f'{label}: duplicate id')
            ids.add(label)
            path = asset['path']
            if not isinstance(path, str) or not path.startswith('/assets/') or '..' in Path(path).parts:
                errors.append(f'{label}: invalid local asset path')
                continue
            file = root / path.lstrip('/')
            if not file.is_file():
                errors.append(f'{label}: missing asset {path}')
            elif asset.get('sha256') and hashlib.sha256(file.read_bytes()).hexdigest() != asset['sha256']:
                errors.append(f'{label}: SHA-256 mismatch')
            try:
                datetime.date.fromisoformat(asset['verified_at'])
            except (ValueError, TypeError):
                errors.append(f'{label}: invalid verified_at')
            if not isinstance(asset['used_by'], list):
                errors.append(f'{label}: used_by must be an array')
            elif not asset['used_by'] and asset.get('status') != 'archived':
                errors.append(f'{label}: active asset must list at least one source')
            elif asset.get('status') == 'archived' and not asset.get('archive_reason'):
                errors.append(f'{label}: archived asset needs archive_reason')
            else:
                for use in asset['used_by']:
                    if not isinstance(use, str) or Path(use).is_absolute() or '..' in Path(use).parts or not (root / use).is_file():
                        errors.append(f'{label}: missing/invalid source file {use}')
            if asset.get('status') == 'archived':
                active_sources = list(root.glob('*.html'))
                for folder in ['_posts', '_layouts', '_includes']:
                    active_sources.extend(f for f in (root / folder).rglob('*') if f.is_file() and f.suffix in {'.md', '.html'})
                for source in active_sources:
                    if path in source.read_text():
                        errors.append(f'{label}: archived asset still referenced in {source.relative_to(root)}')
            if asset['kind'] == 'generated-illustration' and not asset.get('generation'):
                errors.append(f'{label}: generated image needs generation evidence')
    return errors, count

if __name__ == '__main__':
    errors, count = validate()
    for error in errors:
        print('ERROR:', error)
    print(f'Visual assets: {count} records, {len(errors)} errors')
    sys.exit(bool(errors))

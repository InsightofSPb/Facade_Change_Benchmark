"""Render prepared private answer proposals without approving any label."""

import argparse
import hashlib
import json
from pathlib import Path


def render(payload_path, out):
    source = Path(payload_path).resolve()
    document = json.loads(source.read_bytes())
    if document.get('kind') != 'facade_assistant_review_proposals':
        raise ValueError('Expected a private assistant proposal package')
    if document.get('automatic_approvals') != 0 or document.get('access') != 'private_evaluation':
        raise ValueError('Proposals must remain private and unapproved')
    for key in ('candidate_sha256', 'proposal_sha256'):
        value = document.get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing actual SHA-256 binding: ' + key)
    encoded_cases = json.dumps(document['cases'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if hashlib.sha256(encoded_cases.encode()).hexdigest() != document['proposal_sha256']:
        raise ValueError('Proposal content does not match its frozen hash')
    ids = set()
    for case in document['cases']:
        if case['question_id'] in ids or case.get('status') != 'candidate':
            raise ValueError('Duplicate question or automatic approval')
        ids.add(case['question_id'])
        for event in case['label']['events']:
            x0, y0, x1, y1 = event['bbox_xyxy']
            height, width = case['shape']
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('Proposed box is outside the original crop')
        for observation in case['observations']:
            if observation['media'] not in document['media']:
                raise ValueError('Missing embedded RGB')
    for value in document['media'].values():
        if not value.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,')):
            raise ValueError('Review images must be embedded PNG/JPEG assets')
    template = Path(__file__).with_name('2026-11-16-assistant-review.html').read_text()
    encoded = json.dumps(document, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    result = template.replace('__REVIEW_DATA__', encoded)
    target = Path(out).resolve()
    with target.open('x', encoding='utf-8') as stream:
        stream.write(result)
    return {'status': 'completed', 'proposals': len(ids), 'automatic_approvals': 0,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'html': str(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.proposals, args.out), ensure_ascii=False, indent=2))

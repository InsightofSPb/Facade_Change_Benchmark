"""Private proposal rendering must preserve explicit human approval boundaries."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'assisted_review', Path(__file__).resolve().parents[1] / 'scripts/2026-11-16-render-assisted-review.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssistedReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.document = {
            'kind': 'facade_assistant_review_proposals', 'access': 'private_evaluation',
            'automatic_approvals': 0, 'candidate_sha256': 'c' * 64,
            'cases': [{'question_id': 'case', 'status': 'candidate', 'shape': [16, 16],
                       'label': {'events': []}, 'observations': [{'media': 'rgb'}],
                       'reference_answer': '</script><script>alert(1)</script>'}],
            'media': {'rgb': 'data:image/png;base64,AA=='}}

    def write(self, document):
        data = copy.deepcopy(document)
        encoded = json.dumps(data['cases'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        data['proposal_sha256'] = hashlib.sha256(encoded.encode()).hexdigest()
        source = self.root / '2026-11-16-input.json'
        source.write_text(json.dumps(data))
        return source

    def test_renderer_embeds_private_data_without_executing_text(self):
        source = self.write(self.document)
        target = self.root / '2026-11-16-review.html'
        result = MODULE.render(source, target)
        self.assertEqual(result['automatic_approvals'], 0)
        self.assertNotIn('</script><script>alert(1)</script>', target.read_text())
        self.assertIn('\\u003c/script>', target.read_text())
        with self.assertRaises(FileExistsError):
            MODULE.render(source, target)

    def test_approved_or_modified_proposals_are_rejected(self):
        document = copy.deepcopy(self.document)
        document['cases'][0]['status'] = 'approved'
        with self.assertRaisesRegex(ValueError, 'approval'):
            MODULE.render(self.write(document), self.root / '2026-11-16-invalid.html')
        source = self.write(self.document)
        data = json.loads(source.read_text())
        data['cases'][0]['reference_answer'] = 'changed after binding'
        source.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'frozen hash'):
            MODULE.render(source, self.root / '2026-11-16-invalid.html')

    def test_box_grid_and_external_media_are_rejected(self):
        document = copy.deepcopy(self.document)
        document['cases'][0]['label']['events'] = [{'bbox_xyxy': [0, 0, 17, 16]}]
        with self.assertRaisesRegex(ValueError, 'outside'):
            MODULE.render(self.write(document), self.root / '2026-11-16-invalid.html')
        document = copy.deepcopy(self.document)
        document['media']['rgb'] = 'https://example.org/image.png'
        with self.assertRaisesRegex(ValueError, 'embedded'):
            MODULE.render(self.write(document), self.root / '2026-11-16-invalid.html')

"""Prepared identity checks and honest model-training exposure classification."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.real_eval.night_common import load_frame, load_prepared, training_exposure
from scripts.real_eval.robustness_common import atomic_json, atomic_npz, sha256


class NightCommonTests(unittest.TestCase):
    def test_changed_artifact_and_cross_split_facade_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = dict(frame=0, year=2000, artifact_path='frame.npz')
            arrays = dict(rgb=np.zeros((3, 4, 3), np.uint8), support=np.ones((3, 4), bool),
                          packed=np.ones((3, 4), np.uint16))
            frame['artifact_sha256'] = atomic_npz(root / 'frame.npz', arrays)
            self.assertEqual(load_frame(root, frame)['rgb'].dtype, np.uint8)
            manifest = dict(schema_version=1, kind='real_temporal_prepared', categories=[dict(id=0, name='a')],
                            sequences=[dict(sequence_id='first', building_id='A', split='train',
                                            frames=[frame, {**frame, 'frame': 1, 'year': 2001}]),
                                       dict(sequence_id='second', building_id='A', split='test',
                                            frames=[frame, {**frame, 'frame': 1, 'year': 2001}])])
            atomic_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                load_prepared(root)
            arrays['rgb'][0, 0] = 255
            atomic_npz(root / 'frame.npz', arrays)
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_frame(root, frame)

    def test_semantic_unknown_identities_do_not_become_unseen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic_json(root / 'identities.json', {'images': []})
            atomic_json(root / 'dataset.json', {'manifest': 'identities.json'})
            manifest = {'dataset': str(root / 'dataset.json')}
            origin = dict(training_manifest_verified=True, training_images=2,
                          current_manifest=dict(path=str(root / 'identities.json'), sha256=sha256(root / 'identities.json')),
                          train_overlap_by_current_split={'train': [dict(building_id='A')]},
                          unverified_training_image_ids=['no-address'])
            contracts = {'semantic_dino': {'expected_metadata': {'checkpoint_origin_audit': origin}},
                         'dinov2': {'expected_metadata': {}}}
            result = training_exposure(manifest, contracts)
            self.assertEqual(result['semantic_dino']['building_ids'], ['A'])
            self.assertFalse(result['semantic_dino']['complete'])
            self.assertEqual(result['dinov2']['status'], 'not_project_trained')
            origin['unverified_training_image_ids'] = []
            self.assertTrue(training_exposure(manifest, contracts)['semantic_dino']['complete'])
            atomic_json(root / 'identities.json', {'images': [dict(building_id='B')]})
            different = training_exposure(manifest, contracts)['semantic_dino']
            self.assertEqual(different['building_ids'], [])
            self.assertFalse(different['complete'])

    def test_msdzip_checkpoint_bound_training_buildings(self):
        metadata = dict(checkpoint_sha256='a'*64, buildings={'train': ['A'], 'val': ['B']},
                        reviewed_split=dict(mode='reviewed', building_assignments={'A': 'train', 'B': 'val'}))
        value = training_exposure({}, {'msdzip_abs': {'expected_metadata': metadata}})['msdzip_abs']
        self.assertTrue(value['complete'])
        self.assertEqual(value['building_ids'], ['A'])


if __name__ == '__main__':
    unittest.main()

"""Frozen provenance recovery from explicitly declared wide runs, CPU only."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.real_eval.robustness_common import atomic_json, digest, sha256
from scripts.real_eval.robustness_run import model_identity
from scripts.real_eval.semantic_contracts import resolve_change_contract


class SemanticContractsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        self.weight = self.root / 'weights.bin'
        self.weight.write_bytes(b'fixed checkpoint fixture')

    def fixture(self, method='anychange', *, source_options=None, unit=True, model=True):
        self.method = method
        self.options = {'architecture': 'fixed-v1', 'device': 'cuda:0', 'worker_python': '/recorded/python'}
        self.metadata = {'method': method, 'architecture': 'fixed-v1',
                         'checkpoint': {'path': str(self.weight), 'sha256': sha256(self.weight)},
                         'configured_options': copy.deepcopy(self.options)}
        self.manifest = {'schema_version': 1, 'kind': 'real_temporal_prepared',
                         'source_run': str(self.source), 'source_records': [],
                         'source_config': {'device': 'cuda:0', 'method_options': {
                             method: copy.deepcopy(self.options if source_options is None else source_options)}}}
        atomic_json(self.prepared / 'manifest.json', self.manifest)
        self.reuse = self.root / ('wide-' + method)
        self.reuse.mkdir()
        self.contract = {
            'schema_version': 1, 'source_run': str(self.source), 'methods': [method],
            'prepared_path': str(self.prepared / 'manifest.json'),
            'manifest_sha256': sha256(self.prepared / 'manifest.json'),
            'models': {method: {'options': copy.deepcopy(self.options),
                                'expected_metadata': copy.deepcopy(self.metadata),
                                'model_files_sha256': {str(self.weight): sha256(self.weight)}}} if model else {},
        }
        self.write_run()
        self.row = {'method': method, 'status': 'completed', 'variant': 'frozen', 'domain': 'real',
                    'method_metadata': copy.deepcopy(self.metadata)}
        if unit:
            atomic_json(self.reuse / 'units' / 'completed.json', self.row)

    def write_run(self):
        atomic_json(self.reuse / 'run.json', {'kind': 'real_temporal_run', 'status': 'running',
                    'contract': self.contract, 'config_hash': digest(self.contract)})

    def resolve(self, **kwargs):
        return resolve_change_contract(self.method, self.manifest, 'cuda:1',
                                       reuse_runs=[self.reuse], prepared=self.prepared, **kwargs)

    def test_recovers_three_requested_methods_from_bound_run(self):
        for method in ('anychange', 'dinov2', 'semantic_dino'):
            with self.subTest(method=method):
                self.fixture(method)
                result = self.resolve()
                self.assertEqual(result['options']['device'], 'cuda:1')
                self.assertEqual(result['options']['worker_python'], '/recorded/python')
                self.assertEqual(result['expected_metadata'], self.metadata)
                self.assertEqual(result['model_files_sha256'], {str(self.weight): sha256(self.weight)})
                origin = result['provenance_origin']
                self.assertEqual(origin['kind'], 'declared_reuse_run')
                self.assertEqual(origin['run_json_sha256'], sha256(self.reuse / 'run.json'))
                self.assertEqual(len(origin['completed_unit_metadata']), 1)
                self.assertEqual(self.manifest['source_records'], [])

    def test_complete_model_contract_does_not_require_completed_inference(self):
        self.fixture(unit=False)
        result = self.resolve()
        self.assertTrue(result['provenance_origin']['contract_model_present'])
        self.assertEqual(result['provenance_origin']['completed_unit_metadata'], [])

    def test_completed_unit_only_recovery_requires_recorded_options(self):
        self.fixture(model=False)
        self.assertFalse(self.resolve()['provenance_origin']['contract_model_present'])
        self.row['method_metadata'].pop('configured_options')
        atomic_json(self.reuse / 'units' / 'completed.json', self.row)
        with self.assertRaisesRegex(ValueError, 'no saved model contract'):
            self.resolve()

    def test_night_method_filename_is_used_and_checked(self):
        self.fixture(model=False, unit=False)
        path = self.reuse / 'method_provenance' / f'{self.method}.json'
        atomic_json(path, model_identity(self.metadata))
        result = self.resolve()
        self.assertEqual(result['provenance_origin']['method_provenance_file'], str(path))
        self.assertEqual(result['expected_metadata'], model_identity(self.metadata))
        atomic_json(path, {**model_identity(self.metadata), 'architecture': 'another-model'})
        self.contract['models'][self.method] = {'options': self.options, 'expected_metadata': self.metadata}
        self.write_run()
        with self.assertRaisesRegex(ValueError, 'architecture'):
            self.resolve()

    def test_source_provenance_has_priority_and_invalid_source_is_not_bypassed(self):
        self.fixture()
        path = self.source / 'method_provenance' / f'full_{self.method}.json'
        atomic_json(path, self.metadata)
        self.assertNotIn('provenance_origin', self.resolve())
        wrong = copy.deepcopy(self.metadata)
        wrong['checkpoint']['sha256'] = 'a' * 64
        atomic_json(path, wrong)
        with patch('scripts.real_eval.semantic_contracts._candidate') as candidate:
            with self.assertRaisesRegex(ValueError, 'исходный файл модели'):
                self.resolve()
            candidate.assert_not_called()

    def test_wrong_source_or_prepared_and_changed_contract_are_rejected(self):
        self.fixture()
        for field, value, message in (
            ('source_run', str(self.root / 'other-source'), 'different source_run'),
            ('manifest_sha256', 'a' * 64, 'changed prepared manifest'),
        ):
            with self.subTest(field=field):
                previous = self.contract[field]
                self.contract[field] = value
                self.write_run()
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve()
                self.contract[field] = previous
        self.write_run()
        atomic_json(self.reuse / 'run.json', {'kind': 'real_temporal_run', 'contract': self.contract,
                                            'config_hash': 'b' * 64})
        with self.assertRaisesRegex(ValueError, 'contract checksum changed'):
            self.resolve()

    def test_changed_model_file_and_conflicting_unit_metadata_are_rejected(self):
        self.fixture()
        original = self.weight.read_bytes()
        self.weight.write_bytes(b'replaced weights')
        with self.assertRaisesRegex(ValueError, 'model file missing or changed'):
            self.resolve()
        self.weight.write_bytes(original)
        self.row['method_metadata']['architecture'] = 'different-model'
        atomic_json(self.reuse / 'units' / 'completed.json', self.row)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            self.resolve()

    def test_source_options_and_forbidden_flags_are_not_overridden(self):
        self.fixture(source_options={'architecture': 'different-model'})
        with self.assertRaisesRegex(ValueError, 'conflicting recorded option architecture'):
            self.resolve()
        self.manifest['source_config']['method_options'][self.method] = {'download_weights': True}
        with patch('scripts.real_eval.semantic_contracts._candidate') as candidate:
            with self.assertRaisesRegex(ValueError, 'скачивание запрещено'):
                self.resolve()
            candidate.assert_not_called()

    def test_different_valid_declared_contracts_are_ambiguous(self):
        self.fixture(source_options={})
        second = self.root / 'second-wide'
        alternative = copy.deepcopy(self.contract)
        alternative['models'][self.method]['options']['architecture'] = 'fixed-v2'
        metadata = alternative['models'][self.method]['expected_metadata']
        metadata['architecture'] = 'fixed-v2'
        metadata['configured_options']['architecture'] = 'fixed-v2'
        atomic_json(second / 'run.json', {'kind': 'real_temporal_run', 'contract': alternative,
                                         'config_hash': digest(alternative)})
        with self.assertRaisesRegex(ValueError, 'different frozen model contracts'):
            resolve_change_contract(self.method, self.manifest, 'cuda:1',
                                    reuse_runs=[self.reuse, second], prepared=self.prepared)


if __name__ == '__main__':
    unittest.main()

"""Retention criteria, atomic checkpoint recovery and exact resumable state."""
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class Ledger:
    def __init__(self):
        self.events = []

    def event(self, kind, **values):
        self.events.append(dict(event_type=kind, **values))
        return str(len(self.events))


@unittest.skipIf(torch is None, "Optional CPU Torch is not installed")
class CheckpointRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("facade_change.research.2026-11-08_history_scenarios_checkpoints")

    def save(self, manager, epoch, f1=.5, *, value=None, **metrics):
        weights = {"weight": torch.tensor([float(epoch if value is None else value)])}
        return manager.save_epoch(weights, dict(steps=epoch, optimizer={"step": epoch},
            scheduler={"last_epoch": epoch}, scaler={}, rng={"seed": 13}, stale=0),
            dict(f1=f1, iou=f1 / 2, recall=.7, fpr=.04, **metrics), epoch)

    def test_top_three_rank_f1_not_threshold_recall_and_secondary_winners(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract")
            self.save(manager, 1, .8, prediction_score=4., semantic_miou=.9)
            self.save(manager, 2, .9, prediction_score=5., semantic_miou=.8)
            self.save(manager, 3, .7, prediction_score=3., semantic_miou=.7)
            selected = self.save(manager, 4, .6, prediction_score=2., semantic_miou=.6)
            self.assertEqual([p["epoch"] for p in selected["winners"]["f1_top3"]], [2, 1, 3])
            self.assertEqual(selected["checkpoint"]["epoch"], 2)
            self.assertEqual(selected["prediction_checkpoint"]["epoch"], 4)
            self.assertEqual(selected["semantic_checkpoint"]["epoch"], 1)
            self.assertEqual(selected["checkpoint"]["metrics"]["prediction_score"], 5.)
            self.assertEqual(len(selected["retained"]), 4)
            self.assertEqual(len(list(Path(root).glob("*resume-*.pt"))), 1)
            for pointer in selected["retained"]:
                payload = self.module._load(pointer["path"])
                self.assertIn("model", payload)
                self.assertIn("restore_contract", payload)
                for field in ("optimizer", "scheduler", "scaler", "rng"):
                    self.assertNotIn(field, payload)

    def test_identical_state_dict_has_one_weights_file_and_multiple_criteria(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract")
            self.save(manager, 1, value=17., prediction_score=1., semantic_miou=.8)
            result = self.save(manager, 2, value=17., prediction_score=1., semantic_miou=.8)
            self.assertEqual(len(result["retained"]), 1)
            self.assertEqual(len(list(Path(root).glob("*weights-*.pt"))), 1)
            self.assertEqual(result["checkpoint"]["epoch"], 1)
            self.assertEqual(set(result["retained"][0]["reasons"]),
                             {"f1_rank_1", "iou", "recall", "prediction_score", "semantic_miou"})

    def test_invalid_fpr_cannot_rank_primary_and_nan_is_na(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract")
            result = manager.save_epoch({"w": torch.ones(1)}, {},
                dict(f1=.99, iou=.9, recall=1., fpr=.051,
                     prediction_score=float("nan"), semantic_miou=float("inf")), 1)
            self.assertIsNone(result["checkpoint"])
            self.assertEqual(result["retained"], [])
            self.assertEqual(manager.resume()["epoch"], 1)

    def test_retirement_is_append_only_and_only_live_artifacts_are_required(self):
        ledger = Ledger()
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract", ledger)
            first = self.save(manager, 1, .1)
            first_events = list(ledger.events)
            for epoch in range(2, 9):
                self.save(manager, epoch, epoch / 10)
            result = manager.finalize()
            self.assertEqual(ledger.events[:len(first_events)], first_events)
            self.assertFalse(Path(first["latest"]["path"]).exists())
            # Recall ties preserve epoch 1; it stays a separate immutable winner.
            self.assertTrue(Path(first["checkpoint"]["path"]).exists())
            self.assertEqual(len(result["retained"]), 4)
            live = self.module.live_checkpoint_artifacts(ledger.events)
            self.assertEqual(set(live), {result["latest"]["path"], *(r["path"] for r in result["retained"])})
            self.assertTrue(all(Path(path).is_file() for path in live))
            self.assertTrue(any(e["event_type"] == "artifact_retired" for e in ledger.events))
            self.module.CheckpointManager(root, "contract", ledger).resume()

    def test_every_atomic_boundary_recovers_or_keeps_previous_epoch(self):
        for stage in ("after_winner_write", "before_resume_write", "after_resume_write",
                      "before_pointer_commit", "after_selection_pointer", "after_latest_pointer",
                      "before_retirement"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                manager = self.module.CheckpointManager(root, "contract", Ledger())
                first = self.save(manager, 1, .4)
                old_resume = Path(first["latest"]["path"])
                def stop(boundary):
                    if boundary == stage:
                        raise KeyboardInterrupt(boundary)
                with mock.patch.object(self.module, "_fault", side_effect=stop):
                    with self.assertRaises(KeyboardInterrupt):
                        self.save(manager, 2, .8)
                # No payload has been deleted before the transaction can commit.
                self.assertTrue(old_resume.exists())
                recovered = self.module.CheckpointManager(root, "contract", manager.ledger)
                expected = 1 if stage in ("after_winner_write", "before_resume_write") else 2
                self.assertEqual(recovered.resume()["epoch"], expected)
                if expected == 1:
                    self.save(recovered, 2, .8)
                self.assertFalse(old_resume.exists())
                self.assertFalse(recovered.journal_path.exists())
                self.assertEqual(len(list(Path(root).glob("*resume-*.pt"))), 1)
                final = recovered.finalize()
                self.assertEqual(len(list(Path(root).glob("*weights-*.pt"))), len(final["retained"]))

    def test_partial_payload_does_not_replace_latest_or_remove_winner(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract", Ledger())
            first = self.save(manager, 1, .4)
            original = manager._atomic_payload
            def partial(path, payload):
                if "resume-" in path.name:
                    path.with_name(path.name + ".tmp").write_bytes(b"incomplete")
                    raise KeyboardInterrupt()
                return original(path, payload)
            with mock.patch.object(manager, "_atomic_payload", side_effect=partial):
                with self.assertRaises(KeyboardInterrupt):
                    self.save(manager, 2, .8)
            recovered = self.module.CheckpointManager(root, "contract", manager.ledger)
            self.assertEqual(recovered.resume()["epoch"], 1)
            self.assertTrue(Path(first["checkpoint"]["path"]).exists())
            self.assertFalse(list(Path(root).glob("*.pt.tmp")))

    def test_missing_live_winner_tamper_and_foreign_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract")
            first = self.save(manager, 1)
            with self.assertRaisesRegex(ValueError, "contract"):
                self.module.CheckpointManager(root, "different")
            with self.assertRaisesRegex(ValueError, "foreign"):
                manager._retire({"path": str(Path(root).parent / "unrelated.pt")}, "test")
            Path(first["checkpoint"]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                manager.resume()
            Path(first["checkpoint"]["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                manager.finalize()

    def test_full_resume_restores_model_optimizer_scheduler_scaler_and_rng(self):
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        def step(net, opt, sched, scale):
            opt.zero_grad(set_to_none=True)
            noise = random.random() + np.random.random()
            loss = net(torch.randn(3, 2)).square().mean() * noise
            scale.scale(loss).backward()
            scale.step(opt)
            scale.update()
            sched.step()

        for _ in range(2):
            step(model, optimizer, scheduler, scaler)
        state = dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), steps=2, stale=1, best=(.6,),
                     history=[{"epoch": 1}, {"epoch": 2}],
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))
        with tempfile.TemporaryDirectory() as root:
            manager = self.module.CheckpointManager(root, "contract", restore_contract={"channels": 2})
            manager.save_epoch(model.state_dict(), state, dict(f1=.6, iou=.5, recall=.7, fpr=.04), 2)
            step(model, optimizer, scheduler, scaler)
            restored = manager.resume()
            resumed_model = torch.nn.Linear(2, 2)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=.01)
            resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=6)
            resumed_scaler = torch.amp.GradScaler("cpu", enabled=True)
            resumed_model.load_state_dict(restored["model"])
            resumed_optimizer.load_state_dict(restored["optimizer"])
            resumed_scheduler.load_state_dict(restored["scheduler"])
            resumed_scaler.load_state_dict(restored["scaler"])
            random.setstate(restored["rng"]["python"])
            np.random.set_state(restored["rng"]["numpy"])
            torch.set_rng_state(restored["rng"]["torch"])
            step(resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler)
            self.assertEqual(restored["steps"], 2)
            self.assertEqual(restored["stale"], 1)
            self.assertEqual(restored["history"], [{"epoch": 1}, {"epoch": 2}])
            self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(resumed_scaler.state_dict(), scaler.state_dict())
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(resumed_model.state_dict()[name], tensor, rtol=0, atol=0)
            for key, values in optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(resumed_optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import yaml

    from scripts.hst_offline_eval import eval_checkpoint, latest_checkpoint
    from trainer.train_hst_pretrain import CharTokenizer, TinyCausalLM, TrainConfig
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class OfflineEvalTest(unittest.TestCase):
    def test_eval_checkpoint_from_run_dir(self):
        scratch_parent = Path.cwd() / "hst_tmp"
        scratch_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_parent) as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "hst_runs" / "run_a"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            data_path = tmp_path / "tiny.jsonl"
            rows = [{"text": "alpha beta gamma"}, {"text": "delta epsilon zeta"}]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            cfg = TrainConfig(
                method="ntp_baseline",
                data_path=str(data_path),
                output_dir=str(run_dir),
                max_steps=1,
                batch_size=2,
                max_seq_len=16,
                baseline_seq_len=16,
                hidden_size=16,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )
            tokenizer = CharTokenizer([row["text"] for row in rows])
            model = TinyCausalLM(tokenizer.vocab_size, cfg.baseline_seq_len, cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.dropout)
            checkpoint = ckpt_dir / "step_1.pt"
            torch.save({"model": model.state_dict(), "step": 1, "config": cfg.__dict__}, checkpoint)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.__dict__, sort_keys=True), encoding="utf-8")

            self.assertEqual(latest_checkpoint(run_dir), checkpoint)
            result = eval_checkpoint(checkpoint, run_dir, "cpu", 1)
            self.assertEqual(result["run"], "run_a")
            self.assertEqual(result["method"], "ntp_baseline")
            self.assertEqual(result["step"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(result["loss_eval_ntp"])))


if __name__ == "__main__":
    unittest.main()

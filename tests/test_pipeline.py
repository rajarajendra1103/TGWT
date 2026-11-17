"""
End-to-end pipeline verification test.
"""

import unittest
import torch
from tgwt.model.tgwt_model import TGWTForCausalLM
from tgwt.model.baseline_llama import BaselineTransformerLM
from tgwt.benchmark.metrics import benchmark_model
from tgwt.benchmark.compare import create_matched_models


class TestEndToEndPipeline(unittest.TestCase):

    def test_matched_parameter_parity(self):
        tgwt, baseline = create_matched_models(
            vocab_size=1000,
            max_seq_len=128,
            d_model=64,
            n_heads=4,
            n_layers=2,
            walk_length=16,
        )

        p_tgwt = tgwt.get_num_params()
        p_base = baseline.get_num_params()
        diff_pct = abs(p_tgwt - p_base) / max(p_tgwt, p_base) * 100.0
        # Check within +/- 2%
        self.assertLessEqual(diff_pct, 2.0)

    def test_benchmark_metrics_computation(self):
        B, N, V = 2, 32, 500
        tgwt = TGWTForCausalLM(
            vocab_size=V,
            max_seq_len=64,
            d_model=64,
            n_heads=4,
            n_layers=2,
            walk_length=16,
        )

        input_ids = torch.randint(0, V, (B, N), dtype=torch.long)
        labels = input_ids.clone()

        res = benchmark_model(
            model=tgwt,
            model_name="TGWT",
            input_ids=input_ids,
            labels=labels,
            n_layers=2,
            d_model=64,
            walk_length=16,
        )

        self.assertIn("perplexity", res)
        self.assertIn("throughput_tokens_per_sec", res)
        self.assertIn("peak_vram_mb", res)
        self.assertIn("flops_per_token", res)
        self.assertIn("accuracy_pct", res)
        self.assertGreater(res["throughput_tokens_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for full TGWT model and parameter-matched baseline.
"""

import unittest
import torch
from tgwt.model.tgwt_model import TGWTForCausalLM
from tgwt.model.baseline_llama import BaselineTransformerLM


class TestModelPipeline(unittest.TestCase):

    def test_tgwt_forward_backward(self):
        B, N, V = 2, 32, 1000
        model = TGWTForCausalLM(
            vocab_size=V,
            max_seq_len=64,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_prop_node=16,
            d_prop_edge=16,
            walk_length=16,
            branch_factor=4,
            oversample_ratio=1.5,
        )

        input_ids = torch.randint(0, V, (B, N), dtype=torch.long)
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)

        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logits"].shape, (B, N, V))
        self.assertIsNotNone(out["loss"])

        # Test backward pass
        out["loss"].backward()

        # Check gradients exist
        has_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
        self.assertTrue(has_grads)

    def test_baseline_forward_backward(self):
        B, N, V = 2, 32, 1000
        model = BaselineTransformerLM(
            vocab_size=V,
            max_seq_len=64,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )

        input_ids = torch.randint(0, V, (B, N), dtype=torch.long)
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)

        self.assertEqual(out["logits"].shape, (B, N, V))
        self.assertIsNotNone(out["loss"])
        out["loss"].backward()

    def test_tgwt_gradient_causality(self):
        """
        Gradient Causality Unit Test:
        Pass random tensor x of shape (1, 64, d_model) with requires_grad=True,
        compute loss ONLY at position t = 32, call .backward(),
        and assert that x.grad[:, 33:, :].abs().max() == 0.0.
        """
        torch.manual_seed(42)
        d_model = 64
        model = TGWTForCausalLM(
            vocab_size=1000,
            max_seq_len=64,
            d_model=d_model,
            n_heads=2,
            n_layers=2,
            walk_length=16,
            branch_factor=4,
        )
        model.eval()

        x = torch.randn(1, 64, d_model, requires_grad=True)
        out = model(inputs_embeds=x)

        # Compute loss ONLY at position t = 32
        loss = out["logits"][:, 32, :].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        # 1. Assert future tokens [33..63] have strictly 0.0 gradient (zero causal leakage)
        future_grad = x.grad[:, 33:, :]
        max_future_grad = future_grad.abs().max().item()
        self.assertEqual(
            max_future_grad,
            0.0,
            f"Causal leakage detected! Max gradient on future tokens: {max_future_grad}",
        )

        # 2. Assert past/current tokens [0..32] receive proper gradients
        past_grad = x.grad[:, :33, :]
        self.assertGreater(
            past_grad.abs().max().item(),
            0.0,
            "Gradients on past/current tokens must be non-zero.",
        )


if __name__ == "__main__":
    unittest.main()

"""fit(): the mid-run snapshot hook fires on schedule and never on the last step.

The writer (run_arm_lib.save_release) needs a real release directory, so this
test only pins the contract fit() owns: `on_snapshot(step, loss)` is called
every `save_every` steps, with the step number the update just finished, and is
NOT called on the final step (the caller's own save covers that one).

    python -m pytest tests/test_fit_snapshot.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from serve.runtime import keep_fused_kernels_off                    # noqa: E402

keep_fused_kernels_off("cpu")      # the torch DeltaNet reference on CPU, not fla's Triton kernels

from rsijev.arch import ArchConfig, DecisionModel                  # noqa: E402
from rsijev.contract import Case, Question                          # noqa: E402
from rsijev.encode import EncodeConfig                              # noqa: E402
from rsijev.fit import FitConfig, fit                               # noqa: E402
from rsijev.train import seed_everything                            # noqa: E402

BASE = "Qwen/Qwen3.5-0.8B"       # tokenizer + layer mix; any Qwen3.5 shares the text tower


def _tiny(hidden=64, layers=8):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE)
    tc = getattr(cfg, "text_config", cfg)
    for k, v in dict(num_hidden_layers=layers, hidden_size=hidden, intermediate_size=128,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                     linear_num_key_heads=2, linear_num_value_heads=2,
                     linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=None).items():
        if v is not None and hasattr(tc, k):
            setattr(tc, k, v)
    tc.layer_types = [("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                      for i in range(layers)]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tc).float().eval()


def _cases(n=12):
    out = []
    for i in range(n):
        q = Question(key="q", mode="noul", instructions=f"Is item {i} relevant to the request?",
                     options=("false", "true"),
                     criteria={"false": "not relevant", "true": "relevant"})
        out.append(Case(case_id=f"c{i}", source="dc_test",
                        state=f"User asks about topic {i}. " * 6, questions=(q,),
                        gold={"q": (1.0, 0.0) if i % 2 else (0.0, 1.0)}))
    return out


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE)


def _model(lm, exit_layer=4):
    tower = lm.model
    for p in tower.parameters():
        p.requires_grad_(True)
    seed_everything(17)
    return DecisionModel(tower, lm.config.get_text_config().hidden_size,
                         ArchConfig(readout="option_xattn", max_options=8, freeze_base=False,
                                    option_pool="mean", residual=False, xattn_combine="mlp",
                                    readout_layer=-1, exit_layer=exit_layer))


def _run(tok, cfg_steps, save_every, n=12):
    m = _model(_tiny())
    seen = []
    fit(m, tok, _cases(n), EncodeConfig(layout="state_first", option_pool="mean"),
        FitConfig(steps=cfg_steps, batch_size=4, log_every=1, autocast_bf16=False,
                  save_every=save_every),
        max_options=8, seed=17, device="cpu",
        on_snapshot=lambda step, loss: seen.append((step, loss)))
    return seen


def test_snapshots_fire_every_n_steps_but_not_on_the_last(tok):
    seen = _run(tok, cfg_steps=7, save_every=2)
    assert [s for s, _ in seen] == [2, 4, 6], seen       # 8 would be past the end
    assert all(torch.isfinite(torch.tensor(l)) for _, l in seen)


def test_disabled_by_default(tok):
    assert _run(tok, cfg_steps=3, save_every=0) == []


def test_save_every_larger_than_run_never_fires(tok):
    assert _run(tok, cfg_steps=3, save_every=10) == []

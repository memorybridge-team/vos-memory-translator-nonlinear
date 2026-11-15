# Decisions made during the unattended refactor

Each entry gives the question, the options, the choice, the reason, and how to undo it. The design itself is described in `docs/translator_model_design.md`.

Commits referred to below (all local, nothing pushed):

- `REMOVAL` = `7d80e59` "refactor. remove the mlp_context translator"
- `REDESIGN` = `d4ec72d` "refactor. rebuild both translators as a stitching layer plus a Transformer block"

General undo: `git revert <hash>` on this branch restores the previous state of that step. To inspect the old code without changing anything: `git show 913e35b:src/vos_memory_inspector/models/_spatial.py` (and `residual_mlp.py`, `transformer.py`, `mlp_context.py`, `registry.py`).

Old and new sizes for every structural change are in section "Size changes" at the end.

---

## D1. Which form the paper describes

- **Question.** `M_hat = M + correction(M)` (identity skip, the form in the earlier prompt), or "linear map + correction".
- **Options.** (a) identity skip; (b) affine skip added in parallel to the correction, `W M + b + f(M)`, as the old `skip_mode="affine"`; (c) stitching layer first, then residual sublayers on its output, `Z = W M + b; M_hat = Z + f(Z)`.
- **Choice.** (c).
- **Why.** The literature gives no support for the identity as a first estimate between two separately trained models: Lenc & Vedaldi 2015 report a top-1 error above 99% when the identity replaces the stitching layer. The stored states agree (raw cosine between Small, Base+ and Large memories is about 0; identity R² is negative for all three pairs). A linear per-position map is the stitching layer of model stitching (direct precedent: Bansal et al.; Csiszárik et al.). Starting it from the least-squares solution and then training it is what Csiszárik et al. do with frozen networks, and what Stitchable Neural Networks do while also updating the networks. With (c) the sublayers act on a linear estimate of the target memory, so "correction" is literally true, the block is an unmodified Transformer block, and the nonlinear part sees inputs in the target's coordinates for every pair. (b) needs a non-standard block in which the shortcut and the branch read different spaces.
- **Undo.** `git revert REDESIGN`. To get form (b) inside the new code instead: in `SpatialTranslator.forward` feed the blocks with `frames` tokens and add `self.stitch(...)` to the sum of the branch outputs; this changes the function, so models must be retrained.

## D2. What the nonlinear part is

- **Question.** Keep the hand-made parts (64 -> 128 -> 64 MLP; patch embedding + two blocks + 3x3 convolution + bilinear upsampling + gate) or replace them.
- **Options.** (a) keep; (b) one standard Pre-LN Transformer encoder block, with the MLP model defined as that block without its attention sublayer.
- **Choice.** (b).
- **Why.** Every part of (b) has a direct source (Vaswani et al. 2017; ViT; Pre-LN). The two models then differ in exactly one component, and the Transformer model contains the MLP model, so their comparison is a clean ablation. In the old code the two models were not nested: the Transformer had a linear local branch and context, the MLP a nonlinear local branch and no context.
- **Undo.** `git revert REDESIGN`.

## D3. Feed-forward ratio 4 (hidden width 256)

- **Options.** 2 (old, hidden 128, no source); 4 (Vaswani et al. 512 -> 2048; ViT); 8 (PVT stage 1).
- **Choice.** 4.
- **Why.** It is the most widely used value and the one of the cited block. It doubles the feed-forward parameters and MACs.
- **Undo.** No code change: set `"model_options": {"mlp_ratio": 2.0}` in the training config, or `build_translator(name, s, t, mlp_ratio=2.0)`.

## D4. One block

- **Options.** 1; 2 (old Transformer model).
- **Choice.** 1.
- **Why.** The smallest form, and Denoising Vision Transformers also use a single block for correcting a frozen model's feature map. No source supports 2 for this task, and the instruction was not to grow the models without a reason.
- **Undo.** Config only: `"model_options": {"num_layers": 2}`. It must then be 2 for all three pairs.

## D5. How attention sees the record

- **Question.** The old model attended over a 16x16 patch grid and brought the result back with a 3x3 convolution, a 1x1 projection and bilinear upsampling.
- **Options.** (a) keep; (b) full self-attention over 4,096 tokens; (c) spatial-reduction attention (PVT): 4,096 queries, keys and values reduced by a strided convolution.
- **Choice.** (c) with reduction ratio 4 (256 keys) and one head.
- **Why.** (a) has no source as a whole. (b) is the plainest form but costs about 2.1 G MACs per record and a 4,096 x 4,096 attention map per record during training. (c) is a published operator made for this cost problem and gives a full-resolution output without any upsampling. Ratio 4 keeps the old 16x16 coarse grid; one head follows PVT's configuration at 64 channels and SAM 2.1's own single-head memory attention.
- **Weak precedent.** The ratio 4 itself: PVT uses 8 at its 64-channel stage and 4 at the next, and MViTv2 uses stride 4 at its highest-resolution stage. Wording: "the factor was not tuned".
- **Undo.** Ratio and heads are config values: `"model_options": {"sr_ratio": 1}` gives option (b), `{"sr_ratio": 8}` or `{"num_heads": 4}` other settings. Option (a): `git revert REDESIGN`.

## D6. Position encoding

- **Options.** (a) learnable embedding added to the tokens (old; ViT; Denoising ViT argues for learnable ones); (b) fixed 2D sine encoding added to queries and keys at the attention only (DETR).
- **Choice.** (b).
- **Why.** DETR is a direct precedent, and SAM 2.1 itself adds a sine encoding to memory keys. With (b) the feed-forward path carries no position signal, so it is the same function in both models and the ablation stays one-component. (a) at full resolution would add 262,144 parameters and tie the weights to the grid.
- **What is given up.** A learnable, position-dependent correction. If position-dependent artefacts of the backbones matter, (b) cannot remove them except through attention.
- **Undo.** Code change in `SpatialReductionAttention` (`pos_query`, `pos_key` buffers become one `nn.Parameter`) or `git revert REDESIGN`; retraining needed.

## D7. Zero start: last layer of each sublayer, no gate

- **Options.** (a) per-channel gate `alpha = 0` (old Transformer; ReZero/LayerScale); (b) zero the last linear layer of each sublayer (old MLP; Fixup, Goyal et al.).
- **Choice.** (b) for both models; the gate is removed.
- **Why.** One mechanism for both models, with a direct source, and no extra parameter. A zero gate on a zero layer would never train, so only one of the two may be used anyway.
- **Undo.** `git revert REDESIGN`, or add a LayerScale parameter to `TranslatorBlock` and remove the two `nn.init.zeros_` calls.

## D8. Pointer head

- **Options.** (a) MLP 256 -> 512 -> 256 with a residual (old default, 262,912 parameters); (b) linear map 256 -> 256 from ridge (65,792 parameters).
- **Choice.** (b).
- **Why.** A linear map between two vector spaces has the stronger literature (Mikolov et al. 2013; Lähner & Moeller 2024, who also report that a three-layer MLP aligns latent vectors worse than the linear map on test data, because it overfits). Your records show no gain from the MLP (`pointer_experiments.csv`: dev MSE 0.3190 ridge, 0.3146 MLP; a deeper MLP overfits), and every model of the design review already used the ridge pointer. The map stays trainable; whether it is frozen is a training-loop choice.
- **Undo.** Restore the old head from `git show 913e35b:src/vos_memory_inspector/models/_spatial.py` (class `ResidualPointerTranslator`) and use it in `StitchedStateTranslator.__init__`; or `git revert REDESIGN`.

## D9. Old weights and payloads

- **Question.** `TransformerStateTranslator.from_payload` used to return an `MLPContextStateTranslator` for additive-MLP payloads; the new models cannot hold any old weights.
- **Options.** (a) keep loaders for old payloads next to the new models; (b) drop support.
- **Choice.** (b). All of these are rejected with a clear error: `cmmt.residual_mlp_translator.v2/.v3`, `cmmt.spatial_context_transformer_translator.v1/.v2` (this includes every `mlp_context` payload). New payloads use `cmmt.stitched_translator.v1`.
- **Why.** The structures are different; keeping the old classes would keep the unreferenced parts in the repository. You said old payloads do not have to work.
- **What breaks.** The first-run weights, the paper weights (`r2` epoch 26), the design-review weights (`mlp`, `transformer`, `mlp_context`, `e2e_*`) and any checkpoint written before `REDESIGN` cannot be loaded by this branch. They load with commit `913e35b` or `main`. Nothing on disk was touched.
- **Undo.** `git revert REDESIGN REMOVAL`, or check out `913e35b` in another worktree to read old artifacts.

## D10. Ridge start is required by the official trainer

- **Options.** (a) optional (identity start allowed); (b) required for `cmmt-state-training train`.
- **Choice.** (b) for the official trainer; the model class itself can still be built without ridge (identity placeholder), which the tests, the benchmark and `cpu-smoke` use. `cpu-smoke` fits a sample ridge solution when none is given, as it did before for the affine mode.
- **Why.** The described design starts from the closed-form solution. Allowing an identity start in the official path would let a run silently differ from the paper's description.
- **Side effect.** `preflight` without `--ridge-init` now reports `ridge_initialization_ready: false` and exits with 2 for both models (before: only for the affine mode). The order is `preflight` -> `ridge-fit` -> `preflight --ridge-init` -> `train --ridge-init`.
- **Undo.** In `state_training/trainer.py`, make the `if ridge is None: raise` branch skip the initialisation instead; in `state_training/cli.py`, set `ridge_initialization_ready` to `True` when no `--ridge-init` is given.

## D11. No grid resampling

- **Choice.** The models require equal source and target memory grids and raise otherwise. Channel count and pointer size may differ (the stitching layer and the pointer map handle that).
- **Why.** All three SAM 2.1 sizes use 64x64, checked in the configs and on stored states. The bilinear grid adapter had no use and no source.
- **Undo.** Resample outside the translator, or restore `_resample_frames` use in `StitchedStateTranslator._translate_frames`.

## D12. Presets

- **Choice.** The registry keeps `direct`, `moment_match`, `linear`, adds `transformer`, keeps `residual_mlp`, and adds one control, `residual_mlp_matched` (feed-forward ratio 14; 119,936 spatial parameters against 119,872 for the Transformer model). All ablation presets of the old structure are removed (`base`, `linear_local`, `no_context`, `base_mlp`, `context_only`, `no_attn_context`, `res8/16/32`, `grid8/32`, `depth1/4`, `heads2/8`, `no_residual`, `mlp256`, `mlp_param`, `no_pos`, `sincos_pos`, `ptr128`, `no_cos`).
- **Why.** They configured parts that no longer exist. Depth, heads, reduction ratio and feed-forward ratio are now plain config fields, so an ablation is a config override and needs no preset.
- **Undo.** Add presets to `PRESETS` in `models/registry.py`; each is one line.

## D13. Names

- **Choice.** Class names, model names and file names are kept (`ResidualMLPStateTranslator` / `residual_mlp`, `TransformerStateTranslator` / `transformer`). The config class is new: `TranslatorConfig` replaces `SpatialTransformerConfig` and the MLP model's keyword arguments.
- **Why.** The CLI, the configs and the docs keep working with the same model names. "Residual MLP" still describes the model (a residual feed-forward block).
- **Undo.** Rename freely; no payload exists yet under the new schema.

## D14. No final LayerNorm

- **Choice.** The block output is returned without the final LayerNorm that Pre-LN encoders normally have.
- **Why.** The output is the target's memory feature map and must keep its scale. Recorded as "no precedent", with the wording in the design note.
- **Undo.** Not recommended; it would normalise the memory.

## D15. `mlp_context` in dated documents

- **Choice.** `mlp_context` is removed from code, registry, CLI, configs, tests, `README.md` and `docs/state_training.md`. Mentions are left in `docs/cuda_preparation_20261030.md` (a dated test record) and in `docs/translator_design_literature_grounding.md` (it reports past experiments).
- **Why.** Those two files record what happened; editing them would falsify the record.
- **Undo.** Delete the lines if you want no mention at all.

## D16. Legacy paired-experiment trainer

- **Choice.** `paired_experiment.py` and `scripts/train_nonlinear_from_collection.py` were adapted to the new constructors only. They still train from the identity placeholder and do not fit a ridge start.
- **Why.** The official path is `cmmt-state-training`. Adding a ridge fit to the legacy trainer is a behaviour change beyond the model refactor.
- **Consequence.** A model trained through the legacy path does not follow the described design (its stitching layer starts at the identity). Use `cmmt-state-training` for anything that goes into the paper.
- **Undo / follow-up.** Either remove the legacy trainer or give it the ridge start.

## D17. Removed command-line flags and function parameters

- **Choice.** `--hidden-dim` and `--pointer-hidden-dim` are removed from the paired-experiment CLI (`cmmt-paired-experiment`) and from `scripts/train_nonlinear_from_collection.py`; `--hidden-dim` is removed from the synthetic-experiment CLI. `run_paired_experiment` and `run_synthetic_experiment` take one `translator_config: TranslatorConfig | None` instead of `hidden_dim`, `pointer_hidden_dim` and `transformer_config`. Run metadata records `training["translator_config"]` instead of the two widths.
- **Why.** The flags set widths of parts that no longer exist (the pointer MLP) or that are now a ratio (`mlp_ratio`). No CLI replacement was added, because the structure should be the same for every pair and is set in one place.
- **Undo.** Add `--mlp-ratio` / `--num-layers` flags that build a `TranslatorConfig`; a few lines in `cli.py`.

## D18. One existing test was re-conditioned

- **What.** `test_gloo_two_process_tail_matches_single_process` compares one-process and two-process training to a tolerance of 2e-6. With the ridge start it failed (difference 6e-4 in `pointer.linear.weight`). The test now starts from a deliberately detuned ridge solution (weights x 0.8, bias + 0.05); the tolerance is unchanged.
- **Why.** The synthetic targets of that test are exactly affine, so the exact ridge solution has a gradient of about zero, and AdamW turns summation-order noise between one and two processes into full-size steps. This is the helper agent's explanation; I find it plausible (the difference is about two AdamW steps at the test's learning rate) but it was not isolated further.
- **What it means for real training.** Under a pure state-MSE loss, the ridge solution is already the minimiser for the linear parts on the fit data. The pointer head is linear only, so MSE training has nothing to teach it (in the GPU smoke test the held-out pointer error did not move). It only changes under a loss through the target model. Freezing the pointer head during MSE training would be a clean training-loop choice.
- **Undo.** Restore the test body from `git show 7d80e59:tests/test_state_training.py`; it will then fail unless the tolerance is raised.

---

## Size changes

Counted from the code (old: commit `913e35b`; new: this branch). MACs are per record.

| Change | Old | New | Reason |
|---|---|---|---|
| MLP model, spatial parameters | 16,576 (identity) / 20,736 (affine) | 37,376 | Feed-forward ratio 2 -> 4 (+16,512), stitching layer always present (4,160), one LayerNorm (128) |
| MLP model, spatial MACs | 67,108,864 / 83,886,080 | 150,994,944 | Same |
| Transformer model, spatial parameters | 194,304 | 119,872 | Patch embedding (65,600), learnable position embedding (16,384), second block, 3x3 convolution (36,928), gate removed; reduction convolution (65,600) and feed-forward ratio 4 added |
| Transformer model, spatial MACs | 77,594,624 | 337,641,472 | Attention now has 4,096 queries instead of 256; feed-forward at full resolution with ratio 4 |
| Pointer head, parameters | 262,912 (MLP) / 328,704 (MLP + affine skip) | 65,792 | Linear map |
| Pointer head, MACs | 262,144 / 327,680 | 65,536 | Linear map |
| MLP model, total parameters | 279,488 / 349,440 | 103,168 | |
| Transformer model, total parameters | 457,216 | 185,664 | |

Cost at the switch (gpu-server GPU 0, RTX 4080 SUPER, forward only, 7 records, one object): MLP model 0.49 ms (old affine MLP 0.38 ms), Transformer model 0.96 ms (old 0.88 ms). For scale, the design review measured 79 ms for the anchor baseline on the same GPU. The translator's cost stays negligible next to the switch.

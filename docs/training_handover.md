# Training handover

For the person who trains and evaluates the translators. The model design is finished and frozen; this page says how to train it and what must stay as it is. The commands below were checked against the argument parsers (`--help`) and the code.

**Status, 2026-11-15:** the `transformer` model for Small -> Base+ was trained with the official trainer on gpu-server (2 GPUs, `configs/state_training/transformer.json`). Early stopping ended the run at epoch 35, and epoch 27 was selected with validation state loss 0.4908 (ridge 0.5219). It was then fine-tuned through the frozen Base+ model and evaluated by J&F on PUMaVOS and M3-VOS; the method, sources and results are in `docs/translator_model_design.md`, section 7. The `residual_mlp` model and the pairs Base+ -> Large and Small -> Large have not been trained.

Background: `docs/translator_model_design.md` (structure and sources), `docs/refactor_decisions.md` (what changed and why), `docs/state_training.md` (every option of the trainer).

## 1. What you are training

| Name | `--config` | What it is | Why it exists |
|---|---|---|---|
| `residual_mlp` | `configs/state_training/residual_mlp.json` | Stitching layer + one Pre-LN feed-forward block; linear pointer | The MLP model: a per-position map, no spatial context |
| `transformer` | `configs/state_training/transformer.json` | The same, with a self-attention sublayer in the block | The Transformer model. It differs from the MLP model in the attention sublayer only, so the pair is an ablation of spatial context |
| `residual_mlp_matched` | `configs/state_training/residual_mlp_matched.json` | `residual_mlp` with feed-forward ratio 14 | A control with the Transformer model's parameter count (119,936 against 119,872 spatial parameters). It answers "is a gain from attention or from size?". It is not a third model for the paper |
| ridge | (the output of `ridge-fit`) | The closed-form linear map alone | The baseline. Both models start from it, so a model that does not beat it has learned nothing |

All four must be trained and evaluated on the same data, split and seed, or the comparison means nothing.

## 2. Commands for one pair

Use the official trainer only (`cmmt-state-training`, or `PYTHONPATH=src python -m vos_memory_inspector.state_training.cli` before installation). Do not use `cmmt-paired-experiment` or `scripts/train_nonlinear_from_collection.py`: that older trainer does not apply the ridge start, so its models do not follow the described design.

Both models need the ridge start. The order is fixed:

```sh
# 1. Check the data (no ridge yet). Exit code 2 is expected here: ridge_initialization_ready is false.
cmmt-state-training preflight \
  --index <raw_index.json> --root-map <root-map.json> \
  --config configs/state_training/residual_mlp.json \
  --world-size 1 --output runs/<pair>/preflight-data.json

# 2. Fit the ridge maps once per pair. The same file serves all three configs.
cmmt-state-training ridge-fit \
  --index <raw_index.json> --root-map <root-map.json> \
  --config configs/state_training/residual_mlp.json \
  --preflight-report runs/<pair>/preflight-data.json --output runs/<pair>/ridge.pth

# 3. Check again with the ridge file. This report is the one that train needs.
cmmt-state-training preflight \
  --index <raw_index.json> --root-map <root-map.json> \
  --config configs/state_training/residual_mlp.json --ridge-init runs/<pair>/ridge.pth \
  --world-size 1 --output runs/<pair>/preflight-ridge.json

# 4. Train. Repeat with transformer.json and residual_mlp_matched.json and a new --output each time.
cmmt-state-training train \
  --index <raw_index.json> --root-map <root-map.json> \
  --config configs/state_training/residual_mlp.json --ridge-init runs/<pair>/ridge.pth \
  --preflight-report runs/<pair>/preflight-ridge.json \
  --device cuda --output runs/<pair>/residual-mlp

# 5. State error of a trained model and of the ridge baseline on the validation role (CPU).
cmmt-state-training evaluate \
  --index <raw_index.json> --root-map <root-map.json> \
  --model residual_mlp --model-artifact runs/<pair>/residual-mlp/paired_translators.pt \
  --output runs/<pair>/residual-mlp-state.json
cmmt-state-training evaluate \
  --index <raw_index.json> --root-map <root-map.json> \
  --model ridge --model-artifact runs/<pair>/ridge.pth \
  --output runs/<pair>/ridge-state.json
```

Notes that follow from the code:

- `--world-size` in `preflight` must equal the number of training processes, and `global_batch` must be divisible by it. For two GPUs use `--world-size 2` and start `train` through `python -m torch.distributed.run --standalone --nproc-per-node=2 -m vos_memory_inspector.state_training.cli train ...` (see `docs/state_training.md`).
- `train` refuses to start without `--ridge-init`, with a ridge file fitted on another index, or into an output folder that already holds a run. To continue a run, add `--resume runs/<pair>/<model>/last.ckpt`; the data, config, model, options, ridge file and world size must be unchanged.
- A training folder holds `last.ckpt`, `best.ckpt`, `paired_translators.pt` (the selected model) and `training_report.json` (the loss per epoch).
- The selected model is the epoch with the lowest validation state loss. In the official index the validation role is the LVOS official validation set, which shares scenes with MOSEv2 training videos. Do not read that loss as evidence of generalisation.
- To time or inspect a model without data: `python -m vos_memory_inspector.transformer_translator --preset transformer` (also `residual_mlp`, `residual_mlp_matched`).

## 3. What may differ between pairs, and what may not

The paper says: one translator structure for Small -> Base+, Small -> Large and Base+ -> Large, with weights trained separately for each pair by one procedure.

**Must be identical for every pair**

- The model code and the `model_options` of each config. `TranslatorConfig` (`num_layers`, `mlp_ratio`, `num_heads`, `sr_ratio`) may not change for one pair only.
- The loss terms. A term that exists for one pair and not for another makes it a different method.
- The inputs of the translator (memory records only), the evaluation protocol, the cases, the switch frames and the baselines.
- The procedure that chooses the values below.

**May differ, if one stated rule produces the values**

- The ridge solution and the weights (always per pair, never reused across pairs).
- Learning rate, number of epochs or updates, batch size, weight decay, chosen from one search space on data that is not the test set.
- Loss weights, when a rule sets them (for example, each term divided by the target variance of that pair, which the trainer already does with the train-split scales).
- Training length, when the stopping rule is the same (the trainer stops on the validation loss with `min_epochs`, `early_stop_checks` and `relative_improvement`).

Record every per-pair value in one table: one row per setting, one column per pair, and the number of configurations tried.

## 4. What to report

- **State loss** per epoch for each model (from `training_report.json`), and the validation state error of the selected model.
- **The ridge baseline** with the same metric, from `evaluate --model ridge`.
- **J&F by method** with the evaluator of the design review, on the same cases and switch frames: target model alone, source model alone, direct copy, the anchor baseline, ridge, `residual_mlp`, `transformer`, `residual_mlp_matched`. Report differences as paired differences with a video-level bootstrap interval.
- **Where the scores come from.** M3-VOS and PUMaVOS never trained or selected anything; treat them as the test. For MOSE dev and LVOS val, report the scene-clean version and say that they share scenes with the training data.
- **The number of seeds.** If it is one, say so next to every comparison.

Earlier records showed that a lower state loss does not mean better tracking. Do not select or rank models by state loss in the paper.

## 5. Known problems

1. **Only Small -> Base+ can be trained on this branch.** `src/vos_memory_inspector/state_training/data.py` line 143 accepts only the pair `('sam2.1-small', 'sam2.1-base-plus')`. The change that lets an index carry its own `model_direction` exists as uncommitted work in the main checkout (branch `feat/split-model-state-cache`). The model itself has nothing tied to a pair, and stored Small, Base+ and Large states have the same shape.
2. **The end-to-end fine-tuning code still uses the old classes.** `train_e2e.py` of the design review (`test14/code/review` on branch `test14-design-review` of `vos-memory-system`) builds the old models and starts from their weights. It must be ported to `ResidualMLPStateTranslator` and `TransformerStateTranslator` before it can run. In the earlier records this step mattered more than the choice of structure.
3. **The J&F evaluators load old payloads.** They need to read `cmmt.stitched_translator.v1` (`ResidualMLPStateTranslator.from_payload`, `TransformerStateTranslator.from_payload`, or `translator_artifact.load_learned_translator`).
4. **No old weight loads.** Everything saved before commit `d4ec72d` needs commit `913e35b` or `main`.
5. **The pointer head has nothing to learn from the state loss.** It is linear, and the ridge solution is already the minimiser of the squared error on the fit data. It only changes under a loss through the target model.
6. **Weight decay acts on the stitching layer and the pointer map** and so pulls them towards zero, away from the ridge solution. It is small (1e-4). If this matters, exclude the two layers in `state_training/policy.py`; that is a training choice, not a design change.
7. **Hyperparameters were not tuned for the new structure.** The configs keep the learning rate, weight decay and schedule of the old models.
8. **CUDA and multi-GPU paths were not run** after the redesign; only the CPU tests (including a two-process CPU test) were.
9. **With joint training the linear maps drift.** In a short diagnostic on real states (`docs/paper_check_report.md`, section 6), a stitching layer trained together with the block was worse on its own than the ridge map it started from, and the block's correction grew to 34 to 46% of the stitched memory. Keeping the stitching layer and the pointer map at the ridge solution cost about 1% in state error and kept the correction at 12 to 23%. Whether to freeze them is the designer's open decision; do not change it silently, and say which setting a run used.
10. **The output has less contrast than the target.** In the same diagnostic the output had 85 to 90% of the target's per-channel standard deviation. Report the energy ratio of the output next to the state loss.

## 6. What you must not change

- Anything in `src/vos_memory_inspector/models/` and the `model_options` of the three configs. If training suggests a structural change, report the evidence to the designer; do not change it for one pair.
- The ridge start: do not train from the identity placeholder, and do not reuse one pair's ridge file or weights for another pair.
- The independence of records: the translator maps one record at a time. Do not add inputs from other records, frames or the source model's features.
- The test set: do not choose a checkpoint, a hyperparameter or a loss weight on M3-VOS or PUMaVOS.
- Existing weights, caches and run folders. Write new runs into new folders.

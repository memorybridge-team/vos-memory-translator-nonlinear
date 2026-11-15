# Paper check report

## 요약 (한국어)

1. **모델은 바꾸지 않았습니다.** 새 결정(D19 이후)은 없습니다. 고친 것은 설명, 일치 강도 표시 2개, docstring입니다.
2. **디스크의 PDF 28편을 직접 읽었습니다**(이번에 새로 25편). 인용한 지점마다 쪽수를 달았습니다(`docs/translator_model_design.md` 7절). 지난 보고서의 미확인 10건 중 6건을 닫았고, 1건은 부분적으로 닫았으며, 3건은 닫지 못했습니다(1절, 8절).
3. **가장 가까운 선행 연구를 새로 찾았습니다.** Heo 외 2026 (언어 모델의 KV cache를 같은 모델군의 다른 크기 모델로 ridge map으로 옮김)이 "저장된 상태를 다른 모델로 번역"에 관해 지금까지 인용한 어떤 논문보다 가깝습니다. Related work에 넣어야 합니다.
4. **코드는 논문의 연산과 수치로 일치합니다.** Attention은 PVT 식 (1)~(4)와 최대 1.5e-14 차이, sine 인코딩은 DETR 식과 3.5e-7 차이(좌표만 반 칸 다름, 문서에 적음). 테스트 17개를 추가했습니다.
5. **설명이 빠져 있던 차이 4가지**를 찾아 문서에 적었습니다: Mikolov는 bias 없이 SGD로 풉니다. UNIC의 projector는 학습 뒤 버립니다. DETR와 좌표가 반 칸 다릅니다. Fixup의 규칙 2·3은 쓰지 않습니다.
6. **[MY DATA] 설계 주장과 어긋나는 결과 ①.** Stitching layer와 블록을 함께 학습하면, 블록을 뗀 stitching layer만의 오차가 ridge 시작점보다 나빠집니다(예: Small→Base+ 0.160 → 0.219). 즉 "선형 map + 보정"이라는 해석이 학습 뒤에는 그대로 성립하지 않습니다.
7. **[MY DATA] 해결 제안.** 선형 map을 ridge 값에 고정하고 블록만 학습하면 오차는 거의 같고(0.6~1.3% 차이), 보정 크기는 stitched 메모리의 12~23%로 줄어듭니다. 구조 변경이 아니라 학습 설정입니다. 결정은 직접 해 주세요.
8. **[MY DATA] 어긋나는 결과 ②.** 출력의 채널별 표준편차는 목표의 85~90%입니다. "목표 모델이 기대하는 scale을 유지한다"고는 쓸 수 없습니다.
9. **[MY DATA] Attention 대 같은 크기의 대조군.** Large가 목표인 두 쌍에서는 attention이 낫고, Small→Base+에서는 대조군이 낫습니다. 차이는 seed 간 편차보다 크지만, state 오차일 뿐 추적 정확도가 아닙니다.
10. **[MY DATA] 가장 어려운 쌍은 Small→Large**(ridge R² 0.685)이고, 문헌의 예측과 같습니다. Base+→Large(0.703)는 Small→Base+(0.790)보다 Small→Large에 가깝습니다.
11. 테스트: 시작 전 183 passed / 4 skipped, 지금 200 passed / 4 skipped (추가 17개).
12. GPU 0만 썼고(두 번, 각 약 6분), 서버에는 `/workspace/design-check-20261112/`만 새로 만들었습니다. 남아 있는 프로세스는 없습니다. J&F 평가와 전체 학습은 하지 않았습니다.
13. Push하지 않았습니다. 로컬 커밋이 추가되었습니다(7절).

---

## 0. Changes to the model

**None.** No decision was added to `docs/refactor_decisions.md`. Everything the reading and the numeric checks found could be made honest by changing the description. Two findings of the diagnostics (section 6) are reported as proposals for the training procedure, not as changes.

## 1. Open details from the last report

Pages are those of the PDF on disk (arXiv version unless noted).

| # | Detail | Closed? | Where | Result |
|---|---|---|---|---|
| 1a | Lenc & Vedaldi: filter size of the stitching layer | Yes | p.4, section 2.3 | The paper gives none. The layer is described as a bank of linear filters. So the 1x1 form cannot be cited to this paper |
| 1b | Lenc & Vedaldi: are the networks frozen? | Yes, as "not stated" | p.4, p.7 | Only the stitching map is described as learned, on the classification loss. The paper does not say whether the other layers are updated |
| 2a | Bansal et al.: kernel-size ablation | Yes | p.14, appendix B.1 | Kernel sizes 1, 3, 5, 7, 9 on two ResNet-18 of different seeds; kernel size has "minimal impact"; they keep 1 |
| 2b | Bansal et al.: exact form of the stitching layer | Yes | p.4, p.5, p.13 (A.4) | Convolutional: BatchNorm, 1x1 convolution, BatchNorm. ViT: one linear map over the embedding dimension (768x768), token-wise. Models A and B are frozen (p.1) |
| 3a | Pan et al.: are anchors updated? | Yes | p.8, section 4.2; p.6 | Yes. Tuning stitching layers alone "is only promising for some stitches", so the whole network is updated by default; hierarchical models use one tenth of the learning rate for anchors |
| 3b | Pan et al.: DeiT-Ti with DeiT-B | Yes | p.8, Figure 9(b) | Stitchable, but most such stitches do not beat Ti-S and S-B; they suspect the large gap leads to worse minima |
| 3c | Pan et al.: initialisation | Yes | p.5, Eq. 5; p.8 | Pseudo-inverse least squares; unstable alone; a good start for SGD; updating the layers is "essential" |
| 4 | Mikolov et al.: objective | Yes | p.4, Eq. 3 | Sum of squared norms of W x_i - z_i over word pairs, solved with SGD; no bias |
| 5 | Lähner & Moeller: full paper | Yes | p.3 (Eq. 1-2), p.4, p.6 | A linear or rigid map fitted by squared error on paired latent vectors of two autoencoders. Linear beats rigid. A three-layer MLP does worse than linear on test data because it overfits, more so with few pairs |
| 6 | Fixup: wording of the rules | Yes | p.5 | Rule 1: classification layer and last layer of each residual branch to 0. Rule 2: standard init elsewhere, branch weights scaled by L^(-1/(2m-2)). Rule 3: a scalar multiplier per branch and scalar biases |
| 7 | DETR: sine encoding and `normalize` | Partly | p.6, p.13, p.22 | The paper has no formula. It says fixed sine encodings are added to queries and keys at every attention layer. The formula and `normalize=True` in the default builder come from the official code, obtained through a fetch tool. SAM 2's copy of the same class, with `normalize=True` as default, was read on disk |
| 8 | Venues of MViTv2 and EdgeTAM | No, not from the PDF | first pages | Both files are arXiv versions and print no venue. Bibliographic records give CVPR 2022, pp. 4804-4814 and CVPR 2025, pp. 13832-13842 |
| 9 | Li et al. 2016 ("Convergent Learning") | No | — | Not on disk. Still known only through Csiszárik et al.'s description; not cited as a source |
| 10 | Citation format of the paper | No | — | The existing paper is an hwp file; the Korean draft keeps numbered references |

Also not on disk, so unchanged since the last report: LLaVA-1.5 (Liu et al. 2024), DINOv2 (Oquab et al. 2024), Hoerl & Kennard 1970.

## 2. Labels changed in this check

| Row | Was | Now | Why (page) |
|---|---|---|---|
| 2, ridge start | Direct precedent (Csiszárik et al.) | The same, plus **close analogy** for ridge on stored state within a model family | Heo et al. 2026, p.1 and p.3: a closed-form ridge map moves the stored cache of one language model to a different-sized model of its family |
| 3, block after the stitching layer | None for the combination | None for the combination, plus **close analogy** for adding a nonlinear map where the linear one falls short | Heo et al. 2026, p.7: an MLP recovers the pairs where ridge fails, but it replaces the ridge map. Pan et al. 2023, p.8: training the stitching layers alone works for some stitches only |
| 4, feed-forward as a cross-model map | Close analogy (UNIC), loose (LLaVA-1.5) | Close analogy (Heo et al.; UNIC), loose (LLaVA-1.5) | Heo et al., p.7. UNIC, p.3: the projectors are removed after distillation, which the description now says |

All other labels were checked against the text and kept: rows 1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17. Two rows gained a source without a change of label: row 9 (Vaswani et al., p.5: each head has 64 dimensions) and row 14 (Lähner & Moeller, p.6: linear beats an MLP for latent vectors). Row 7 gained a source that points the other way (Bansal et al., p.14).

## 3. Code against the papers

Verdicts: **matches**; **differs, stated** (the documents already said so); **differs, was not stated** (found in this check; the description is now corrected). Numeric checks are in `tests/test_operators_against_papers.py`.

| Operator | Paper | Code | Numeric check | Verdict |
|---|---|---|---|---|
| Spatial reduction of keys and values | PVT p.4-5, Eq. 3: reshape each R x R group to one token, linear projection, LayerNorm; each side divided by R. Official code: convolution with kernel = stride = R, then LayerNorm, on the key/value input only | `SpatialReductionAttention`: the same convolution and LayerNorm; queries at full resolution | With the position signal set to zero the output equals the paper's equations: max difference 1.5e-14 (64 channels, 64x64 grid, R=4, 1 head; also R=2, 2 heads) | **Matches** |
| Attention | PVT Eq. 1-2, 4: softmax(q k^T / sqrt(d_head)) v, output projection | The same; separate key and value layers where PVT's code has one joint layer (the same function) | Same test | **Matches** |
| Reduction ratio | PVT Table 1, p.6: 8 at 64 channels, 4 at 128. MViTv2 p.4: pooling stride 4 at the first stage | 4 at 64 channels, by convolution | — | **Differs, stated** |
| Heads and head size | PVT Table 1: 1 head at 64 channels. Vaswani p.5: head size 64 | 1 head of 64 | Asserted in a test | **Matches** |
| Feed-forward ratio and activation | Vaswani p.5: two linear layers, inner width 4x, ReLU. ViT p.4: GELU. PVT stage 1: ratio 8 | Two linear layers, ratio 4, GELU | Equals W2 GELU(W1 x + b1) + b2 with GELU(x) = x Phi(x): 1e-12 | **Matches** ViT; ratio differs from PVT, **stated** |
| Pre-LN placement | ViT p.4, Eq. 2-3; Xiong p.2 | `x + Attn(LN(x))`, then `x + FFN(LN(x))` | Equal to the written-out form: 1e-12 | **Matches** |
| Final LayerNorm | Xiong p.2 and ViT Eq. 4 have one | None | — | **Differs, stated** |
| Zero start | Fixup p.5, rule 1; Goyal p.6 | Last linear layer of each sublayer at zero | Block is an exact identity (`torch.equal`) | **Matches** rule 1. Rules 2 and 3 are not used: **was not stated** in full, now stated (rule 2's factor is 1 for one block) |
| Sine position encoding: formula | DETR official code; SAM 2 code (read on disk); Vaswani p.6 | The same frequencies, sine on even and cosine on odd channels, y half first | Equals the DETR formula at the same coordinates: 3.5e-7 | **Matches** |
| Sine position encoding: coordinates | DETR and SAM 2: position i at (i + 1) / H | (i + 0.5) / H | Difference to DETR's own values: up to 0.049 | **Differs.** "Cell centres" was stated; the exact half-cell difference **was not stated**, now stated. With centres the reduced key grid lies at the mean of the cells it covers (tested) |
| Where position enters | DETR p.22: added to queries and keys at every attention layer, not to values | Added to query and key inputs, not to values | Tested | **Matches** |
| Stitching layer | Bansal p.13: BatchNorm, 1x1 convolution, BatchNorm. Csiszárik p.4: linear 1x1 convolution | 1x1 (per-position linear) with bias, no normalisation | Equals a 1x1 convolution with the same weights: 1e-12 | **Matches** the convolution; no BatchNorm, **stated** |
| Start of the stitching layer | Csiszárik p.6 and Pan p.5: least squares (pseudo-inverse) | Ridge; identity placeholder before it | With penalty 0 the solver returns the least-squares solution (1e-9) and the model reproduces it | **Matches** at penalty 0; ridge and placeholder **stated** |
| Linear pointer map | Mikolov p.4: W x, no bias, SGD. Lähner p.3: linear map by squared error | Affine map with bias, closed-form ridge | Covered by the least-squares test | **Differs, was not stated** (no bias and SGD in Mikolov et al.); now stated |
| Separate pointer head | UNIC p.4: dedicated projectors for CLS and patch tokens. SAM 2: pointer from the mask decoder token | Separate head | — | **Matches** in kind. UNIC's projectors are removed after training: **was not stated**, now stated |
| One record at a time | SAM 2 section 4; EdgeTAM p.4 | Records never mix | Tested earlier (`test_records_are_translated_independently`) | **Matches** |

## 4. The structure as a whole

**Q1. Is a linear stitching layer alone enough between two models of different sizes?** Sometimes, and less so as the models differ more.

- Enough in the cases studied by Bansal et al. (p.9: a wider bottom network can be stitched into a thinner top one and even improves it) and Csiszárik et al. (p.6: same architecture, different seeds). Both are classification networks.
- Not always enough: Pan et al. (p.8) find that training the stitching layers alone works for some stitches only and that non-neighbouring sizes are worse. Heo et al. (p.1, p.6) find that a linear map keeps 73 to 98% of the target's accuracy on four of six pairs and fails on two, where an MLP helps. Lenc & Vedaldi (p.7) find deeper layers less interchangeable.
- What it implies: there is published support for needing more than a linear map on some pairs. The published remedies are to update the target network (Pan et al.) or to replace the linear map by an MLP (Heo et al.). Since the target is frozen here, the extra capacity can only sit in the translator. Putting it in a block after the stitching layer is consistent with this, but no paper tests that arrangement.

**Q2. "Linear map first, then a residual correction": does any source discuss it, or the risk that the block undoes the linear map?** No source on disk discusses a block after the linear map, or that risk. What they do say: the trained stitching layer moves away from its least-squares start and that this helps (Csiszárik et al., p.6; Pan et al., p.8). He et al. (p.2) give the general reason for learning a difference to a reference. Cache-to-Cache (p.5) builds its fuser on a residual principle, but adds to the receiver's own cache. The diagnostics in section 6 measure the risk directly: with joint training the stitching layer does not stay a good map on its own.

**Q3. Position encoding and scale of the output.**

- Position: SAM 2 adds position encodings inside memory attention, so the stored memory carries none and the target regenerates them. Heo et al. (p.1) do the corresponding thing for language models: they remove the position rotation from the keys before mapping, so the map is position-free. This supports not translating position.
- Scale: no stitching paper discusses keeping the scale of the output for a frozen reader. Two sources match statistics of the target features: Maiorca et al. (p.4) standardise every feature of both spaces before the closed-form fit and de-normalise afterwards, and UNIC (p.4-5) standardises teacher features before the loss. Heo et al. (p.2, p.8) and CacheBridge (p.2) add that the place of the error matters more than its size.

**Q4. Contradictions, better-grounded alternatives, and the closest precedent.**

- **Closest precedent for "translating a model's stored state into another model's space": Heo et al. 2026, "Cross-Model KV Cache Transfer in LLM Families" (arXiv 2608.03893, p.1).** It is closer than anything cited before: different-sized models of one family, a switch in the middle of a session, stored state mapped by a closed-form ridge fit, small-to-large as the main direction, evaluation by the target's downstream accuracy, and one design applied to several pairs with unequal results. It is a preprint on language models, and its nonlinear map replaces the linear one. It is now cited in rows 2, 3, 4 and 17 of the design table and in the Korean draft as [24].
- Against a design choice: Bansal et al. (p.14) weakly speak against expecting a gain from spatial context, for networks of the same architecture. This is now stated next to the attention sublayer.
- For design choices: Lähner & Moeller (p.6) support the linear pointer head over an MLP; Heo et al. (p.7) support a nonlinear part where the linear map fails; the ablation of Denoising Vision Transformers supports not using learnable position embeddings.
- Alternatives worth knowing, none of them a change of structure: CacheBridge (p.1-2) keeps the closed-form linear map and weights its fitting error by the effect on the target's attention; Mixture-of-Translators (p.1-2) adds a loss through the target model. Both belong to how the translator is fitted or trained.
- EdgeTAM (p.2) reports that naive spatial pooling of SAM 2 memory hurts accuracy. The translator pools only the keys and values inside its own attention; its output stays at full resolution.

## 5. Changes to the model

None.

## 6. Diagnostics on real states — MY DATA

Everything in this section is **MY DATA**. It is not evidence about tracking accuracy, and no paper is cited for it.

### What was run

| | Run 1 (joint) | Run 2 (linear maps kept at ridge) |
|---|---|---|
| Command | `CUDA_VISIBLE_DEVICES=0 nice -n 5 python design_check.py --out results --fit-cases 160 --held-cases 40 --updates 2000 --seeds 0 1 2 --commit 3643942` | `CUDA_VISIBLE_DEVICES=0 nice -n 5 python design_check_v2.py --out results_frozen_linear --freeze-linear --fit-cases 160 --held-cases 40 --updates 2000 --seeds 0 1 2 --commit 3643942` |
| Folder | `/workspace/design-check-20261112/results/` | `/workspace/design-check-20261112/results_frozen_linear/` |
| Time | 377 s, GPU 0, peak 844 MiB | 353 s, GPU 0, peak 811 MiB |

- Code: `/workspace/design-check-20261112/src` is a copy of commit `3643942` of this branch; the scripts `design_check.py` and `design_check_v2.py` are in the same folder. Each results folder holds `log.txt` and `results.json`.
- Data: `/workspace/states/{small,base_plus,large}/MOSEv2/train`, 20,841 cases common to the three sizes; one case per video, chosen with seed 7; 160 cases to fit (2,249 records) and 40 held out (543 records, other videos). Shapes were the same for all: spatial (64, 64, 64) bfloat16, pointer (256,) float32.
- Procedure: ridge fit on the 160 cases (penalty 1e-6 of the mean input variance for the spatial map, 1e-2 for the pointer). Then 2,000 updates, batch of 16 records, AdamW, learning rate 3e-4, weight decay 1e-4, seeds 0, 1, 2. Loss: squared error divided by the target's mean square on the fit set, spatial plus pointer (run 2: spatial only).
- A dry run with 8 + 4 cases and 20 updates is in `/workspace/design-check-20261112/smoke/`.
- "Error" below is the held-out spatial squared error divided by the target's mean square. "±" is half the range over the three seeds.

### Results that disagree with a design claim or a source

1. **"The block adds a correction to the linear estimate."** After joint training this reading no longer holds. If the block is removed from a trained model, the stitching layer alone is worse than the ridge map it started from (table D). The stitching layer and the block adapt to each other, and the correction is large: 34 to 46% of the stitched memory in size.
2. **"The output keeps the scale the target expects."** It does not fully. The output has 85 to 90% of the target's standard deviation per channel after training, and 82 to 88% for the ridge map alone (table E). The mean is right (gap below 2% of a standard deviation).
3. **"Attention adds spatial context that the MLP model lacks."** In state error, attention beats its size-matched control on the two pairs with Large as target and loses on Small -> Base+ (table C). Bansal et al.'s kernel ablation would have predicted little effect for similar models.

Nothing else disagreed: the identity is worse than predicting the mean for every pair, as Lenc & Vedaldi report for their networks; Small -> Large is the hardest pair, as predicted.

### A. Linear start, and what a short training adds

| Pair | Identity R² | Ridge R² | Ridge error | `residual_mlp` | `transformer` | `residual_mlp_matched` |
|---|---:|---:|---:|---:|---:|---:|
| Small -> Base+ | -2.44 | 0.790 | 0.1603 | 0.1484 ± 0.0001 | 0.1465 ± 0.0003 | 0.1452 ± 0.0001 |
| Base+ -> Large | -0.71 | 0.703 | 0.2237 | 0.1957 ± 0.0001 | 0.1832 ± 0.0001 | 0.1894 ± 0.0000 |
| Small -> Large | -1.04 | 0.685 | 0.2375 | 0.2154 ± 0.0001 | 0.2078 ± 0.0003 | 0.2099 ± 0.0001 |

R² after training: 0.806 / 0.808 / 0.810 (Small -> Base+), 0.740 / 0.757 / 0.749 (Base+ -> Large), 0.714 / 0.724 / 0.722 (Small -> Large).

- The linear map explains 69 to 79% of the held-out variance of the target memory. The block removes a further 7 to 18% of the remaining error in 2,000 updates: 7.4 / 8.6 / 9.4% for Small -> Base+, 12.5 / 18.1 / 15.3% for Base+ -> Large, 9.3 / 12.5 / 11.6% for Small -> Large.
- The worst single channel has a ridge R² of 0.68, 0.48 and 0.55 for the three pairs, and 0.71 to 0.72, 0.58 to 0.64 and 0.59 to 0.60 after training.
- Pointer: ridge R² is 0.525, 0.432 and 0.414. Training with the squared error does not improve it on held-out data (error 0.328 -> 0.336, 0.362 -> 0.369, 0.373 -> 0.382). This fits the expectation that a linear head has nothing to learn from this loss beyond its ridge start.

### B. One structure for three pairs

The same code and configuration ran for all three pairs. By every measure Small -> Large is the hardest pair and Small -> Base+ the easiest, which agrees with the literature's prediction. Base+ -> Large is much closer to Small -> Large than to Small -> Base+: the fit is lower whenever Large is the target. The block's relative gain is largest for Base+ -> Large.

### C. Attention against its size-matched control

| Pair | `transformer` | `residual_mlp_matched` | Difference | Largest seed half-range |
|---|---:|---:|---:|---:|
| Small -> Base+ | 0.1465 | 0.1452 | +0.0013 (control better) | 0.0003 |
| Base+ -> Large | 0.1832 | 0.1894 | -0.0061 (attention better) | 0.0001 |
| Small -> Large | 0.2078 | 0.2099 | -0.0021 (attention better) | 0.0003 |

The differences are 4 to 60 times the seed half-range, so they are not seed noise in this setting. They are still small (0.9%, 3.3% and 1.0% of the error), come from one choice of 200 cases and one training length, and are state errors. The earlier records of this project showed that state error does not predict J&F. The honest summary: attention and the wider MLP are close; attention has a visible edge only when Large is the target.

### D. Does the block undo the linear map?

Run 1, joint training. "Stitch only" removes the block from the trained model.

| Pair | Model | Ridge error | Trained, full | Trained, stitch only | Attention / stitched | Feed-forward / stitched | Correction / stitched |
|---|---|---:|---:|---:|---:|---:|---:|
| Small -> Base+ | `residual_mlp` | 0.1603 | 0.1484 | 0.2189 | — | 0.337 | 0.337 |
| | `transformer` | | 0.1465 | 0.2211 | 0.130 | 0.361 | 0.352 |
| | matched | | 0.1452 | 0.2525 | — | 0.439 | 0.439 |
| Base+ -> Large | `residual_mlp` | 0.2237 | 0.1957 | 0.2734 | — | 0.361 | 0.361 |
| | `transformer` | | 0.1832 | 0.2755 | 0.295 | 0.393 | 0.397 |
| | matched | | 0.1894 | 0.2987 | — | 0.452 | 0.452 |
| Small -> Large | `residual_mlp` | 0.2375 | 0.2154 | 0.2892 | — | 0.355 | 0.355 |
| | `transformer` | | 0.2078 | 0.2919 | 0.262 | 0.389 | 0.383 |
| | matched | | 0.2099 | 0.3202 | — | 0.464 | 0.464 |

Ratios are root-mean-square sizes relative to the stitched memory. The stitching weights moved by only 0.5 to 1.5% in Frobenius norm, yet the stitched memory alone became 22 to 58% worse than at the ridge start.

Run 2, stitching layer and pointer map kept at the ridge solution:

| Pair | Model | Joint | Linear maps kept | Correction / stitched | Cosine of the correction with the ridge residual |
|---|---|---:|---:|---:|---:|
| Small -> Base+ | `residual_mlp` | 0.1484 | 0.1493 | 0.123 | 0.26 |
| | `transformer` | 0.1465 | 0.1475 | 0.139 | 0.28 |
| | matched | 0.1452 | 0.1466 | 0.140 | 0.29 |
| Base+ -> Large | `residual_mlp` | 0.1957 | 0.1973 | 0.188 | 0.34 |
| | `transformer` | 0.1832 | 0.1852 | 0.226 | 0.41 |
| | matched | 0.1894 | 0.1918 | 0.209 | 0.38 |
| Small -> Large | `residual_mlp` | 0.2154 | 0.2168 | 0.168 | 0.29 |
| | `transformer` | 0.2078 | 0.2096 | 0.197 | 0.34 |
| | matched | 0.2099 | 0.2121 | 0.190 | 0.33 |

- Keeping the linear maps at the ridge solution costs 0.6 to 1.3% in state error and makes the description "ridge map plus a correction" literally true after training. The correction is then 12 to 23% of the stitched memory.
- In the Transformer model the two sublayers partly cancel. With the linear maps kept, the attention output is 19 to 32% and the feed-forward output 22 to 33% of the stitched memory, but their sum is 14 to 23%. The feed-forward sublayer reads the stitched memory plus the attention output, so it can subtract part of it. This is an observation, not a defect that I can name from a source.

### E. Scale and statistics of the output

| Pair | | Energy ratio | Std ratio per channel, median | Std ratio, lowest channel | Mean gap / target std |
|---|---|---:|---:|---:|---:|
| Small -> Base+ | ridge | 0.834 | 0.882 | 0.819 | 0.013 |
| | after training (three models) | 0.857 to 0.860 | 0.898 to 0.901 | 0.850 to 0.855 | 0.017 to 0.018 |
| Base+ -> Large | ridge | 0.777 | 0.837 | 0.704 | 0.015 |
| | after training | 0.811 to 0.825 | 0.865 to 0.876 | 0.760 to 0.797 | 0.018 to 0.020 |
| Small -> Large | ridge | 0.763 | 0.820 | 0.752 | 0.014 |
| | after training | 0.790 to 0.799 | 0.848 to 0.857 | 0.777 to 0.786 | 0.017 to 0.020 |

Energy ratio is the mean square of the output over that of the target. A least-squares fit always loses variance in proportion to what it cannot explain, and the block recovers only a little of it. The earlier records of this project found the same loss of contrast and that it costs tracking accuracy.

### What these numbers do and do not show

- They show: the same structure runs and trains for the three pairs on real states; the linear map carries most of the explainable variance; the block adds a real but modest gain in state error; the gain is larger when Large is the target; joint training lets the linear part drift; the output is lower in contrast than the target.
- They do not show: anything about J&F or tracking; which model is better for the paper; what happens with a full training or with a loss through the target model; behaviour on other datasets. The fit set is small (160 cases) and the pointer statistics rest on 2,249 pointers.

### Proposals (not applied; yours to decide)

1. **Keep the stitching layer and the pointer map at the ridge solution during state-loss training**, or give them a much smaller learning rate. Evidence: section D. It makes the paper's description hold after training at a cost of about 1% in state error. Source for not training the closed-form map: Heo et al. apply their ridge map without further training; against it, Csiszárik et al. and Pan et al. report that training the stitching layer on the *task* loss helps. This is a training setting; it would go into the handover guide, not into the model.
2. **Match the per-channel scale of the output to the target** after the map, with one gain per channel from fit statistics. Evidence: section E and the earlier "contrast gain" result of this project. Nearest source: Maiorca et al. standardise both spaces and de-normalise into the target's statistics (p.4); no source does exactly this. It would add one fixed operation at the output, so it would need a decision (D19) if adopted.
3. **Weight the ridge fit by the target's attention sensitivity**, as CacheBridge does for language models (p.1-2). Not tested here.

## 7. Tests, files, commits

Tests on CPU (torch 2.10, Python 3.12): 183 passed, 4 skipped before; **200 passed, 4 skipped** after. The 17 new tests are in `tests/test_operators_against_papers.py`. No test was removed or changed. The 4 skips need CUDA or a newer torch API.

Files changed:

| File | What |
|---|---|
| `tests/test_operators_against_papers.py` | New: 17 numeric checks of the operators against the papers |
| `docs/translator_model_design.md` | Section 2 rewritten with pages and the new sources; section 6 updated; section 7 citations with a page for every point |
| `docs/paper_design_section_ko.md` | Reference [24] added; markers carry pages; sections 3.2 to 3.5 corrected |
| `src/vos_memory_inspector/models/_stitched.py` | Docstrings and one comment; no behaviour change |
| `docs/refactor_decisions.md` | Source statements in D1 and D8 |
| `docs/refactor_progress.md` | Log entry |
| `docs/paper_check_report.md` | This file |

Local commits (nothing is pushed; `origin/feat/translator-model-refactor` is still at `913e35b`):

| Hash | Message |
|---|---|
| `3643942` | test. check the borrowed operators numerically against the cited papers |
| (next commit) | docs. record the paper check: pages, corrected sources and diagnostics — the commit that contains this file |

## 8. What I could not read or check

- **"Read" means the cited passages, not whole papers.** I extracted the text of each PDF and read the sections that carry the cited points, with their surroundings. I did not read every page of every paper, and I did not read figures. Table 1 of PVT and the algorithm box of Pan et al. were read from the extracted text, which keeps their numbers but not their layout.
- **Official code was not on disk.** The PVT and DETR code came through a fetch tool, as in the last check. SAM 2's code, which contains the same sine encoding class, was read on disk.
- **Venues that the PDFs do not print** (most files are arXiv versions): Lenc & Vedaldi, Bansal et al., Pan et al., He et al., PVT, MViTv2, DETR, Denoising ViT, Lin et al., UNIC, EdgeTAM, SAM 2. They rest on bibliographic records. DINOv2's venue rests on a library catalogue only.
- **Not on disk:** Hoerl & Kennard 1970, LLaVA-1.5, DINOv2, Li et al. 2016, SegFormer, SimCLR.
- **Heo et al., CacheBridge and Mixture-of-Translators are 2026 preprints.** I read them as they are on disk; they have no peer-reviewed venue.
- **The diagnostics** used one dataset (MOSEv2 train), one choice of cases, one training length and no J&F. The CUDA paths of the official trainer were not run; the diagnostics use their own short loop.
- **The server connection dropped once** during run 2. The run finished on the server and its files are complete (27 runs and the closing line in `log.txt`); nothing is left running.
- A wait command of mine looped on the server for about five minutes because it matched its own process; I stopped it by process ID. It wrote nothing.

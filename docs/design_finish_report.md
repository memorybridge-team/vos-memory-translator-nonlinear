# Design finish report

## 요약 (한국어)

1. **모델은 바꾸지 않았습니다.** 출처를 다시 읽어 보니 고칠 것은 모두 설명과 일치 강도 표시였습니다. 새 결정(D19 이후)은 없습니다.
2. **인용 28건을 모두 다시 확인했습니다.** 논문 자체가 확인되지 않은 것은 없습니다. 세부 사항 10건은 아직 확인하지 못해 따로 적었습니다.
3. **일치 강도 표시 9개를 고쳤습니다.** 4개는 낮췄고(1, 3, 7의 동기, 17), 3개는 새 출처로 올렸으며(8의 비율, 15, 16), 2개는 근거 논문을 바꿨습니다(2, 4).
4. 가장 중요한 수정: Lenc & Vedaldi 2015는 stitching layer의 **필터 크기를 밝히지 않습니다.** "1×1"의 근거는 Bansal 외 2021과 Csiszárik 외 2021입니다.
5. 두 번째: Pan 외 2023은 이어 붙인 모델도 함께 학습합니다. 두 모델을 고정한 채 최소제곱 해에서 시작하는 선례는 Csiszárik 외 2021입니다.
6. 새로 찾은 출처: UNIC(ECCV 2024, FFN 형태와 전역 token용 별도 head), EdgeTAM(CVPR 2025, 저장된 메모리를 기록별로 attention 처리), MViTv2(축소 비율 4).
7. **출처가 없는 부분**과 논문에 쓸 문장을 설계 문서 6절에 넣었습니다. 축소 비율 4, 블록 1개, 마지막 LayerNorm 생략, "stitching layer 뒤에 학습 블록"이라는 조합이 해당합니다.
8. 논문용 한국어 초안(`docs/paper_design_section_ko.md`)과 학습 담당자용 안내(`docs/training_handover.md`)를 썼습니다. 초안에는 성능 주장이 없습니다.
9. 테스트: 시작 전 182 passed / 4 skipped, 지금 183 passed / 4 skipped (추가한 테스트 1개).
10. **직접 읽은 논문은 로컬 PDF 3편뿐입니다**(Csiszárik, SAM 2, Tang). 나머지 인용문은 가져오기 도구가 돌려준 원문이며, 긴 페이지는 앞부분만 읽힙니다. 인용 전에 5절의 미확인 항목을 봐 주세요.
11. 논문의 인용 형식은 확인하지 못했습니다(hwp를 열 수 없음). 초안은 번호 형식 `[n]`입니다.
12. 학습, GPU 작업, 평가, push는 하지 않았습니다. 로컬 커밋 4개가 더 생겼습니다.

---

## 1. Model change

None. No decision was added to `docs/refactor_decisions.md`. Task 2 found descriptions and labels that were stronger than their sources; all of them could be made honest by rewriting the description. The only code edits are docstrings and one comment in `src/vos_memory_inspector/models/_stitched.py`, one new config file, and one new test.

## 2. Match labels changed

Row numbers refer to the table in `docs/translator_model_design.md`, section 2.

| # | Part | Was | Now | Why |
|---|---|---|---|---|
| 1 | Stitching layer | Direct precedent, with Lenc & Vedaldi cited for the 1x1 form | Direct precedent from Bansal et al. and Csiszárik et al.; Lenc & Vedaldi as the origin of the idea; close analogy for the setting | Lenc & Vedaldi call it "a bank of linear filters" and give no filter size. All three papers stitch classification networks mid-network, not a stored memory |
| 2 | Ridge start | Direct precedent, Pan et al. first | Direct precedent, Csiszárik et al. first | Pan et al. also train the stitched networks (anchor parameters at one tenth of the learning rate for hierarchical models). Csiszárik et al. keep the rest fixed, as here |
| 3 | Block after the stitching layer | Close analogy for the placement | None for the combination | No source places a trained block after a stitching layer |
| 4 | Feed-forward sublayer as a cross-model map | Close analogy (LLaVA-1.5 as "connector between two frozen models") | Close analogy (UNIC), loose analogy (LLaVA-1.5) | In LLaVA the language model is frozen only in the first training stage. UNIC's projector is the same form: two linear layers, GELU, hidden width 4d |
| 7 | Attention: motivation | Close analogy (Lin et al.) | Loose analogy (Lin et al.); close analogy for attention within one stored memory map (EdgeTAM, new) | Lin et al. compute their map from teacher and student features for a distillation loss with a trained student |
| 8 | Reduction ratio 4 | None | Loose analogy, not tuned | MViTv2 sets the key/value pooling stride to 4 at its first stage. PVT uses 8 at its 64-channel stage and 4 at its 128-channel stage |
| 15 | Separate pointer head | None | Follows from the model; close analogy (UNIC), loose analogy (DINOv2) | UNIC uses "dedicated ... projector heads for each type of tokens"; the SAM 2 sentences were read |
| 16 | One record at a time | None | Follows from the model; close analogy (EdgeTAM) | EdgeTAM encodes "each memory feature M_t" separately; the SAM 2 sentence was read |
| 17 | One structure for all pairs | Direct precedent | Close analogy | Pan et al. use one kind of stitching layer for every stitch but do not state it as a principle, and stitch neighbouring sizes only by default |

Unchanged after the check: rows 5 (Pre-LN), 6 (zero start), 9 (one head), 10 (position encoding), 11 (one block), 12 (no final LayerNorm: none), 13, 14 (linear pointer).

The three questions the prompt gave as examples:

- **Does Pan et al. 2023 initialise stitching layers by least squares, and for which models?** Yes: the pseudo-inverse solution on 100 training images, then SGD; anchors are DeiT, Swin Transformer and ResNet families. The networks are trained too, so it is not a frozen-network precedent.
- **Does Lenc & Vedaldi 2015 stitch with a 1x1 layer at every position?** The paper does not say. It states "a bank of linear filters" with no filter size. Csiszárik et al. describe it as the origin of "1x1 convolution stitching layers", which is their reading.
- **Does PVT use reduction ratio 4 at its first stage?** No. Stage 1 (64 channels, 1 head) uses 8. Ratio 4 is stage 2 (128 channels, 2 heads).

## 3. Parts with no source, and the wording

Full table with sentences: `docs/translator_model_design.md`, section 6. In short:

| Item | Result of the extra search | Sentence (short form) |
|---|---|---|
| Reduction ratio 4 | No rule in any source; MViTv2 and PVT stage 2 use 4 | "We did not tune this factor." |
| One block | Denoising ViT uses one block for a related task | "We use a single block, the smallest configuration. We did not tune the depth." |
| No final LayerNorm | No source; the convention it departs from is sourced | "We omit it, because the output must keep the scale that the target model expects." |
| Stitching layer + trained block | No source for the combination | "The combination is our design; each part follows the cited work." |
| Separate pointer head | Follows from SAM 2; UNIC as analogy | "The two come from different modules and have different sizes." |
| One record at a time | Follows from SAM 2; EdgeTAM as analogy | "SAM 2 builds each memory record from one frame." |
| Memory translation between VOS models | No source found | "To our knowledge ... has not been studied." |

Searches made for task 3 (by a helper agent, reviewed): "adapter output not normalized no final layer norm pre-LN transformer"; "preserve scale of target features"; "model switching video object segmentation memory"; "cross-model memory transfer SAM 2"; "dynamic submodel video object segmentation"; "memory translation between segmentation models"; "swap encoder without re-encoding past frames"; plus targeted look-ups of PVT, SegFormer, MViT, MViTv2, Denoising ViT, DINOv2, UNIC, SAM 2, XMem, Cutie and EdgeTAM. Not found or not opened: Twins, CvT, EfficientViT, Focal, FeatUp, JAFAR, Houlsby adapters, AdaptFormer, the MAE decoder.

Candidates the helper named that I did not verify and therefore did not cite: MViT (ICCV 2021, first-stage stride), XMem (ECCV 2022) and Cutie (CVPR 2024) for per-frame memory, Park et al. (CVPR 2021, reuse gate), CLIP-Adapter and Tip-Adapter, and three 2026 preprints on KV-cache translation.

## 4. Citations verified

All 28 entries of `docs/translator_model_design.md`, section 7, exist with the stated title, authors, venue and year, except for the venue details listed in section 5. How each was checked is marked there as **read** (local PDF), **quoted** (verbatim passage through a fetch tool, **2x** when two independent fetches agree) or **record** (bibliographic record only).

- Read by me in the local PDFs: Csiszárik et al. 2021; Ravi et al. (SAM 2); Tang et al. 2026.
- Quoted twice: Lenc & Vedaldi 2015; Bansal et al. 2021; Pan et al. 2023; Vaswani et al. 2017; Dosovitskiy et al. 2021; Xiong et al. 2020; Zhang et al. 2019; Goyal et al. 2017; Wang et al. 2021 (PVT); Carion et al. 2020; Yang et al. 2024.
- Quoted once: Mikolov et al. 2013; Lähner & Moeller 2024; He et al. 2016; Xie et al. 2021; Li et al. 2022 (MViTv2); Lin et al. 2022; Sariyildiz et al. 2024 (UNIC); Liu et al. 2024; Chen et al. 2020; Oquab et al. 2024; Zhou et al. 2025 (EdgeTAM).
- Record only: Hoerl & Kennard 1970; Hendrycks & Gimpel 2016; Ba et al. 2016.

Open items of the earlier reports, now closed:

| Item | Result |
|---|---|
| Bansal et al.: form of the stitching layer (was from memory) | Confirmed: 1x1 convolution with BatchNorm before and after; token-wise linear map for Transformers |
| Kornblith et al. and Morcos et al.: width and similarity (were from memory) | Confirmed by quote from both papers |
| Pan et al.: non-neighbouring stitches (appendix not read) | The main text says stitching without the nearest-stitching constraint "achieves inferior performance"; the sentence on DeiT-Ti with DeiT-B came from a summary only |
| Klabunde et al.: journal (was UNVERIFIED) | ACM Computing Surveys 57(9), Article 242, 2025, DOI 10.1145/3728458, from search records; the ACM page was not opened |
| DETR and ResNet venues (were from memory) | ECCV 2020, pp. 213-229 and CVPR 2016, pp. 770-778, from bibliographic records |
| Nguyen et al.: venue (was from memory) | ICLR 2021, from the conference page |
| SAM 2: sine position encoding in memory attention (was from the config only) | Confirmed in the paper: "In addition to sinusoidal absolute positional embeddings, we use 2d spatial Rotary Positional Embedding" |

Two author lists in my earlier brief to a helper were wrong and were never written into a document: the projector-ensemble paper and the fifth author of Csiszárik et al. (it is Dániel Varga).

## 5. Still unverified

No paper is unverified. These details are:

1. Lenc & Vedaldi: whether the two networks are frozen during stitching (not stated in the text read), and the filter size (not stated at all).
2. Bansal et al.: the appendix ablation on kernel sizes (a summary says sizes 1 to 9 make little difference; not read verbatim, so not cited).
3. Pan et al.: the exact sentences "fully updated during training" and "stitching DeiT-Ti with DeiT-B" (summary only). What is verbatim: anchor parameters are trained at one tenth of the learning rate for hierarchical models.
4. Mikolov et al.: the squared-error form of the objective (the equation was garbled in the HTML).
5. Lähner & Moeller: only the abstract page was read.
6. Fixup: the wording of rule 2 (not used by this design).
7. DETR: that the default builder passes `normalize=True` (the class default is `False`).
8. Venues taken from memory or from a file name, not from a primary page: MViTv2 (CVPR 2022), DINOv2 (TMLR 2024), the original LLaVA (NeurIPS 2023), EdgeTAM (CVPR 2025) and Tang et al. (CVPR 2026) from the CVF file names.
9. Li et al. 2016 ("Convergent Learning"): only Csiszárik et al.'s description of it was read; it is listed as "not used as a source".
10. The citation format of the paper: the existing paper file is hwp and could not be opened, so the Korean draft uses numbered references.

A limit of the method, which applies to every **quoted** entry: the fetch tool returns verbatim text for roughly the first 40,000 characters of a page and a summary for the rest. Quotes from late sections and appendices are the ones most likely to be summaries, and I have marked those I know of. I read in full only the passages of the three local PDFs.

## 6. Consistency check

- Parameter and multiply-accumulate numbers were counted again from the code for `linear`, `residual_mlp`, `transformer` and `residual_mlp_matched` and compared with every number in `README.md`, the design note, the decisions file, the morning report, `docs/state_training.md`, the handover guide and the Korean draft. No mismatch. (The only large number in the documents that is not a size is 4,096, the token count.)
- Decision numbers D1 to D18 are the same in the decisions file and the morning report.
- The part table of the morning report was brought in line with the corrected labels; the literature report got a corrections block at its top and is otherwise unchanged.
- Found and fixed: the `evaluate` example in `docs/state_training.md` pointed at the first-run weights file, which this branch cannot load.
- Not read in full by me: `docs/architecture/*`, `docs/design/*`, `docs/index.html` (a helper reported earlier that they do not describe the model structure).

## 7. Tests

CPU, torch 2.10, Python 3.12, command in `docs/refactor_progress.md`.

| When | Result |
|---|---|
| Before this task (`3f8e01d`) | 182 passed, 4 skipped |
| After | 183 passed, 4 skipped |

Added: `test_training_configs_build_the_documented_models` (the three trainer configs build the documented sizes and share their training settings). No test was removed or changed. The 4 skips need CUDA or a newer torch API, as before.

Not run, by instruction: any training, any GPU job, any evaluation. The GPU server was not used at all in this task.

## 8. Files written or changed

| File | What |
|---|---|
| `docs/translator_model_design.md` | Section 2 rewritten (what each source does, where the design differs, corrected labels, table of changed labels); new section 6 "What has no source, and how to say it"; section 7 citations with the way each was checked |
| `docs/paper_design_section_ko.md` | New. Korean draft, about one page, with a source marker after each sentence that rests on a source, and a numbered reference list |
| `docs/training_handover.md` | New. Commands for one pair, what may and may not differ, what to report, known problems, what not to change |
| `docs/design_finish_report.md` | New. This file |
| `configs/state_training/residual_mlp_matched.json` | New. Lets the official trainer run the parameter-matched control |
| `src/vos_memory_inspector/models/_stitched.py` | Docstrings and one comment corrected to match the sources; no behaviour change |
| `tests/test_stitched_models.py` | One test added |
| `docs/refactor_decisions.md` | Wording of the source statements in D1 and D5 |
| `docs/refactor_morning_report.md` | Part table labels; one test row |
| `docs/translator_design_literature_grounding.md` | Corrections block at the top |
| `docs/state_training.md` | Config table row for the control; `evaluate` example |
| `docs/refactor_progress.md` | Log entry |

## 9. Local commits

Nothing is pushed. `origin/feat/translator-model-refactor` is still at `913e35b`.

| Hash | Message |
|---|---|
| `61c2e55` | docs. correct source statements after reading the cited papers |
| `d8b4b94` | docs. add the training handover guide and a config for the matched control |
| `dabc066` | docs. add the Korean draft of the design section for the paper |
| (next commit) | docs. add the design finish report — the commit that contains this file |

The five commits of the refactor (`7d80e59` to `3f8e01d`) are unchanged below these.

# Literature grounding for the translator design

Answer to `docs/fable_reference_prompt.md`. Written 2026-11-12.

How to read this file:

- Sections 1 to 5, 7 and 8 use **LITERATURE** only, plus the SAM 2.1 configuration files and the translator code (the design itself). They were written before I opened any experiment record.
- Section 6 is **MY DATA**. It was added afterwards and is kept apart.
- Each citation has a tag in section 8: *confident* (I checked the title, authors and venue on the web during this session, or I know the paper well) or *UNVERIFIED* (check before citing).
- Match strength uses your four levels: direct precedent, close analogy, loose analogy, none.

Two facts about the setting that I checked in the code, because the answer depends on them:

- `sam2/configs/sam2.1/sam2.1_hiera_{s,b+,l}.yaml` all set the memory encoder `out_dim: 64`, memory attention `kv_in_dim: 64`, `image_size: 1024`, `num_maskmem: 7`, and the same fuser (two layers, 7x7 kernel). So the three sizes do produce memory records of the same shape, [64, 64, 64] plus a 256-long pointer. The backbones differ (Hiera embed 96 / 112 / 144, stages [1,2,11,2] / default / [2,6,36,4]).
- In `models/_spatial.py`, candidate A (preset `base`) is `M + alpha * (W_l M + Up(W_c ctx))` with `alpha = 0` and normally initialised branches. Candidate B (`residual_mlp`) is `M + W_2 GELU(W_1 M)` with `W_2 = 0`. So A starts at the identity because of the gate, and B because of the zero last layer. Neither has both at zero at the same time, which is correct: a zero gate on a zero branch gives zero gradient to both and never trains.

---

## 1. Recommendation (LITERATURE)

**B, on one condition about how the shared form is written.**

The literature cannot separate A and B on expected accuracy. I found no work that measures whether translating one vision model's feature map into another's needs spatial context. B is cheaper to defend, because it is the smallest step away from the only form with direct precedent: the per-position stitching layer (a 1x1 convolution) of Lenc & Vedaldi, Bansal et al., Csiszárik et al. and Stitchable Neural Networks.

- **What you claim by choosing B:** the difference between two SAM 2.1 sizes' memory spaces is a change of channel coordinates that is the same at every position, plus a mild nonlinearity.
- **Main risk:** that claim fails wherever the target record at a position depends on image context that the source record does not hold at that position. B cannot repair that. A reviewer will ask for the spatial-context ablation, and that ablation is A. So A is not wasted: it becomes the row that tests B's assumption.
- **The condition:** both candidates are written as `M + correction`. The literature gives no reason to expect that two independently trained networks share channel coordinates, so "identity plus a small correction" has no support as a description of the problem (section 4.2). The grounded form is "linear map plus correction", with the linear map from ridge regression. Your code already has this as `skip_mode="affine"`. This is a remark on the shared form, not a third candidate.

What you give up with B: any claim about spatial misalignment or position-dependent differences between the two models. What you would give up with A: the per-position nonlinearity, and simplicity. A has six extra choices that each need an origin (patch size 4, two blocks, learnable positional embedding, 3x3 convolution, bilinear upsampling, the gate), and only the gate and the Pre-LN block have clean ones.

One point for the paper: A and B are not nested. A has a *linear* local branch plus context. B has a *nonlinear* local branch and no context. A direct A-versus-B comparison mixes two factors. The 2x2 in your registry (`linear_local`, `base`, `no_context`, `base_mlp`) is the comparison a reviewer will accept.

Both candidates contain the 1x1 linear stitching layer as a special case. For A it is the local branch. For B it follows from the identity `GELU(x) - GELU(-x) = x`: with hidden width 128 = 2 x 64, `W_1 = [I; -I]` and `W_2 = [D, -D]` give exactly `M + D M` for any 64x64 matrix `D`. So you can honestly write that each candidate extends the linear stitching layer in one direction, A by spatial context and B by a per-position nonlinearity.

---

## 2. Summary (LITERATURE)

**Well grounded**

- 3. Zero-initialised residual branches and gates. Direct precedents for both variants.
- 4. The *linear* per-position map. Direct precedent in model stitching.
- 7. Ridge regression as baseline and as initialisation. Direct precedent (Stitchable Neural Networks initialises stitching layers by least squares, then trains with SGD).
- 9. One structure for several pairs of one family. Direct precedent (Stitchable Neural Networks; common adapter practice).
- 10. Per-pair hyperparameters. Common practice, if selected by one stated procedure.
- 1. The problem itself, as a problem. Close analogies in KV-cache translation and in "backfilling-free" embedding upgrades.

**Weak, analogy only**

- 2. A residual around the *identity* between two independently trained models. The technique is standard. Its premise is not supported here.
- 4. The *nonlinear* part of the per-position map. Stitching work avoids it on purpose.
- 5. Spatial context. One close analogy in feature distillation; nothing in stitching.
- 6. A separate pointer head. An engineering consequence of SAM 2's design.
- 9. Which pair is hardest. Indirect evidence only, and it points in two directions for Base+ -> Large.

**No precedent**

- 1. Translating a VOS memory bank between two models. I searched and found none.
- 5. Keeping attention within one record. It follows from how SAM 2 builds a record, not from a paper.
- 8. Same shape on both sides. A fact about SAM 2.1, not a design choice with a literature.
- The specific numbers in A (patch 4, two blocks, 4 heads, FFN ratio 2) and in B (hidden 128).

---

## 3. Table (LITERATURE)

| # | Decision | Best reference(s) | Match | Effect on your study |
|---|---|---|---|---|
| 1 | Translate internal state, do not recompute | Cache-to-Cache and the KV-cache work you know; Ramanujan et al. 2022 (forward-compatible training); Shen et al. 2020; Su et al. 2022; Lester et al. 2022 | Close analogy; none in VOS | You may claim a new setting. You may not claim the idea of state translation is new. |
| 2 | Predict a correction (residual) | He et al. 2016; Houlsby et al. 2019; Rebuffi et al. 2017; Gao et al. 2024; Wang et al. 2020 (RBT) | Technique: direct. Premise (identity is a good start): none | Call it an initialisation device, not a statement that the spaces are close. Prefer "linear map + correction". |
| 3 | Zero-init branch / gate | Goyal et al. 2017; Zhang et al. 2019 (Fixup); Bachlechner et al. 2021 (ReZero); Touvron et al. 2021 (LayerScale); Hu et al. 2022 (LoRA); Zhang et al. 2023 (ControlNet) | Direct precedent | Safe. State which variant each candidate uses. |
| 4 | Per-position channel map | Lenc & Vedaldi 2015; Bansal et al. 2021; Csiszárik et al. 2021; Pan et al. 2023; Romero et al. 2015 | Linear: direct. Nonlinear: loose | The linear part needs no defence. The GELU layer needs an ablation against the linear map. |
| 5 | Patchified Transformer context, learnable PE | Lin et al. 2022 (target-aware transformer); Dosovitskiy et al. 2021; Yang et al. 2024 (Denoising ViT); Luo et al. 2016 | Loose to close analogy; "within one record": none | A claims spatial mismatch exists. That is an empirical claim you must show. |
| 6 | Separate pointer head; linear vs MLP | Mikolov et al. 2013; Smith et al. 2017; Lähner & Moeller 2024; Wang et al. 2020; Hewitt & Liang 2019 | Separate head: none (engineering). Linear map for a vector: close analogy | Linear is the default. An MLP must beat ridge on held-out data to be justified. |
| 7 | Ridge as baseline and as init | Pan et al. 2023; Csiszárik et al. 2021; Lähner & Moeller 2024; Hoerl & Kennard 1970; Chen et al. 2016 (Net2Net) | Direct precedent | Report ridge as its own row. Expect "low regression error" and "good tracking" to disagree. |
| 8 | Same shape, no resizing | SAM 2.1 configs; Li et al. 2016; Ainsworth et al. 2023 | None (fact of the model family) | Guarantees the interface and the spatial registration. Guarantees nothing about channel meaning. |
| 9 | One structure, three pairs | Pan et al. 2023; Houlsby et al. 2019; Hu et al. 2022; Kornblith et al. 2019; Nguyen et al. 2021; Mirzadeh et al. 2020 | One structure: direct. Hardest pair: loose | Small -> Large is predicted hardest. This argues for neither A nor B. |
| 10 | Per-pair training differences | Dodge et al. 2019; Bouthillier et al. 2021; Musgrave et al. 2020; Romero et al. 2015 | Common practice | Differences are fine if one written rule produces them. |

---

## 4. Details per decision (LITERATURE)

### 4.1 Translating internal state instead of recomputing it

**(a) Closest work.**

- The KV-cache and latent translation papers you listed (Cache-to-Cache, CacheBridge, cross-model KV cache transfer, Mixture-of-Translators, Maiorca et al.). These are the closest in kind: state that one model computed, mapped by a learned module so that another model's attention can read it.
- Ramanujan, Vasu, Farhadi, Tuzel, Pouransari, "Forward Compatible Training for Large-Scale Embedding Retrieval Systems", CVPR 2022. Shen, Xiong, Xia, Soatto, "Towards Backward-Compatible Representation Learning", CVPR 2020. Wang, Chang, Yang, Chen, Lai, "Unified Representation Learning for Cross Model Compatibility", BMVC 2020.
- Su et al., "On Transferability of Prompt Tuning for Natural Language Processing", NAACL 2022. Lester, Yurtsever, Shakeri, Constant, "Reducing Retraining by Recycling Parameter-Efficient Prompts", arXiv 2022.
- Oozeer et al., "Activation Space Interventions Can Be Transferred Between Large Language Models", ICML 2025. Jha, Zhang, Shmatikov, Morris, "Harnessing the Universal Geometry of Embeddings", arXiv 2025.

**(b) What they did.** The embedding-compatibility line is the closest vision analogy. A retrieval system stores features computed by an old model. When the model is upgraded, recomputing every stored feature ("backfilling") is expensive. Shen et al. avoid it by constraining how the new model is trained. Ramanujan et al. keep the new model untouched and learn a transformation that maps the stored old features into the new space. Wang et al. learn a small residual module that maps features between two models. Your case has the same shape: stored state from the old model, a new model you may not change, and a learned map applied once to the stored state. Su et al. and Lester et al. do the same for soft prompts, which are also state tied to one frozen model. Oozeer et al. map steering vectors between LLMs with learned mappings between activation spaces.

**(c) Effect on your study.** You can say the problem class is established: avoiding recomputation by translating stored state. You cannot say the idea of translating state is yours. What is specific to you is the object (a VOS memory bank read by cross-attention over many frames) and the consequence (an error in one translated record stays in the bank and affects every later frame). A reviewer may object that retrieval features are compared once by a distance, while your records are consumed by a multi-layer attention module, so the analogy does not transfer accuracy expectations. Accept that: cite these for the problem framing, not for the expected quality.

**(d) Match.** Close analogy for the framing. **No direct precedent in VOS**: a web search for memory transfer between SAM 2 sizes returned only memory selection (SAM2Long) and memory compression (EdgeTAM) work. Search terms to re-check before submission: "SAM 2 cross-model memory", "model switching video object segmentation", "memory bank translation", "anytime video segmentation model cascade".

### 4.2 A small adapter that predicts a correction

**(a) Closest work.** He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016. Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019. Rebuffi, Bilen, Vedaldi, "Learning multiple visual domains with residual adapters", NeurIPS 2017. Gao et al., "CLIP-Adapter: Better Vision-Language Models with Feature Adapters", IJCV 2024. Chen et al., "AdaptFormer", NeurIPS 2022. Wang et al. 2020 (the Residual Bottleneck Transformation, a residual module that maps features between two embedding models). For predicting only the difference to the input: Kim, Lee, Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 2016; Zhang et al., "Beyond a Gaussian Denoiser", IEEE TIP 2017.

**(b) Relevance.** All of these add a small learned branch to an identity path. He et al. state the reason: if the wanted function is close to the identity, the residual is easier to fit than the function. Kim et al. and Zhang et al. use it because the output really is close to the input (a low-resolution image and its sharp version).

**(c) Effect on your study.** This is the weakest point of the shared form, and it is a matter of premise, not technique. Adapters modify a model that stays the same, so the identity is the right start. Your translator maps between two networks trained separately. The literature on such pairs says their units are not aligned one-to-one: Li et al. (ICLR 2016) found only partial unit-level correspondence between networks that differ only by seed; Ainsworth et al. (ICLR 2023) and Entezari et al. (ICLR 2022) show that hidden units of separately trained networks must be permuted before the networks even line up; and every stitching paper learns a full linear layer, never a correction to the identity. In SAM 2 the memory encoder ends in a linear 1x1 projection and the memory attention begins with linear key and value projections, so nothing in the architecture fixes the channel basis either. The literature therefore predicts that `M` is not a useful first guess for the target record, and that the "correction" must carry the entire map.

What this means in practice:

- You may keep the residual form. It is still a valid way to get a defined starting point.
- You should not write "the translator predicts a small correction to the source memory". Write that the residual form gives a function-preserving initialisation.
- The grounded alternative is a residual around a *linear* map: `M_hat = W M + b + correction(M)`, with `W, b` from ridge regression. He et al. already use linear projection shortcuts when the two sides of a block differ. This turns "correction" into a true statement (a correction to the best linear map), and it connects decisions 2, 3 and 7.
- A reviewer objection to expect: "the identity skip has no meaning between two different models". The affine skip removes it.

**(d) Match.** Direct precedent for the technique. None for the premise in the cross-model case.

### 4.3 Zero-initialised residual branches and gates

**(a) Closest work.**

- Zero last layer: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv 2017 (zero the last BN scale of each residual block). Zhang, Dauphin, Ma, "Fixup Initialization", ICLR 2019 (zero the last layer of each residual branch). Hu et al., "LoRA", ICLR 2022 (one factor starts at zero). Zhang, Rao, Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023 ("zero convolutions").
- Learnable gate at zero: Bachlechner et al., "ReZero is All You Need", UAI 2021 (`x + alpha F(x)`, scalar `alpha = 0`). De & Smith, "Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks", NeurIPS 2020 (SkipInit). Touvron et al., "Going deeper with Image Transformers", ICCV 2021 (LayerScale: a learnable *per-channel* scale on the residual branch, initialised near zero). Alayrac et al., "Flamingo", NeurIPS 2022 (tanh gate at zero). Zhang et al., "LLaMA-Adapter", ICLR 2024 (zero-initialised attention gate). Houlsby et al. 2019 (near-identity adapter initialisation).

**(b) Relevance.** Both variants exist in the literature and both are used for the reason you give: the new module must not disturb a working system at the start of training.

**(c) Effect on your study.**

- B and the pointer head use the "zero last layer" variant. Cite Goyal et al., Fixup, or ControlNet.
- A uses the "gate" variant. Per-channel `alpha` is LayerScale with the initial value set to exactly zero as in ReZero. Cite both. LayerScale uses a small positive value, so "following LayerScale" alone is not exact.
- In the paper, do not write "the last layer of every residual branch is zero-initialised" for A. In A the branches are normally initialised and only the gate is zero. The effect is the same, the mechanism is different, and a careful reviewer will read the code.
- With an affine skip from ridge, the statement changes from "the untrained translator is the identity" to "the untrained translator equals the closed-form linear baseline". This is the same principle and a stronger one. The closest named idea is function-preserving initialisation (Chen, Goodfellow, Shlens, "Net2Net", ICLR 2016).
- What it cannot claim: zero initialisation says nothing about the final accuracy. It is an optimisation choice.

**(d) Match.** Direct precedent.

### 4.4 A per-position channel-wise map

**(a) Closest work.** Lenc & Vedaldi, "Understanding image representations by measuring their equivariance and equivalence", CVPR 2015. Bansal, Nakkiran, Barak, "Revisiting Model Stitching to Compare Neural Representations", NeurIPS 2021. Csiszárik et al. 2021 (you know it). Pan, Cai, Zhuang, "Stitchable Neural Networks", CVPR 2023. Yang et al., "Deep Model Reassembly", NeurIPS 2022. Feature distillation connectors: Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015; Heo et al., "A Comprehensive Overhaul of Feature Distillation", ICCV 2019. Similarity measures: Raghu et al., "SVCCA", NeurIPS 2017; Morcos, Raghu, Bengio, "Insights on representational similarity in neural networks with canonical correlation", NeurIPS 2018; Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019. Limits: Ding, Denain, Steinhardt, "Grounding Representation Similarity Through Statistical Testing", NeurIPS 2021; Davari et al., "Reliability of CKA as a Similarity Measure in Deep Learning", ICLR 2023; Hernandez et al., "Model Stitching: Looking For Functional Similarity Between Representations", arXiv 2023; Smith, Mannering, Marcu, "Functional Alignment Can Mislead: Examining Model Stitching", ICML 2025. Also Merullo et al., "Linearly Mapping from Image to Text Space", ICLR 2023; Lähner & Moeller, "On the Direct Alignment of Latent Spaces", UniReps (PMLR 243) 2024; Huh et al., "The Platonic Representation Hypothesis", ICML 2024.

**(b) What they did.**

- Stitching connects the first part of one frozen network to the last part of another through a small trainable layer. For convolutional feature maps that layer is a 1x1 convolution, and for Transformers it is one linear map shared by all tokens. Both are exactly "the same channel map at every position". Bansal et al. keep the layer low-capacity on purpose, so that success says something about the representations and not about the layer. They show that well-trained networks of the same architecture stitch with little loss, and that networks with more width or data can be stitched into weaker ones.
- Stitchable Neural Networks stitch DeiT-Ti, -S and -B, a family of sizes like yours, with 1x1 convolutions.
- FitNets map a small student's feature map into a large teacher's feature space with a convolutional regressor. That is your Small -> Large direction for per-position maps.
- SVCCA, PWCCA and linear CKA all measure similarity *up to a linear or orthogonal transform*. Their use implies the working assumption that two networks' representations differ mainly by a linear change of basis. Lähner & Moeller test this directly and report that a full linear map aligns latent spaces better than more constrained maps.

**(c) Effect on your study.**

- Why a per-position map is a reasonable model: both models place position (i, j) of the memory grid on the same 16x16 image patch (same input size, same stride), so the spatial registration is given. What is not given is the channel basis. A per-position map models exactly that missing part.
- Is it enough? Stitching says a per-position *linear* map is often enough to keep task accuracy between networks of the same family. It does not say it is always enough. Stitching penalties grow when the networks differ more, and Hernandez et al. and Smith et al. warn that a stitch can work for reasons that have nothing to do with the two representations holding the same information.
- CKA and SVCCA cannot tell you whether a per-position map is enough. They pool over positions and examples and are invariant to things your target model is not. Ding et al. and Davari et al. show that these scores can stay high while function changes, and can be moved without changing function. Csiszárik et al. show the related point that a map with low regression error is not the map with the best task accuracy. So: do not cite a similarity score as evidence that translation will work.
- The nonlinear layer in B has little support in this literature. Stitching avoids nonlinear layers on purpose. The honest position is "linear is the grounded part, and the GELU layer is an added hypothesis tested by an ablation against the linear map". Hewitt & Liang ("Designing and Interpreting Probes with Control Tasks", EMNLP 2019) give the general argument: a more expressive map can succeed by its own capacity, so its success tells you less.
- Reviewer objection: "why not just a 1x1 convolution?" You need the linear row in the table (decision 7).

**(d) Match.** Direct precedent for the linear per-position map. Loose analogy for the nonlinear one.

### 4.5 Spatial context through a patchified Transformer (A only)

**(a) Closest work.** Lin et al., "Knowledge Distillation via the Target-aware Transformer", CVPR 2022. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021 (patch embedding, learnable positional embedding). Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020 (Pre-LN). Luo et al., "Understanding the Effective Receptive Field in Deep Convolutional Neural Networks", NeurIPS 2016. Yang et al., "Denoising Vision Transformers", ECCV 2024. Darcet et al., "Vision Transformers Need Registers", ICLR 2024. Bolya et al., "Window Attention is Bugged: How not to Interpolate Position Embeddings", ICLR 2024. Ryali et al., "Hiera", ICML 2023.

**(b) What they did, and the two arguments they give you.**

- *Spatial mismatch.* Lin et al. argue that when teacher and student have different architectures, the same spatial location does not carry the same semantic content, so matching features position by position is wrong. They let every teacher position be rebuilt from all student positions with a transformer, and they process large feature maps in patch groups to control cost. This is the one paper I know that argues for A's core idea, and it also uses a patch-level reduction. Your three backbones differ a lot in depth (Hiera-S has 16 blocks, Hiera-L has 48), so their effective receptive fields at the stride-16 level differ (Luo et al.), and the argument applies in principle.
- *Position-dependent differences.* Yang et al. show that ViT feature maps contain artefacts that depend on position, caused by the positional embeddings, and that these are a property of each model. Hiera uses learned absolute positional embeddings, and Bolya et al. show that their interaction with window attention leaves position-dependent effects. If each SAM 2.1 size has its own position-dependent component, a translator with a positional embedding can remove one and add the other, and a position-shared map cannot. This is the only literature argument I can find for the *learnable positional embedding* in A.

**(c) Effect on your study.**

- Both arguments say spatial context *may* matter. Neither shows that it does for SAM 2 memory, and neither concerns stitching: in distillation the student is trained, here both models are frozen. So A's central claim is empirical and you must show it with an ablation (context on versus off with the same local head).
- The memory encoder weakens the first argument a little. Every record passes through the same kind of fuser (two 7x7 depthwise blocks) in each model, and it mixes in the mask, which is spatially aligned. This is an architecture observation, not a citation.
- The learnable positional embedding has a cost that a reviewer may raise: the translator is tied to the 64x64 grid and is no longer translation-equivariant. SAM 2.1 always runs at 1024x1024, so this is acceptable, but say so.
- Patch size 4, two blocks, four heads, FFN ratio 2, the 3x3 convolution and bilinear upsampling have no specific origin. Honest wording: "we use a standard Pre-LN Transformer encoder (Dosovitskiy et al.; Xiong et al.) on a 4x4-patchified grid to keep the cost of attention low; these sizes were not tuned" (or "were chosen by ...", if they were).
- *Attention within one record.* No precedent. The reason is structural: SAM 2's memory encoder builds each spatial record from one frame's image feature and that frame's mask, so the target record is a function of one frame only. A per-record translator therefore has, in principle, the right input domain. Mixing records would also make the translator depend on how many records the bank holds and on their order. Write it as a design constraint that follows from the model, not as a finding.

**(d) Match.** Close analogy for "spatial mismatch needs non-local matching" (Lin et al.), loose analogy for position-dependent components (Yang et al.), direct precedent for the block design (ViT, Pre-LN), none for attention within a single record.

### 4.6 A separate head for the object pointer; linear or MLP

**(a) Closest work.** For mapping one vector space into another: Mikolov, Le, Sutskever, "Exploiting Similarities among Languages for Machine Translation", arXiv 2013; Xing et al., "Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation", NAACL 2015; Smith et al., "Offline bilingual word vectors, orthogonal transformations and the inverted softmax", ICLR 2017; Conneau et al., "Word Translation Without Parallel Data", ICLR 2018; Lähner & Moeller 2024; Maiorca et al. 2023; Merullo et al. 2023. For a small nonlinear module on a single embedding: Wang et al. 2020; Ramanujan et al. 2022. For choosing the simpler map: Hewitt & Liang 2019; Hoerl & Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", Technometrics 1970.

**(b) Relevance.** The pointer is one global vector per record, like a sentence or image embedding. The standard way to map such a vector between two independently trained spaces is a linear map, often constrained to be orthogonal, fitted in closed form. Wang et al. use a small residual nonlinear module for the same job and report that making it a deeper MLP did not help.

**(c) Effect on your study.**

- *Separate head.* This is an engineering consequence of SAM 2, not a literature choice. The pointer comes from the mask decoder's output token through an MLP, and the spatial memory comes from the memory encoder. They have different sizes (256 and 64) and are made by different modules in both models, so there is no reason for one map to serve both. Honest wording: "the pointer and the spatial memory are produced by different modules of SAM 2 and have different dimensions, so we translate them with separate heads". A loose analogy, if you want one: Oquab et al., "DINOv2", TMLR 2024, use separate heads for the image-level and patch-level objectives.
- *When linear is the better-grounded choice.* (i) When you have few samples per parameter. One record gives 4,096 spatial positions and one pointer, so the pointer head sees about four thousand times fewer training vectors than the spatial head, while the pointer MLP is the largest part of the model (263k of 280k parameters in B). (ii) When the MLP does not beat ridge on data that neither was fitted on. By the Hewitt & Liang argument, the burden of proof is on the more expressive map. (iii) When you want a result that does not depend on the optimiser, the seed or the stopping rule: ridge has a closed form.
- *When the MLP is defensible.* Each model produces its pointer with its own nonlinear MLP after its own decoder, so the true relation between the two pointers has no reason to be linear. That justifies trying an MLP. It does not justify keeping it if the ablation shows no gain.
- Reviewer objection for B: "your spatial path has 17k parameters and your pointer head has 263k; which part does the work?" You need the pointer ablation in any case.

**(d) Match.** None for the separate head (engineering). Close analogy for a linear map between vector spaces. Loose analogy for the pointer MLP.

### 4.7 A closed-form linear map as baseline and as initialisation

**(a) Closest work.** Pan et al. 2023; Csiszárik et al. 2021; Lähner & Moeller 2024; Maiorca et al. 2023; Mikolov et al. 2013; Hoerl & Kennard 1970; Alain & Bengio, "Understanding intermediate layers using linear classifier probes", ICLR workshop 2017; Chen et al. 2016 (Net2Net); Romero et al. 2015.

**(b) What they did.** Stitchable Neural Networks is a direct precedent for both halves of this decision. The stitching layers are solved by least squares (pseudo-inverse) on a small sample. The authors report that the least-squares solution alone gives unstable accuracy, but is a good initialisation for training with SGD, and they make it the default. Csiszárik et al. compare a stitching layer fitted by regression to the target activations with one trained on the task loss, and find that they are different solutions. FitNets train in two stages: first regress the student feature onto the teacher feature, then train on the task-level loss.

**(c) Effect on your study.**

- Ridge as a baseline row is standard and expected. Without it a reviewer cannot tell what the nonlinear parts add.
- Ridge as initialisation of the linear path is supported by Pan et al. This is also the cleanest source for the sentence "the untrained translator equals the linear baseline".
- The literature predicts that the closed-form solution will have good regression error and worse downstream accuracy than a map trained through the target model (Pan et al.: unstable alone; Csiszárik et al.: regression matching and task-loss matching differ). So if your ridge map scores well on state error and worse on tracking, that is the expected pattern, not an anomaly.
- Use ridge, not plain least squares, when channels are correlated or samples are few (the pointer). Say how the ridge strength was chosen.
- What it cannot claim: a good ridge fit does not show that the two memory spaces "are linearly related" in a functional sense.

**(d) Match.** Direct precedent.

### 4.8 The same shape on both sides

**(a) Closest work.** There is no literature for this, because it is a property of the SAM 2.1 family (checked in the three config files). For what equal shape does *not* imply: Li et al., "Convergent Learning: Do different neural networks learn the same representations?", ICLR 2016; Ainsworth, Hayase, Srinivasa, "Git Re-Basin", ICLR 2023; Entezari et al., "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks", ICLR 2022; Kornblith et al. 2019.

**(b) Relevance.** Two networks with identical architecture and different seeds already have differently ordered and differently mixed units. Equal shape is the normal case in that literature, and alignment still has to be learned.

**(c) Effect on your study.**

- What equal shape guarantees: the translated record fits the target's interface without any change to the target, and position (i, j) covers the same image patch in both models, because input size and stride are equal.
- What it does not guarantee: that channel k means the same thing, that the two spaces have the same scale, or that either space can be mapped onto the other without loss. A larger backbone may write information into its 64 channels that the smaller model never computed.
- Consequence for wording: do not write "the memory spaces are compatible because they have the same shape". Write "the memory records have the same shape, so no resizing is needed; the channel coordinates are not shared, which is why a translator is needed".
- Choosing no resizing is then just the absence of an unnecessary operation. No citation is needed.

**(d) Match.** None. It is a fact of the model family.

### 4.9 One structure for three pairs

**(a) Closest work.** Pan et al. 2023; Bansal et al. 2021; Houlsby et al. 2019 and Hu et al. 2022 (one adapter design, trained per model and per task); Kornblith et al. 2019; Morcos et al. 2018; Huh et al. 2024; Nguyen, Raghu, Kornblith, "Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth", ICLR 2021; Mirzadeh et al., "Improved Knowledge Distillation via Teacher Assistant", AAAI 2020; Cho & Hariharan, "On the Efficacy of Knowledge Distillation", ICCV 2019.

**(b) What they say.**

- *One structure for all pairs.* Stitchable Neural Networks use the same kind of stitching layer for Ti-S and S-B. Adapter papers apply one adapter design to models of several sizes with separate weights. "One design, trained per pair" is the normal way to present such a module.
- *Direction.* Pan et al. compare stitching from the small model into the large one ("Fast-to-Slow") with the reverse and report that small-into-large is more stable and better. All three of your directions are small-into-large.
- *Adjacency.* Pan et al. restrict stitching to neighbouring sizes and report that stitching without this restriction is worse. In your family the neighbours are Small-Base+ and Base+-Large, and Small-Large skips one.
- *Capacity gap.* Mirzadeh et al. and Cho & Hariharan show that a small student imitates a much larger teacher worse than a moderately larger one. Your translator plus the Small model's records must imitate what Large would have stored, so the same gap applies by analogy.
- *Do larger models resemble each other more?* Two answers exist. Kornblith et al. and Morcos et al. find that wider networks learn representations that are more similar to each other, and Huh et al. argue that more capable models converge. That predicts Base+ -> Large is the easiest pair. Nguyen et al. find that large-capacity networks develop a "block structure" whose representations are unique to each trained model, while representations outside it are similar across models. That predicts a pair of two large models is *not* automatically easy.
- *Information.* A translator cannot add information that the source record lacks. Small -> anything is bounded by what Small stored. Base+ -> Large starts from a richer source. This is a general argument, not a citation.

**(c) Effect on your study.**

- The prediction, as far as the literature goes: Small -> Large is the hardest pair (not adjacent, largest capacity gap, weakest source). The order of Small -> Base+ and Base+ -> Large is not decided: similarity work favours Base+ -> Large, the block-structure result adds doubt, and all of this evidence comes from classification networks, mostly CNNs.
- Does it argue for A or B? For neither. The reasons that make Small -> Large hard are a lack of information in the source and a larger distance between the spaces. More translator capacity addresses the second only. Lin et al.'s argument would predict that spatial mismatch grows with the difference in backbone depth, which is largest for Small -> Large, so *if* A helps anywhere it should help most there. That is a hypothesis to test, not a result to cite.
- The literature does not study "pair of two large models" versus "pair starting from the smallest" as such. Say that plainly.
- One structure for all pairs is therefore well supported, and a structure change for one pair would need strong evidence. I see no literature reason to expect a very large gain from a pair-specific structure.

**(d) Match.** Direct precedent for one structure across pairs of a family. Loose analogy for which pair is hardest.

### 4.10 Training differences between pairs

**(a) Closest work.** Dodge et al., "Show Your Work: Improved Reporting of Experimental Results", EMNLP 2019. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021. Musgrave, Belongie, Lim, "A Metric Learning Reality Check", ECCV 2020. Lipton & Steinhardt, "Troubling Trends in Machine Learning Scholarship", ACM Queue 2019. For staged losses: Romero et al. 2015; Johnson, Alahi, Fei-Fei, "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV 2016; Mathieu, Couprie, LeCun, "Deep multi-scale video prediction beyond mean square error", ICLR 2016; Bansal et al. 2021 (stitching layers are trained on the task loss through the frozen networks).

**(b) Relevance.** These papers set the reporting standard: state the search space, the budget, the selection data, and the variance. Adapter and LoRA papers routinely use a different learning rate and number of steps for each model, and nobody reads that as a different method.

**(c) Easy to justify.**

- Learning rate, number of updates, batch size, weight decay, chosen per pair on validation data from one stated search space.
- Longer training for a harder pair, if the stopping rule is the same for all pairs (for example, stop when validation has not improved for N evaluations).
- Initialisation data: each pair has its own ridge solution, fitted by the same procedure.
- Loss weights, if a single rule sets them. For example: "each loss term is divided by its value at initialisation", or "the feature term is divided by the variance of the target state". The numbers then differ per pair and the rule does not.
- Two-stage training (state regression, then a loss through the frozen target model) has precedent in FitNets and in stitching. An MSE-only stage is known to give averaged, low-contrast outputs (Mathieu et al.), which is a literature reason for the second stage.

**(d) What would make "one design" look dishonest.**

- Any change to width, depth, patch size, kernel size, or a branch switched on or off for one pair. That is a different structure.
- A loss *term* that exists for one pair and not for another, as opposed to a different weight. A reviewer will call it a different method.
- A different input for one pair (for example an extra feature from the source model). That is a different method.
- A post-hoc correction applied to one pair only (for example a per-channel rescaling added after seeing test results).
- Choosing hyperparameters or checkpoints on data that share scenes with the test set, or with a different selection set per pair.
- Reporting only the best pair's settings, or hiding that a pair needed far more search than the others.

**Match.** Common practice; direct precedent for per-model hyperparameters and for staged losses.

---

## 5. One design, three pairs (LITERATURE)

How to make the claim hold in the paper:

1. **Define the structure once, with every size as a number.** One figure and one table: layers, widths, parameter counts, initialisation. State: "The structure is identical for all three pairs. No layer, width or branch differs."
2. **Define the training procedure once, as a procedure with free values.** For example: "For each pair we (i) fit the ridge maps on the fit split, (ii) train with loss L for at most U updates, (iii) choose the learning rate from {…} and the checkpoint by validation score V." The procedure is shared. Its outcomes differ.
3. **Report the outcomes in one table with one column per pair.** Rows: learning rate, updates, loss weights, ridge strength, number of configurations tried, selection set, seed count. Leave nothing to the text.
4. **Say which values came from a rule and which from a search.** "Loss weights follow rule R" reads very differently from "loss weights were tuned per pair".
5. **If one pair needed something the others did not, apply it to all pairs and report all three.** If it is neutral on the others, the design is still one design. If it hurts them, you have a real finding about that pair and should report it as a limitation, not hide it in a per-pair setting.
6. **Use the same evaluation protocol for all pairs:** same cases, same switch frames, same baselines (direct copy, ridge, target model alone, source model alone).
7. **Never reuse weights across pairs, and say so**, because a reader may assume a Small -> Base+ model was fine-tuned into the Small -> Large one.

A sentence that stays true under all of the above: "We use one translator structure for all three pairs. The weights are trained separately for each pair with the same procedure; Table X lists the values that the procedure selected for each pair."

---

## 6. MY DATA vs LITERATURE

Everything in this section is **MY DATA**. None of it is a reference, and none of it should be cited as prior work. It was added after sections 1 to 5 were written and did not change them.

### 6.1 What I read

| Record | Files |
|---|---|
| First training run | `1차 학습 종료/DELIVERY_READY.json`, `1차 학습 종료/delivery/FINAL_REPORT_KO.md` |
| Design review (test14) | branch `test14-design-review` of `memorybridge-team/vos-memory-system`: `test14/docs/02_catalog.md`, `03_results.md`, `07_limits_and_open_items.md`; `test14/results/tables/training_models.csv`, `pointer_experiments.csv`, `jf_levels.csv`, `jf_paired.csv`, `external_levels.csv`, `external_paired.csv` |
| Small -> Large | `linear-transfer/runs/small-to-large-check-20261108/linear_r2.json`, `linear-transfer/runs/small-to-large-overview-20261109.md` |

Not read: the GPU server (the local copies above hold the summaries, so I started nothing there), the weights, and the case-level archives. No GPU job was run. Nothing was changed in any record.

Naming, from `02_catalog.md` and `training_models.csv`: `transformer` is candidate A (194,304 spatial parameters). `mlp` is candidate B **with an affine skip** (20,736 = 16,576 + 4,160), not the identity form in your prompt. `legacy` is the first-run A. All review models use the ridge pointer according to the catalog.

### 6.2 A versus B on Small -> Base+ (the only pair with trained A and B)

| Question | Literature said | Your records say | Verdict |
|---|---|---|---|
| Does A beat B, or B beat A? | Cannot be separated; B is cheaper to defend | After fine-tuning (`e2e_mlp` vs `e2e_tf`): M3-VOS 62.76 vs 61.52, PUMaVOS 70.22 vs 70.77. Against `e2e_ctx`, every interval includes zero (M3-VOS +0.75 [-0.53, +2.46] and -0.48 [-1.20, +0.01]; PUMaVOS -0.24 [-0.72, +0.21] and +0.31 [-0.82, +1.83]). Scene-clean MOSE dev and LVOS val differ by 0.13 or less | **Agrees.** Not separable |
| Is a large A worth more than a linear map under a regression loss? | Extra capacity on a regression loss need not help the task (Csiszárik et al.; Pan et al.) | First-run A (`legacy`) and `ridge` are equal: MOSE dev 60.79 vs 60.63, M3-VOS 59.10 vs 59.08, PUMaVOS ridge - legacy = -0.72 [-1.39, -0.13] | **Agrees** |
| MSE-only models | Nothing specific | `mlp` is ahead of `transformer` on scene-clean MOSE dev (57.54 vs 55.50) and LVOS val (72.70 vs 69.34) | **Says little.** No paired interval between the two in the tables, one seed, sets used for selection, no external evaluation, and the two differ in skip type as well as in structure |
| Does spatial context help? | It may (Lin et al.); must be shown | State error: yes. A 3x3 linear map beats a 1x1 one in R² for every pair (+.030 S->B+, +.053 S->L, +.062 B+->L), and context models have the lowest dev MSE (`ds_context` .1275, `mlp_context` .1338, `mlp` .1492). Tracking: no. `mlp_context` and `mlp` are equal on scene-clean sets (57.79 vs 57.54; 72.17 vs 72.70), and `ds_context` has the lowest MSE and a middle J&F | **Agrees for state error, no support for tracking** |

Check before publishing any A-versus-B number: in `training_models.csv`, `e2e_tf` has `"ridge_pointer": false` in its arguments, while the catalog says every review model uses the ridge pointer. If `e2e_tf` used a different pointer, the fine-tuned comparison has a second difference besides structure.

### 6.3 Per pair

| Pair | Literature said | Your records say | Verdict |
|---|---|---|---|
| Small -> Base+ | Adjacent pair; expected to be feasible | Works, close to Base+ after fine-tuning (scene-clean MOSE dev about -1.6, M3-VOS above Base+, PUMaVOS about -3.5 to -4) | **Agrees** |
| Small -> Large | Hardest: not adjacent, largest gap, weakest source | Linear R² is lowest (1x1 .635, 3x3 .689, against .751 and .781 for S->B+). A closed-form linear map reaches 65.92 on PUMaVOS against Large 74.98 and Small 72.79, and 63.59 on M3-VOS against 66.47 and 63.22. Memory-space maps stop near R² .75 on the prompt-frame record, and a map from Small's backbone feature goes higher, which points to missing information in the source record and not to translator capacity | **Agrees.** But **silent on A versus B**: neither was trained for this pair |
| Base+ -> Large | Undecided between "larger models are more alike" and "large models are idiosyncratic" | Linear R² only (1x1 .656, 3x3 .718). It sits next to Small -> Large, not next to Small -> Base+. Large -> Base+ gives .738 and .777. So in this one measurement the *target* being Large is what lowers the fit, whatever the source | **Leans to the second reading, weakly.** One linear fit on 800 cases; no tracking score, no trained A or B |

Other points where the records meet the literature:

- **Identity as a starting point (decision 2).** Predicted to be useless between separately trained models. Your records: identity R² is negative for every pair (spatial -2.35, -1.00, -0.69; pointer -4.6 to -3.2), and direct copy scores J&F 0.64, 0.03 and 0.01. **Agrees strongly.** `07_limits_and_open_items.md` already notes that the README and registry default say identity skip while the trained configuration was affine. Your prompt describes the identity form too. Decide which one the paper describes, and make the code default match.
- **Regression error versus tracking (decision 7).** Predicted to disagree. `03_results.md`: "MSE 순위와 J&F 순위는 일치하지 않았다"; the Small -> Large overview: state R² does not predict J&F. **Agrees.**
- **Pointer: linear or MLP (decision 6).** Predicted that linear is the default and the MLP must earn its place. `pointer_experiments.csv`: ridge dev MSE .3190, MLP .3146; validation .3541 and .3505; the deep MLP overfits (fit .163, dev .425). With the ridge pointer the tracking IoU equals the one with the true Base+ pointer (.9735 vs .9734). **Agrees.** The 263k-parameter pointer MLP in your shared form is not supported by your own records; the ridge pointer is.
- **Loss (decision 10).** MSE training was predicted to shrink contrast. Your records: output energy 82% of target after MSE, 95% after fine-tuning through the frozen target, and the gain from fine-tuning (about +3 J&F over `legacy`) is larger than any structure difference. **Agrees.**
- **Zero initialisation, same shape, attention within one record (decisions 3, 5, 8).** Your records do not test these. **Silent.**

### 6.4 What the records add to the recommendation

Section 1 stays as written. The records add three things, as MY DATA:

1. On Small -> Base+, structure changes tracking by about one J&F point or less after fine-tuning, and the training loss changes it by about three. So I see no sign that a pair-specific structure would give a very large gain. Keep one structure.
2. The form that your review actually trained and that worked is "ridge-initialised affine skip + per-position MLP + ridge pointer". That is B plus the two changes that the literature part also suggests. If you choose B, describe that form, not `M + MLP(M)` with a pointer MLP.
3. Nothing in the records supports A or B for Small -> Large or Base+ -> Large. Both pairs have linear maps only.

### 6.5 Honest wording where the records disagree or are silent

- A versus B: "On Small -> Base+, the two structures could not be distinguished after fine-tuning (differences within about one J&F point on two external sets, with intervals that include zero). We use the per-position structure because it is simpler and cheaper." Do not write that B is more accurate.
- Spatial context: "Adding spatial context lowered the state regression error but did not change tracking accuracy in our experiments." Both halves are needed.
- Three pairs: until B is trained for Small -> Large and Base+ -> Large, write "we design one structure for all three pairs and evaluate it on Small -> Base+; for the other two pairs we report linear baselines only". Change it when the runs exist.
- Base+ -> Large: "a linear fit suggests that pairs with Large as the target are harder (R² .66 to .72 against .75 to .78), but we have no tracking result for this pair."
- Scores on MOSE dev and LVOS val: report the scene-clean versions and call M3-VOS and PUMaVOS the external test. The first-run model also trained on the review's dev cases and was selected on LVOS val, so comparisons on those sets favour it.
- State loss: the first-run report itself says that a lower state loss is not evidence of better segmentation. Keep that sentence.

---

## 7. Suggested wording (LITERATURE)

For a Related Work or Design Rationale section. Adjust "B" and the skip form to your final choice.

> Model stitching connects two frozen networks through a small trainable layer that acts identically at every spatial position (Lenc & Vedaldi, 2015; Bansal et al., 2021; Csiszárik et al., 2021), and Stitchable Neural Networks apply it to several sizes of one model family, initialising the layer by least squares (Pan et al., 2023). Our translator follows this form: it maps each position of a stored memory record with the same channel-wise function, and it starts from the closed-form linear solution. We add one nonlinear layer to this map and examine its effect by ablation; we do not claim that a nonlinear map is required in general. The three SAM 2.1 sizes produce memory records of the same shape, so no resizing is needed, but equal shape does not imply shared channel coordinates (Li et al., 2016), which is why a learned map is needed at all. To our knowledge, translating the memory bank of a video object segmentation model for use by a different model has not been studied; the closest settings are key-value cache translation between language models and embedding upgrades without recomputation (Ramanujan et al., 2022). We use one translator structure for all model pairs and train separate weights for each pair with the same procedure.

If you choose A, replace the third sentence with:

> In addition to the per-position linear map, we add a branch that attends over a coarse grid of the same record, motivated by the observation that features of different architectures at the same location need not carry the same content (Lin et al., 2022); whether this branch helps is an empirical question that we test by ablation.

---

## 8. Citations

Tags: **confident** = verified on the web in this session (marked ✓) or well known to me; **UNVERIFIED** = check before use.

Stitching, alignment, similarity

1. Lenc, Vedaldi. Understanding image representations by measuring their equivariance and equivalence. CVPR 2015. confident ✓
2. Bansal, Nakkiran, Barak. Revisiting Model Stitching to Compare Neural Representations. NeurIPS 2021. confident ✓
3. Pan, Cai, Zhuang. Stitchable Neural Networks. CVPR 2023. confident ✓ (least-squares init, Fast-to-Slow and nearest stitching read from the paper's main text)
4. Yang, Zhou, Liu, Ye, Wang. Deep Model Reassembly. NeurIPS 2022. confident
5. Hernandez, Dangovski, Lu, Soljačić. Model Stitching: Looking For Functional Similarity Between Representations. arXiv 2303.11277, 2023 (NeurIPS 2022 workshop). confident ✓
6. Smith, Mannering, Marcu. Functional Alignment Can Mislead: Examining Model Stitching. ICML 2025 (PMLR 267). confident ✓
7. Lähner, Moeller. On the Direct Alignment of Latent Spaces. UniReps workshop, PMLR 243, 2024. confident ✓
8. Merullo, Castricato, Eickhoff, Pavlick. Linearly Mapping from Image to Text Space. ICLR 2023. confident
9. Huh, Cheung, Wang, Isola. The Platonic Representation Hypothesis. ICML 2024. confident
10. Raghu, Gilmer, Yosinski, Sohl-Dickstein. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. NeurIPS 2017. confident
11. Morcos, Raghu, Bengio. Insights on representational similarity in neural networks with canonical correlation. NeurIPS 2018. confident
12. Kornblith, Norouzi, Lee, Hinton. Similarity of Neural Network Representations Revisited. ICML 2019. confident
13. Nguyen, Raghu, Kornblith. Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth. ICLR 2021. confident ✓ (findings verified; venue from memory)
14. Ding, Denain, Steinhardt. Grounding Representation Similarity Through Statistical Testing. NeurIPS 2021. confident
15. Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky. Reliability of CKA as a Similarity Measure in Deep Learning. ICLR 2023. confident
16. Li, Yosinski, Clune, Lipson, Hopcroft. Convergent Learning: Do different neural networks learn the same representations? ICLR 2016. confident
17. Ainsworth, Hayase, Srinivasa. Git Re-Basin: Merging Models modulo Permutation Symmetries. ICLR 2023. confident
18. Entezari, Sedghi, Saukh, Neyshabur. The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. ICLR 2022. confident
19. Klabunde, Schumacher, Strohmaier, Lemmerich. Similarity of Neural Network Models: A Survey of Functional and Representational Measures. arXiv 2305.06329. Paper confident ✓; journal venue (ACM Computing Surveys) UNVERIFIED. Not cited in the text; useful as a survey.

State translation and compatible representations

20. Shen, Xiong, Xia, Soatto. Towards Backward-Compatible Representation Learning. CVPR 2020. confident
21. Ramanujan, Vasu, Farhadi, Tuzel, Pouransari. Forward Compatible Training for Large-Scale Embedding Retrieval Systems. CVPR 2022. confident ✓
22. Wang, Chang, Yang, Chen, Lai. Unified Representation Learning for Cross Model Compatibility. BMVC 2020. confident ✓ (the remark that a deeper MLP did not help comes from a search summary; check the paper before quoting it)
23. Su et al. On Transferability of Prompt Tuning for Natural Language Processing. NAACL 2022. confident ✓
24. Lester, Yurtsever, Shakeri, Constant. Reducing Retraining by Recycling Parameter-Efficient Prompts. arXiv 2208.05577, 2022. confident ✓
25. Oozeer, Nathawani, Prakash, Lan, Harrasse, Abdullah. Activation Space Interventions Can Be Transferred Between Large Language Models. ICML 2025. confident ✓
26. Jha, Zhang, Shmatikov, Morris. Harnessing the Universal Geometry of Embeddings. arXiv 2505.12540, 2025. confident ✓ (preprint)
27. Mikolov, Le, Sutskever. Exploiting Similarities among Languages for Machine Translation. arXiv 1309.4168, 2013. confident
28. Xing, Wang, Liu, Lin. Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation. NAACL 2015. confident
29. Smith, Turban, Hamblin, Hammerla. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. ICLR 2017. confident
30. Conneau, Lample, Ranzato, Denoyer, Jégou. Word Translation Without Parallel Data. ICLR 2018. confident

Residual adapters and initialisation

31. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016. confident
32. Houlsby et al. Parameter-Efficient Transfer Learning for NLP. ICML 2019. confident
33. Rebuffi, Bilen, Vedaldi. Learning multiple visual domains with residual adapters. NeurIPS 2017. confident
34. Gao et al. CLIP-Adapter: Better Vision-Language Models with Feature Adapters. IJCV 2024 (arXiv 2021). confident
35. Chen et al. AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition. NeurIPS 2022. confident
36. Kim, Lee, Lee. Accurate Image Super-Resolution Using Very Deep Convolutional Networks. CVPR 2016. confident
37. Zhang, Zuo, Chen, Meng, Zhang. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP 2017. confident
38. Goyal et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv 1706.02677, 2017. confident
39. Zhang, Dauphin, Ma. Fixup Initialization: Residual Learning Without Normalization. ICLR 2019. confident
40. Bachlechner, Majumder, Mao, Cottrell, McAuley. ReZero is All You Need: Fast Convergence at Large Depth. UAI 2021. confident
41. De, Smith. Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks. NeurIPS 2020. confident
42. Touvron, Cord, Sablayrolles, Synnaeve, Jégou. Going deeper with Image Transformers. ICCV 2021. confident
43. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. confident
44. Zhang, Rao, Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023. confident
45. Alayrac et al. Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS 2022. confident
46. Zhang et al. LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention. ICLR 2024. confident
47. Chen, Goodfellow, Shlens. Net2Net: Accelerating Learning via Knowledge Transfer. ICLR 2016. confident

Spatial context, architecture

48. Lin, Xie, Wang, Yu, Chang, Liang, Wang. Knowledge Distillation via the Target-aware Transformer. CVPR 2022. confident ✓
49. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. confident
50. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML 2020. confident
51. Luo, Li, Urtasun, Zemel. Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. NeurIPS 2016. confident
52. Yang, Luo, Li, Deng, Guibas, Krishnan, Weinberger, Tian, Wang. Denoising Vision Transformers. ECCV 2024. confident ✓
53. Darcet, Oquab, Mairal, Bojanowski. Vision Transformers Need Registers. ICLR 2024. confident
54. Bolya, Ryali, Hoffman, Feichtenhofer. Window Attention is Bugged: How not to Interpolate Position Embeddings. ICLR 2024. confident ✓
55. Ryali et al. Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles. ICML 2023. confident
56. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024. confident

Distillation, capacity gap, losses

57. Romero et al. FitNets: Hints for Thin Deep Nets. ICLR 2015. confident
58. Heo et al. A Comprehensive Overhaul of Feature Distillation. ICCV 2019. confident
59. Mirzadeh et al. Improved Knowledge Distillation via Teacher Assistant. AAAI 2020. confident
60. Cho, Hariharan. On the Efficacy of Knowledge Distillation. ICCV 2019. confident
61. Johnson, Alahi, Fei-Fei. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. ECCV 2016. confident
62. Mathieu, Couprie, LeCun. Deep multi-scale video prediction beyond mean square error. ICLR 2016. confident

Baselines and reporting

63. Hoerl, Kennard. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 1970. confident
64. Alain, Bengio. Understanding intermediate layers using linear classifier probes. ICLR workshop 2017 (arXiv 2016). confident
65. Hewitt, Liang. Designing and Interpreting Probes with Control Tasks. EMNLP 2019. confident
66. Dodge, Gururangan, Card, Schwartz, Smith. Show Your Work: Improved Reporting of Experimental Results. EMNLP 2019. confident
67. Bouthillier et al. Accounting for Variance in Machine Learning Benchmarks. MLSys 2021. confident
68. Musgrave, Belongie, Lim. A Metric Learning Reality Check. ECCV 2020. confident
69. Lipton, Steinhardt. Troubling Trends in Machine Learning Scholarship. ACM Queue 2019 (arXiv 2018). confident

Details to check in the PDFs before quoting: the exact stitching-layer form in Bansal et al. (I state it as a 1x1 convolution for CNNs and a shared linear map for Transformer tokens, from memory); the width-similarity statements in Kornblith et al. and Morcos et al. (from memory); and the appendix results of Stitchable Neural Networks on non-adjacent stitches (only the main-text statement was read).

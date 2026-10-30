# Design memo: beating GrowFields with growth priors under small data

Date: 2026-10-30. Author: Claude (research pass for Ayush / Yancong Lin). Status: proposal, nothing implemented.
Scope: one benchmark (GrowFields, arXiv 2607.03330, TrackPlant3D test split), ~120 real sequences available for training (TrackPlant3D 43 plants / 378 scans; Pheno4D 7 maize + 7 tomato; Sugar4D 48 sugar beet × 16 time points), 5–20 frames each, organ instance labels. Facts about GrowFields come from the arXiv HTML and the released code (`Joaquin-Gajardo/growfields`, commit 3f8e947); everything marked **[spec]** is speculation.

---

## 0. What GrowFields actually is (one paragraph, so the rest is precise)

Per plant sequence, each organ i (stem or leaf) is cut out, PCA-canonicalised per frame (principal axis → Z, base → origin; rigid transforms stored), and represented as a point set advected by a shared SIREN velocity field

  v = f_θ(x, t, z_i) ∈ ℝ³,  x ∈ ℝ³ canonical, t ∈ [0,1] organ-normalised scan-day, z_i ∈ ℝ³² auto-decoded,

integrated with forward Euler between observed frames: x^(k) = x^(s) + Σ_{τ=s}^{k-1} f_θ(x^(τ), τ, z_i) Δt_τ (code: `train_integration_steps: 1`, i.e. one Euler step per inter-frame gap; Δt = day gap / organ day range). Loss: bidirectional squared Chamfer at *every* intermediate predicted frame s+1..k (multi-step CD), plus 1e-4·‖z‖². Network: 5 × 512 SIREN (paper suppl. says 256; REPRODUCIBILITY.md says 512 was used), ~1.1 M params; 50 000 Adam steps per sequence, lr 5e-6 (θ) / 5e-3 (z), 10 organs × 8 192 points per step. Emergence: a new organ is appended with its own z at its first observed frame, initialised from that frame's points; nothing is generated. Evaluation: per organ per frame a fixed 50/50 random point split; all 234 annotated leaf tips are in the test half; the whole trajectory is rolled out from each organ's first observed frame; metrics are CD (mm², squared) and leaf-tip EPE (mm) averaged over frames, sequences, 5 seeds; predictions are re-posed with GT per-frame rigid transforms. Main table (8 test seqs, fit on all frames): GF 0.86 / 1.47 vs COAP 1.69 / 4.23 vs DPF 2.47 / 15.15. Interpolation (train every 2nd frame): GF 10.29 / 5.82 vs DPF-per-organ 10.24 / 4.00 (GF loses on EPE). Extrapolation (hold out last 3 frames): GF 149.63 / 18.99 vs COAP 451.09 / 31.39, with maize/sorghum at 320–360 mm². Test sequences: MaC2 (9 frames), MaC3 (8), SoC2 (8), SoH2 (5), TbC1 (7), TbS3 (8), Tm1H3 (16), Tm1S1 (17); 4 validation sequences, one per species.

Two protocol facts matter for strategy. (a) The headline CD is a *fitting* score: test points are other samples of the same observed surfaces, so the model that reproduces each observed frame most densely wins. (b) Only the interpolation / extrapolation tables measure generalisation in time, and that is where priors can produce order-of-magnitude gains. Both tables must be reported to claim a win.

---

## 1. GrowFields' limitations that we can exploit

1. **No transfer across plants.** θ and {z_i} are re-fitted from scratch for each of the 8 test sequences. The "cross-organ prior" is only across organs of one plant. Short organ tracks (2–4 frames) are underconstrained; the paper itself names 5-frame SoH2 and short sequences as failure cases. ~110 other sequences of the same species are unused.
2. **Unconstrained velocity.** f_θ may point inward: organs can shrink, fold, or oscillate. Nothing encodes "length is non-decreasing". In extrapolation the Euler rollout continues at whatever velocity was last fitted — linear, unbounded growth, no saturation.
3. **Time is organ-relative and leaks.** t is normalised by each organ's own observed day range (`use_global_tnorm: false`), so t = 1 means "last observation", not an age; in extrapolation the range is computed over *all* frames including held-out ones (REPRODUCIBILITY.md §6). Organs with different lifespans share one clock, which blurs the developmental program the field is supposed to learn. Absolute organ age in days is the natural variable for allometry and is never used.
4. **Fixed point budget.** Points from the first frame are advected and never densified; "regions undergoing rapid growth become spatially sparse" (their words). Since CD includes a GT→pred term, sparsity on the growing tip is paid for directly in the headline metric. Maize (CD 2.5 and 1.4, the two worst sequences) is exactly the fast-elongating case.
5. **Emergence is not modelled.** A leaf that first appears in a held-out frame contributes its full Chamfer mass to the error (the paper attributes part of the 300+ mm² extrapolation CD to this). No emergence time, no initial shape prior, no generative sampling, no uncertainty.
6. **Canonicalisation is per frame from GT labels and PCA**, so pose is factored out by the benchmark (fine for us too), but the model never sees the plant-level context (leaf rank, stem height) that determines an organ's growth curve.
7. **Expensive per sequence**: 50 k steps per plant; no amortisation. A learned prior should fit a new plant in a few hundred to a few thousand steps.

Where we should expect to gain: 2+4 on the headline table (modest), 1+2+3+5 on interpolation/extrapolation (large). **[spec]** Headline CD 0.86 → ~0.6–0.7 is realistic; extrapolation 149 → well under 100 is plausible if monotone saturating growth and a species prior are in place; emergence generation is the only route to further gains on MaC2/MaC3/SoC2 extrapolation.

---

## 2. Shared scaffolding (used by all three candidates)

- **Canonical organ frame**: reuse GrowFields'/PSA's leaf canonicalisation (base at origin, principal axis = +Z). Define for each point its base distance ρ = ‖x‖ and axial coordinate x₃. Stems: axis = gravity, base = soil point.
- **Clocks**: organ age a_i(d) = d − d_i^first (days, ≥0), plant age p(d) = d − d^first, species one-hot/embedding s. Never normalise by a held-out range.
- **Data for training the prior**: all training sequences of TrackPlant3D (31 plants outside the 8 test + 4 val), Pheno4D, Sugar4D, canonicalised identically. **Leakage check (required)**: TrackPlant3D maize and tomato2 are re-samples of the Bonn 4D-registration / Pheno4D plants; any Pheno4D plant that is the same physical plant as a test sequence (MaC2, MaC3, Tm1H3, Tm1S1 — Tm1 is from Conn et al., so probably safe) must be excluded from the prior's training set. Match by scan dates and heights before trusting any number.
- **Densification (cheap, orthogonal, apply to every candidate and to the GF re-run as an ablation)**: at inference, advect 4× the source points (resample the first-frame surface by jittering along estimated normals' tangent plane within the local k-NN radius). Addresses limitation 4 without touching the model. Expected to move the headline CD on maize noticeably. **[spec]**

---

## 3. Candidate A — MonoGrow: amortised GrowFields with a monotone velocity field and a learned cross-plant prior

**Math.** Keep Lagrangian advection in the canonical frame but factor the velocity into a guaranteed-outward growth part and a tangential residual:

  v(x, a, c) = softplus(g_θ(x, a, c)) · x̂ + P_⊥(x) w_θ(x, a, c),   x̂ = x/‖x‖,  P_⊥(x) = I − x̂x̂ᵀ,

with c = [z_i; s; e(p)] (organ code, species embedding, plant-age features). Then d‖x‖²/dt = 2 xᵀv = 2 softplus(g)‖x‖ ≥ 0, so every point's distance from the organ base is non-decreasing along the flow: length, span and (for a convex envelope) size are monotone by construction while bending/twisting is still representable through w. Saturation is encoded by multiplying the growth term by a learned gate σ(k_θ(c)·(a_e(c) − a)) **[spec, optional]**, which makes the extrapolated rate decay instead of staying constant. Integration: Euler or RK2 on a fixed daily grid (Δa = 1 day), so intermediate frames and extrapolation share one discretisation.

**Parameterisation.** g_θ, w_θ: one SIREN trunk (5 × 256, ω₀ = 30) with two heads, input [x; a/a_max; c], ≈ 0.4 M params. Species embedding 8-d (5 species), z_i ∈ ℝ³² auto-decoded. Hypernetwork variant: h_φ(c) emits the last two layers of the SIREN; keep as an ablation only, concat conditioning is what GF found best.

**Loss (training the prior across sequences).** For organ i of training plant n, sample source frame s, target k > s, roll out, and use GF's multi-step CD plus:
  L = Σ_i [ Σ_{τ=s+1}^{k} CD(X̂_i^(τ), X_i^(τ)) ] + λ_z‖z_i‖² + λ_w E‖P_⊥w‖² + λ_g E[softplus(g)] 
(residual and growth-rate penalties keep the field smooth; the last term discourages spurious growth in frames where none is observed).
Because every organ needs a z, and GF's fitted z are themselves our training targets, fit all z_i jointly with θ (DeepSDF auto-decoding). After training, fit a Gaussian (or a small mixture) q(z | s) to the trained codes; this is the cross-plant prior used at test time.

**Emergence.** Not generated in A (same as GF): new organs enter at first observation. The gain here is that an organ seen in only 2–3 frames inherits the species growth program through θ and q(z | s). Emergence generation is left to B.

**Inference on the benchmark.** Freeze θ. For each test organ initialise z_i = μ_s and run N_z Adam steps (≈1 000) on the observed training points of the frames allowed by the protocol (all frames for Table 1; every 2nd for interpolation; all-but-3 for extrapolation), minimising multi-step CD + λ(z−μ_s)ᵀΣ_s⁻¹(z−μ_s). Optional second stage: unfreeze θ with lr 1e-6 for 2–5 k steps on the test sequence (this exactly recovers GF's regime as a special case, so we can only gain). Roll out from each organ's first frame, re-pose with GT transforms, score with GF's code.

**Prior encoded**: outward-only growth, smoothness, species-conditional distribution of growth programmes, saturation.

**Pros**: smallest conceptual delta from GF, reuses their code (`shared_continuous` model, integration, metrics, splits), guaranteed monotone, test-time fitting cost drops 10–50×, extrapolation becomes bounded. **Cons**: still fits a per-organ code at test time; cannot create organs; compound leaves (tomato) grow new leaflets off-axis — the outward-radial constraint is correct for leaflets' distance from the leaf base but P_⊥w must carry the unfolding; the plant-base of a stem is a poor "origin" for stem thickening (treat stem radius growth as x̂ in the cross-section plane — same formula with x̂ ⊥ gravity).

**Compute.** Prior training: ~110 sequences × ~8 organs ≈ 900 organ tracks; 150–200 k steps at GF's batch (10 organs × 8 192 pts) ≈ 6–10 h on one A10G/L4 (GF's 50 k steps per sequence is of the same order). Test-time: ~2 min per sequence for z-only fitting, ~10 min with θ fine-tuning. Params: 0.4–1.1 M + 32 per organ.

---

## 4. Candidate B — TrajDiff: latent diffusion over per-organ growth-trajectory codes with a monotone template decoder

**Representation.** Each organ is a canonical template plus a trajectory code: template T_i = its first observed cloud normalised to unit length (or, for generated organs, a species template), code c_i = (c_i^(1..K)) ∈ ℝ^{K×d}, K = 8 tokens placed at ages a_k = a_max·k/K, d = 32. Decoder (stage 1), shared across all plants and species:

  x(a) = S(a; c_i) ⊙ x₀ + ε_φ(x₀, a, c_i),   S(a) = exp( ∫₀^a softplus(r_φ(τ, c_i)) dτ ) ∈ ℝ³₊ (per-axis cumulative log-scale),

where ε_φ is a small residual deformation with ‖ε‖ penalised and, to keep monotonicity of base distance, projected so that x₀ᵀε ≥ 0 (e.g. ε = P_⊥ε̃ + softplus(ε̃_∥)x̂₀). The integral is a 1-D quadrature over days (UMNN-style monotone network, Wehenkel & Louppe 2019): S is non-decreasing per axis, so length/width/thickness scale monotonically; the per-axis log-scale form directly expresses allometry (log W = β log L + const is a linear relation between the axes' integrated rates). c_i is read at age a by linear interpolation of the K tokens (or cross-attention), so the decoder is time-conditioned and continuous in a.

**Stage 1 (auto-decoding across all sequences).** Minimise Σ_i Σ_frames CD(x(a_k; c_i, T_i), X_i^(k)) + λ‖c_i‖² over φ and {c_i}; same point sampling as GF. Output: ≈ 900–1 100 training codes c_i with metadata (species, leaf rank, plant age at emergence a_i^first, final observed length).

**Stage 2 (diffusion prior).** A small DiT (6 blocks, width 256, K = 8 tokens + conditioning tokens) learns p(c | s, rank, p_emerge) with a standard ε/v-prediction objective (DDPM, 1 000 steps, or flow matching). Conditioning dropout for classifier-free guidance. Also learn an **emergence model**: p(a_{j+1}^first − a_j^first | s, rank j, p) as a small conditional density (mixture of log-normals; or a second token in the DiT) — this is the phyllochron and is where plant science has 60 years of evidence that the quantity is low-dimensional (thermal-time phyllochron models; Plant Methods 2023 time-to-event model).

**Inference on the benchmark.** Two options. (i) *Posterior sampling* (reconstruction guidance / DPS): run the reverse diffusion for each test organ while adding the gradient of CD(observed frames) w.r.t. the current estimate ĉ₀; needs the decoder to be differentiable through to CD, which it is. (ii) *MAP with diffusion prior* (simpler, deterministic, recommended first): optimise c_i on observed frames with the loss CD + λ·‖ε_ψ(c_i + σ_t η, t) − η‖² averaged over a few noise levels (score-distillation-style regulariser), then roll out. For generation of organs not yet observed (extrapolation only): sample emergence times from the emergence model given the plant's observed phyllochron, sample c from the prior conditioned on species/rank, decode with the species template, place using the plant-level stem model and the GT pose convention (extrapolation re-poses with GT transforms, which for an *unobserved* organ do not exist — we must predict pose; flag this as a protocol gap to raise with the authors or report both "with/without generated organs").

**Prior encoded**: distribution over whole growth programmes (not just a Gaussian on codes), monotone allometric scaling, emergence timing, uncertainty (multiple samples → CD of the best-of-N and of the mean; calibration curves).

**Pros**: the only candidate that can produce the missing organs and uncertainty, and the one closest to the lab's longer-term generative goal; the stage-1 decoder is Candidate A's cousin so code is shared. **Cons**: two stages; ~1 000 trajectories is very small for a diffusion model (expect to need heavy augmentation: random time-warps, mirror in canonical frame, per-axis scale jitter, mixing codes between organs of the same rank) **[spec]**; generated organs placed with predicted poses will likely *increase* CD on the current protocol if the pose is off by more than the leaf size, so the headline risk is on the benchmark, not on scientific merit; diffusion sampling per organ adds minutes.

**Compute.** Stage 1 as A (6–10 GPU-h). Stage 2: ~5 M params, 1 000 codes × K tokens, trains in <1 h. Inference (MAP) ~ same as A; posterior sampling ~ 50–250 decoder calls per organ.

---

## 5. Candidate C — RichardsNet: explicit growth curves per organ + species hierarchy + learned residual

The strong structured baseline; if it is competitive, A and B must beat it, not GF.

**Math.** Per organ i, scalar size trajectories follow a determinate-growth curve. Richards (1959):
  L_i(a) = A_i [1 + ν_i exp(−k_i (a − a_{m,i}))]^{−1/ν_i},
or Yin et al. (2003) beta-sigmoid, which has zero rate at both ends and is the plant-science default for leaf elongation (LEAF-E, Plant Methods 2014):
  L_i(a) = L_{m,i} (1 + (a_{e,i} − a)/(a_{e,i} − a_{m,i})) (a/a_{e,i})^{a_{e,i}/(a_{e,i} − a_{m,i})},  0 ≤ a ≤ a_{e,i}, constant after.
Width by allometry W_i(a) = α_i L_i(a)^{β_s} (β_s species-level). Deformation of the template X_i^0 (first frame, length L_i^0):
  x(a) = diag(W_i(a)/W_i^0, W_i(a)/W_i^0, L_i(a)/L_i^0) x₀ + ε_ψ(x₀, a/a_{e,i}, s, rank),
with ε_ψ a tiny shared MLP (4 × 128, ≈ 70 k params) trained across all training sequences to capture systematic non-affine growth (tip bending, lamina unrolling), penalised by ‖ε‖² and by the monotone-projection trick from B. Hierarchical prior: (log A, log k, a_m, log ν) ~ N(μ_s + Δ_rank, Σ_s) estimated by empirical Bayes from training organs (fitting each training organ's PCA length series with a 4-parameter nonlinear least squares, then pooling). Emergence: fit a per-species phyllochron Gaussian on Δa^first between successive ranks.

**Loss / fitting.** Per test organ: MAP estimate of the 4–5 curve parameters from observed length series (lengths measured as 99th-percentile axial extent in the canonical frame), closed-form-ish (Levenberg–Marquardt with the prior as a quadratic penalty — milliseconds), then ε_ψ is *not* re-fitted (frozen shared residual) or fitted with 1–2 scalar gains. Points at a are decoded in closed form. Note this fits sizes, not Chamfer; a final Chamfer-based refinement of (A, k) is a cheap optional step.

**Prior encoded**: monotone, saturating, 4-parameter developmental program; species/rank hierarchy; allometry; phyllochron. **Pros**: ~5 scalars per organ, interpretable traits for free (final length, max elongation rate, elongation duration), robust with 2–3 frames because the prior fixes shape, extrapolation saturates. **Cons**: axis-aligned scaling cannot model tomato compound-leaf unfolding or leaf-tip droop; headline CD will likely be *worse* than GF on tomato/tobacco **[spec]** unless ε_ψ carries real deformation; the template is the (small, noisy) first frame, so template quality bounds the fit — a mid-sequence template with inverse scaling for early frames is better (allowed: the protocol rolls out from the first frame but nothing forbids using later observed frames to define the template, except in extrapolation, where only the training frames may be used).

**Compute.** Negligible: seconds per sequence on CPU for the curve fits; ε_ψ trains in <1 GPU-h.

---

## 6. Relevant prior work (2022–2026 unless foundational)

Neural deformation fields / 4D priors across instances
- GrowFields (Gajardo et al., ECCV 2026) — arXiv 2607.03330. Target.
- Dynamic Point Fields (Prokudin et al., ICCV 2023) — arXiv 2304.02626; GF's strongest full-plant baseline; two-frame deformation fields with keypoint guidance.
- DOMA, "Degrees of Freedom Matter: Inferring Dynamics from Point Trajectories" (Zhang et al., CVPR 2024) — GF's code is a fork of DOMA's reference implementation (README acknowledgement); arXiv id not verified in this pass.
- COAP (Mihajlovic et al., CVPR 2022) — arXiv 2204.06184; compositional part-aware occupancy, GF's part-aware baseline.
- Neural Deformation Graphs (Božič et al., CVPR 2021) — arXiv 2012.01451. CaDeX (Lei & Daniilidis, CVPR 2022; id unverified) — canonical map + homeomorphism per frame; the "shared canonical template + per-instance deformation code" pattern used in B and C.
- Motion2VecSets (Cao et al., CVPR 2024; TPAMI 2026) — arXiv 2401.06614; shape latent set + per-frame deformation latent sets denoised synchronously across time; the closest existing design to Candidate B (there the shapes are animals, topology fixed).
- HyperDiffusion (Erkoç et al., ICCV 2023) — arXiv 2303.17015; weight-space diffusion over per-sample MLPs, 4D on DeformingThings4D; shows that diffusing over *fitted field parameters* works with ~1.8 k sequences — our ~1 k organ trajectories is the same order.
- DNF: Unconditional 4D Generation with Dictionary-based Neural Fields (Zhang, Li, Dai, CVPR 2025 oral) — arXiv 2412.05161; shape vs motion latents with a shared dictionary + transformer diffusion; the factorisation we copy (template × trajectory code).
- Learning Neural Deformation Representation for 4D Dynamic Shape Generation (Han et al., 2026) — arXiv 2606.01021; part-wise skinning + separate motion latent space, diffusion on top; argues separating motion from shape fixes HyperDiffusion's temporal inconsistency.
- Sculpt4D — arXiv 2604.21592 (frame-wise VAEs give jittery latents → sequence-level latents; motivates trajectory tokens in B).

Learned priors + test-time latent optimisation
- DeepSDF (Park et al., CVPR 2019) — arXiv 1901.05103; auto-decoding, used by GF and by A/B.
- MetaSDF (Sitzmann et al., NeurIPS 2020) — arXiv 2006.09662; meta-learned initialisation beats auto-decoders when observations are sparse (surface-only) — relevant to the "2–3 frames per organ" regime; a MAML variant of A is an ablation.
- Functa (Dupont et al., ICML 2022) — arXiv 2201.12204; modulation-only fitting of shared fields; cheaper than full z optimisation.
- Survey of non-rigid reconstruction incl. test-time optimisation of priors — arXiv 2403.15064.

Monotone / positive constraints
- Unconstrained Monotonic Neural Networks (Wehenkel & Louppe, NeurIPS 2019) — arXiv 1908.05164; monotone function = ∫ positive network; used for S(a) in B and the optional saturating gate in A.
- Input Convex Neural Networks (Amos et al., ICML 2017) — arXiv 1609.07152; non-negative weights + softplus; alternative construction.
- "Learning Monotone Dynamics by Neural Networks" — arXiv 2006.06417; monotone dynamical systems via non-negative network structure.
- GrowFlow, "Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields" (Luo et al., 2026) — arXiv 2602.08958; assumes monotonically non-decreasing visible volume, models growth *backwards* in time from the mature plant so no primitives need to be added; HexPlane velocity + neural ODE on Gaussians. The reverse-time trick is a legitimate alternative handling of emergence (shrink organs to zero backwards) — compatible with A's outward-only field run in reverse. **[spec]**

Plant-specific 4D and growth models
- Structure-aware Riemannian Growth Fields for 4D Plant Modeling (Kuo & Kawahara, WACV 2027) — arXiv 2608.13007; geodesic flow on a structure-aware Riemannian field with symbolic growth rules; releases a 10-day two-species dataset (candidate extra training data).
- Denoising-Aware Temporal Point Cloud Completion / Adaptive Temporal PoinTr + SynthCrop4D (Mittal & Kar, 2026) — arXiv 2608.28343; t−1 → t completion on Pheno4D; synthetic growing-plant clouds usable for pre-training.
- PlantDreamer (Hartley et al., ICCVW 2025) — arXiv 2505.15528; diffusion-guided 3DGS from L-system meshes; static, but L-system templates are a source of species templates for B/C.
- 4DPlants / 3D L-Plants (Hétroy-Wheeler et al., CVPR 2026; L-system latent representation of plants; code on 4dplants.icube.unistra.fr; arXiv id not found).
- Review of data-driven plant growth modelling — arXiv 2412.10538 (flags absence of probabilistic 3D growth models).
- Datasets: Pheno4D (Schunck et al., PLOS ONE 2021); TrackPlant3D (Li et al., COMPAG 2024); Sugar4D (Bonn, 48 plants × 16 time points, LiDAR, organ labels); BonnBeetClouds3D — arXiv 2312.14706.

Growth curves and emergence in plant science
- Richards, J. Exp. Bot. 1959 — flexible growth function; Yin et al., Annals of Botany 2003 — beta-sigmoid of determinate growth (zero rate at onset and cessation; outperformed logistic/Richards/Gompertz on maize, pea, wheat); LEAF-E (Voorend et al., Plant Methods 2014) — fits beta-sigmoid to leaf length to extract elongation rate and duration; Karadavut et al. 2010 — Richards/logistic/Gompertz best among nonlinear models for maize leaf growth.
- Phyllochron / leaf appearance: linear thermal-time phyllochron vs the time-to-event model of phyllochron dynamics (Plant Methods 2023, PMC10245529); Wang–Engel nonlinear leaf-number model (RMSE ≈ 0.8 leaves). Basis for the emergence model in B and C.

---

## 7. Recommended first experiment and ablation plan

**First experiment (simplest thing that could beat GrowFields): Candidate A without the saturation gate, plus densification.**
1. Re-run GF's released code on the 8 test sequences, 5 seeds, to reproduce 0.86 / 1.47 and the interpolation/extrapolation tables (fix the extrapolation normalisation leak only as a reported variant; keep their protocol for the headline comparison).
2. Modify their `shared_continuous` model in two lines: velocity head → softplus(g)·x̂ + P_⊥w (Section 3). Train per sequence exactly as GF (no prior yet). This isolates the monotone constraint. Report headline, interpolation, extrapolation.
3. Train one θ on all non-test, non-val sequences (after the leakage check) with per-organ codes; fit q(z | species). Test: z-only fitting (1 k steps), then z + θ fine-tune (2 k steps). Report the same three tables plus wall-clock.
4. Add 4× source-point densification at inference to both GF and ours.
Decision rule: if (3) does not beat (2) on extrapolation CD by a clear margin on maize/sorghum, the cross-plant prior is not transferring and effort should move to C (explicit curves) before B.

**Ablations** (each on the 8 test sequences, 5 seeds; report CD, EPE, and the growth metrics below):
- Monotone velocity on/off (per-sequence training) — isolates the prior.
- Prior training set: TrackPlant3D-only vs + Pheno4D vs + Sugar4D vs + Riemannian-growth-fields dataset — data scaling.
- Test-time: z only vs z+θ vs MetaSDF-style inner loop — amortisation quality.
- Conditioning: z only vs z + species vs z + species + plant age — what the field actually uses.
- Clock: GF organ-normalised t vs absolute organ age in days.
- Latent regulariser: L2 vs Gaussian q(z|s) vs diffusion prior (B-MAP) — this is the bridge from A to B.
- Densification factor 1×/2×/4×/8×.
- Observed-frame budget: fit on the first 2, 3, 5 frames and extrapolate the rest — the few-shot curve nobody has reported.
- C as a baseline on all tables; C + ε_ψ; A with C's Richards gate.

**Metrics beyond Chamfer** (all computable from GF's output PLYs + GT labels, so they can be reported for the baselines too):
- *Monotonicity violation rate*: fraction of (organ, consecutive frame) pairs with predicted length decrease > 2 mm, length = 99th-percentile axial extent in the canonical frame; and *negative growth mass* Σ max(0, L̂(t) − L̂(t+1)) in mm. GF's own GT has measurement noise, so report the GT's violation rate as the floor.
- *Organ-size trajectory error*: MAE of L̂_i(t) vs L_i(t) (mm) and of projected leaf area (GF ships a leaf-area script); report separately for seen and held-out frames and per extrapolation horizon (+1, +2, +3 days).
- *Growth-rate error*: MAE of per-day elongation rate (mm/day), since GF advertises growth speed as a trait.
- *Emergence timing*: for models that generate organs, |â_e − a_e| in days and precision/recall of emerged organs within ±1 day; for models that do not, report CD with and without frames containing newly emerged organs so the two failure modes are separated.
- *Density ratio*: local point density of predictions vs GT on the distal 20% of each leaf (diagnoses limitation 4 directly).
- *Leaf-tip EPE* and *F-score at 5 mm* as the standard complements to squared CD (squared CD is dominated by outliers, which hides sparsity).
- For B only: best-of-N and mean-sample CD, and calibration (coverage of GT lengths by the sampled 90 % interval).

**Honest expectations [spec].** Headline table: small improvement (0.86 → 0.6–0.75), mostly from densification and the monotone constraint on the short maize/sorghum tracks; interpolation EPE should fall below DPF's 4.00 if the monotone field removes GF's oscillations; extrapolation is where a 2× or better CD reduction is plausible, with the remaining error dominated by unobserved new organs until B's emergence model exists. The fastest publishable story is A (+ C as the structured baseline), with B as the second paper / the generative extension the lab actually wants.

---

## 8. Open issues to settle before coding
- Leakage between Pheno4D/Bonn plants and TrackPlant3D test sequences (Section 2).
- Extrapolation pose for generated organs: no GT transform exists; propose reporting generated-organ results under a predicted-pose protocol and raising it with the authors.
- Whether to report squared CD in mm² (their convention) alongside RMS CD in mm for readability.
- Stem handling in the monotone field (origin at the soil point; radial growth in the cross-section).

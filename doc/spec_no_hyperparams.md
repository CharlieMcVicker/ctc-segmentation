# Issue: Refactor DP Trellis to Use Unified Relative Contrastive Acoustic Gating

**Status:** Ready for Dev

**Components:** `ctc_segmentation_dyn.pyx`, `ctc_segmentation.py`, `TrellisRuntimeContext`

---

## 1. Context & Motivation

Our fork introduces non-monotonic transitions (blank-constrained vowel syncope and blank-tolerant intrusive detours) to reconcile citation ground truth with conversational surface phonetics. However, the current implementation relies on calibrated log-space penalties (`syncope_penalty`, `intrusive_penalty`/`intrusive_penalties`) and absolute posterior thresholds (`intrusive_min_logprobs`) to prevent the shortest-path exploit (blank-farming) and acoustic hallucinations.

This creates several issues:

1. **Calibration brittleness:** Fixed thresholds fail on lenited, muffled, or noisy speech where absolute posterior peaks drop below the gating threshold.
2. **Hyperparameter overhead:** Developers must hand-tune penalties across different recording conditions and dialects.
3. **Acoustic blocking:** When penalties or thresholds are set too conservatively, the DP trellis is locked out of genuine phonetic realizations, causing lower accuracy on clean data than an unconstrained greedy pass.

We can eliminate these hyperparameters by replacing both subsystems with a unified **Relative Contrastive Acoustic Evidence** model. Because the natural logarithm is strictly monotonic ($x > y \iff \log(x) > \log(y)$), all checks run directly on `lpz` without invoking `exp()`, preserving zero-overhead C-level performance in the `nogil` loop.

---

## 2. Mathematical Formulation

Instead of asking whether an acoustic emission crosses an arbitrary absolute threshold, the dynamic programming engine asks a contrastive local question: **"Does the acoustic evidence at frame $t$ favor the phonological detour over the strict grammatical anchor?"**

### A. Syncope: Skip Gate (Consonant vs. Vowel)

To drop an optional vowel at $c - 1$ and advance directly to the subsequent consonant at $c$, frame $t$ must show stronger acoustic evidence for the incoming consonant than for the skipped vowel:

$$\text{Gate}_{\text{syncope}}(t, c) =  \begin{cases}  \text{OPEN}, & \text{if } \text{lpz}[t + \text{offset}, L(c, s)] > \text{lpz}[t + \text{offset}, L(c - 1, s)] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$

When open, evaluate the transition with zero lambda penalty ($\lambda_{\text{syncope}} = 0.0$):

$$P_{\text{sync}}(t, c) = \text{table}[t - 1 + \Delta_{\text{offset}}, c - 2 \text{ or } c - 3] + \text{lpz}[t + \text{offset}, L(c, s)]$$

### B. Intrusion: Detour Gate (Intrusive Token vs. Next Anchor)

To evaluate an intrusive acoustic detour through token $j_k \in J$ at frame $t_{\text{detour}} = t - 1 - \delta + \text{offset}$, the acoustics must favor the intrusive candidate over the next expected grammatical token $L(c, s)$:

$$\text{Gate}_{\text{intrusive}}(t_{\text{detour}}, c, k) =  \begin{cases}  \text{OPEN}, & \text{if } \text{lpz}[t_{\text{detour}}, j_k] > \text{lpz}[t_{\text{detour}}, L(c, s)] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$

When open, evaluate the path with zero per-token penalties ($\lambda_k = 0.0$):

$$P_{\text{intrusive}}(t, c) = \max_{s, \delta, k} \left( \text{base\_prob} + \text{lpz}[t_{\text{detour}}, j_k] \right)$$

*Note on CTC Blanks:* The transition still accumulates the surrounding blank frame probabilities across stride $\delta$. If the frame is purely uninformative noise, the path cost remains heavily disfavored compared to the standard blank-stay state, naturally preventing spurious detours.

---

## 3. Detailed Implementation Tasks

### Task 1: Update `TrellisRuntimeContext` & Parameter Schema (`ctc_segmentation.py`)

1. Deprecate the following parameters in `CtcSegmentationParameters` with `DeprecationWarning`:
* `syncope_penalty`
* `intrusive_penalty`
* `intrusive_penalties`
* `intrusive_min_logprobs`


2. Retain structural and mask inputs:
* `syncope_tokens` / `is_syncope_token`
* `intrusive_tokens`
* `is_intrusive_site`
* `intrusive_max_stride` (remains necessary as the temporal search window bound)
* `blank`


3. Refactor `TrellisRuntimeContext` to remove penalty and threshold arrays, minimizing struct size passed to the C-layer:
```python
@dataclass(frozen=True)
class TrellisRuntimeContext:
    is_syncope_token: np.ndarray       # 1D int8 (L,)
    intrusive_token_ids: np.ndarray    # 1D int64 (K,)
    is_intrusive_site: np.ndarray      # 1D int8 (L,)
    intrusive_max_stride: int          # int
    blank: int                         # int
    flags: int                         # int bitmask

```



### Task 2: Cython Syncope Logic (`ctc_segmentation_dyn.pyx`)

1. Remove references to `syncope_penalty` in `cython_fill_table`.
2. Prior to computing $P_{\text{sync}}$, extract the token indices for the anchor $c$ and skipped vowel $c - 1$ (or $c - 2$ depending on word-boundary blank interleaving).
3. Gate the skip evaluation:
```cython
cdef int anchor_tok = text_tokens[c]
cdef int vowel_tok = text_tokens[c - 1]  # adjust index for blank interleaving

if lpz[t + offset, anchor_tok] > lpz[t + offset, vowel_tok]:
    # Evaluate Case A / Case B skip transitions without penalty
    p_sync = table[t_prev, c_prev] + lpz[t + offset, anchor_tok]
    if p_sync > max_prob:
        max_prob = p_sync

```



### Task 3: Cython Intrusive Gating Logic (`ctc_segmentation_dyn.pyx`)

1. In the intrusive token candidate scan at $t_{\text{detour}}$, replace the absolute threshold comparison with the relative contrast check against target anchor token $L(c, s)$:
```cython
cdef int target_anchor_tok = text_tokens[c]
cdef int has_candidate = 0
cdef int j_idx, candidate_tok

for j_idx in range(num_intrusive_tokens):
    candidate_tok = intrusive_token_ids[j_idx]
    if lpz[t_detour, candidate_tok] > lpz[t_detour, target_anchor_tok]:
        has_candidate = 1
        break

if not has_candidate:
    continue

```


2. Remove subtraction of $\lambda_k$ when calculating `base_prob + lpz[t_detour, j_k]`.

---

## 4. Acceptance Criteria

* [ ] **Zero Hyperparameter Runtime:** `syncope_penalty`, `intrusive_penalty`, `intrusive_penalties`, and `intrusive_min_logprobs` are completely eliminated from the Cython computational path.
* [ ] **Pure Log-Space Operations:** Zero `exp()` calls in `ctc_segmentation_dyn.pyx`; all comparative logic operates directly on float arrays within the `nogil` block.
* [ ] **Syncope Invariant Tests:**
* Vowel skip is strictly blocked when $P(\text{vowel}) \ge P(\text{anchor})$.
* Vowel skip triggers without numerical penalty when $P(\text{anchor}) > P(\text{vowel})$, preserving the Consonant Preservation Invariant.


* [ ] **Intrusion Invariant Tests:**
* Detour triggers when $P(\text{intrusive}) > P(\text{next\_anchor})$, even when the intrusive posterior peak is sub-0.5 or not the global non-blank argmax.
* Pure noise / blank frames do not cause intrusive detours.


* [ ] **Backward Compatibility:** Users passing legacy parameters to `CtcSegmentationParameters` receive clean deprecation warnings without execution failure.
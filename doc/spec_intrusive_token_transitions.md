# Technical Specification: Intrusive Token Transitions for `ctc-segmentation`

**Version:** 1.0  
**Target Package:** `ctc-segmentation` (Cython core: `ctc_segmentation_dyn.pyx`, Python API: `ctc_segmentation.py`)  
**Status:** Ready for Implementation  

---

## 1. Overview & Problem Statement

In automated speech-to-text alignment and segmentation, the provided "ground-truth" transcript often represents canonical or orthographic citation forms rather than surface phonetic realities. 

### Key Challenge
* In languages with unwritten or variable aspiration, glottal stops, or epenthetic consonants (e.g. Cherokee Syllabary transliteration `akeya` $\rightarrow$ spoken surface realization `akeyha`), intrusive acoustic phonemes are **frequently absent from the ground-truth label sequence**.
* Traditional CTC segmentation trellises are strictly constrained forced-alignment graphs: they only permit transitions among characters present in the ground-truth sequence and the blank symbol.
* When an intrusive sound (such as pre-/post-aspiration `/h/` or glottal stop `/'/`) occurs acoustically, standard CTC segmentation is forced to absorb the acoustic frames into adjacent consonants or blank symbols, losing the intrusive sound entirely.

### Objective
Extend the `ctc-segmentation` forward DP trellis and backtracking algorithms to support **linguistically-agnostic intrusive token transitions**. This allows the engine to dynamically insert 1-frame/multi-frame intrusive token detours between ground-truth states when acoustic evidence in the CTC log-probabilities matrix (`lpz`) justifies it, while guaranteeing **at most one** intrusion per transition site.

---

## 2. Architectural Design & Invariants

1. **Linguistically Agnostic:** The engine operates on token IDs and log-probabilities. It requires zero language-specific grammars or dictionaries.
2. **Single-Slot Mutual Exclusivity:** Between any ground-truth state $c$ and $c+1$, the trellis permits **at most one** intrusive token (e.g., `'h'` OR `"'"` OR direct $\epsilon$-transition), preventing ungrammatical stacking (such as `h'`).
3. **Complexity Preservation:** The time complexity remains strictly $O(T \times |S| \times |J_{\text{intrusive}}|)$, adding negligible overhead ($<5\%$) over the baseline forward pass.
4. **Backwards Compatibility:** When `intrusive_tokens=None` (default), the trellis behaves identically to baseline CTC segmentation.

---

## 3. Configuration API (`CtcSegmentationParameters`)

Add the following configuration fields to `CtcSegmentationParameters` in `ctc_segmentation/ctc_segmentation.py`:

```python
class CtcSegmentationParameters:
    # ... existing parameters ...
    
    # Intrusive Token Subsystem
    intrusive_tokens: Optional[Sequence[Union[str, int]]] = None
    intrusive_penalty: float = 0.5
    is_intrusive_token: Optional[np.ndarray] = None  # 1D boolean/int8 mask for vocab IDs
```

### Parameter Semantics
* `intrusive_tokens`: List of character strings or token IDs from `char_list` eligible for intrusive insertion (e.g. `["h", "'"]`).
* `intrusive_penalty`: Non-negative float subtracted from the log-probability of taking an intrusive detour. Higher values require stronger acoustic confidence to trigger insertion. (Recommended default: `0.5`).
* `is_intrusive_token`: 1D `int8` array of size `len(char_list)` where `is_intrusive_token[v] = 1` if vocabulary token index `v` is in `intrusive_tokens`, else `0`.

---

## 4. Mathematical Formulation (Forward DP Trellis)

Let:
* $t \in [0, T-1]$ be the audio frame time index.
* $c \in [0, |S|-1]$ be the ground-truth label sequence column index.
* $L(c)$ be the ground-truth token index at column $c$.
* $\text{lpz}[t, v] = \ln P(\text{token } v \mid \mathbf{x}_t)$ be the acoustic model log-probability at frame $t$.
* $\lambda_{\text{intrusive}} \ge 0$ be the `intrusive_penalty`.
* $J \subset \{0, \dots, V-1\}$ be the set of intrusive token vocabulary indices.
* $\text{table}[t, c]$ be the maximum joint log-probability aligning audio frames up to $t$ with ground truth up to $c$.

### Trellis Transition Topology

Between ground-truth character state $c-1$ and state $c$:

```
                        ┌─── [Intrusive State j ∈ J] ───┐
                        │    costs lpz[t-1, j]           │
                        │    - intrusive_penalty         │
                        ▼                               ▼
State c-1 (e.g. 'y') ──────────────────────────────────────> State c (e.g. 'a')
                     (Direct: table[t-1, c-1] + lpz[t, L(c)])
```

### Forward Equations in `cython_fill_table`

For each column $c > 0$ and time index $t \ge 1$:

1. **Standard Switch Probability (Direct Transition):**
   $$P_{\text{switch}}(t, c) = \text{table}[t - 1 + \text{offset}, c - 1] + \text{lpz}[t + \text{offset\_sum}, L(c)]$$

2. **Intrusive Detour Probability:**
   For each candidate intrusive token $j \in J$:
   $$P_{\text{intrusive}}(t, c, j) = \text{table}[t - 2 + \text{offset}, c - 1] + \text{lpz}[t - 1 + \text{offset\_sum}, j] - \lambda_{\text{intrusive}} + \text{lpz}[t + \text{offset\_sum}, L(c)]$$

   $$P_{\text{intrusive\_best}}(t, c) = \max_{j \in J} P_{\text{intrusive}}(t, c, j)$$

3. **Syncope Skip Probability (Existing Vowel Deletion):**
   $$P_{\text{syncope}}(t, c) = \text{table}[t - 1 + \text{offset}, c - 2] + \text{lpz}[t + \text{offset\_sum}, L(c)] - \lambda_{\text{syncope}}$$

4. **Stay Probability:**
   $$P_{\text{stay}}(t, c) = \text{table}[t - 1, c] + \max\Big(\text{lpz}[t + \text{offset\_sum}, \text{blank}],\; \text{lpz}[t + \text{offset\_sum}, L(c)]\Big)$$

5. **Cell Value:**
   $$\text{table}[t, c] = \max\Big( P_{\text{switch}}(t, c),\; P_{\text{stay}}(t, c),\; P_{\text{syncope}}(t, c),\; P_{\text{intrusive\_best}}(t, c) \Big)$$

---

## 5. Backtracking & Path Reconstruction

During backtracking from $(t, c)$ back to $(0, 0)$:

### Transition Type Identification
At current cell $(t, c)$:
1. Calculate delta residuals for:
   * $\Delta_{\text{stay}} = \big| \text{table}[t, c] - (\text{table}[t-1, c] + \text{prob}_{\text{stay}}) \big|$
   * $\Delta_{\text{switch}} = \big| \text{table}[t, c] - (\text{table}[t-1+\text{offset}, c-1] + \text{lpz}[t+\text{offset}, L(c)]) \big|$
   * $\Delta_{\text{syncope}} = \big| \text{table}[t, c] - (\text{table}[t-1+\text{offset}, c-2] + \text{lpz}[t+\text{offset}, L(c)] - \lambda_{\text{syncope}}) \big|$
   * $\Delta_{\text{intrusive}}(j) = \big| \text{table}[t, c] - (\text{table}[t-2+\text{offset}, c-1] + \text{lpz}[t-1+\text{offset}, j] - \lambda_{\text{intrusive}} + \text{lpz}[t+\text{offset}, L(c)]) \big|$

2. **If Intrusive Detour was selected ($\min \Delta = \Delta_{\text{intrusive}}(j^*)$):**
   * Record ground-truth token $L(c)$ at frame $t$:
     $$\text{state\_list}[t + \text{offset}] = \text{char\_list}[L(c)]$$
     $$\text{char\_probs}[t + \text{offset}] = \text{lpz}[t + \text{offset}, L(c)]$$
     $$\text{timings}[c] = (t + \text{offset}) \times \text{index\_duration}$$
   * **Record intrusive token $j^*$ at frame $t-1$:**
     $$\text{state\_list}[t - 1 + \text{offset}] = \text{char\_list}[j^*]$$
     $$\text{char\_probs}[t - 1 + \text{offset}] = \text{lpz}[t - 1 + \text{offset}, j^*]$$
   * Step indices back:
     $$c \leftarrow c - 1$$
     $$t \leftarrow t - 2$$

---

## 6. Cython Implementation Details (`ctc_segmentation_dyn.pyx`)

### Function Signature Update

```cython
def cython_fill_table(np.ndarray[np.float32_t, ndim=2] table,
                      np.ndarray[np.float32_t, ndim=2] lpz,
                      np.ndarray[np.int64_t, ndim=2] ground_truth,
                      np.ndarray[np.int64_t, ndim=1] offsets,
                      np.ndarray[np.int8_t, ndim=1] is_syncope_token,
                      float syncope_penalty,
                      np.ndarray[np.int64_t, ndim=1] intrusive_token_ids,
                      int num_intrusive_tokens,
                      float intrusive_penalty,
                      int blank,
                      int flags):
```

### Inner Loop Optimization
```cython
# Inside Cython time step loop:
intrusive_skip_prob = prob_max
if num_intrusive_tokens > 0 and c >= 1 and t >= 2:
    for j_idx in range(num_intrusive_tokens):
        j = intrusive_token_ids[j_idx]
        delta_offset = offset_sum - offsets[c - 1]
        t_prev = t - 2 + delta_offset
        if 0 <= t_prev < table.shape[0]:
            p_cand = table[t_prev, c - 1] + lpz[t - 1 + offset_sum, j] - intrusive_penalty + lpz[t + offset_sum, ground_truth[c, 0]]
            if p_cand > intrusive_skip_prob:
                intrusive_skip_prob = p_cand

table[t, c] = max(max(max(switch_prob, stay_prob), syncope_skip_prob), intrusive_skip_prob)
```

---

## 7. Test Suite & Verification Matrix

### Unit Tests (`tests/test_intrusive_trellis.py`)

| Test Case | Ground Truth Input | Spoken Audio / LPZ Features | Expected Extracted Path |
| :--- | :--- | :--- | :--- |
| `test_intrusive_h_inserted` | `akeya` | High acoustic energy for `h` between `y` and `a` | `akeyha` |
| `test_unvoiced_h_not_inserted` | `akeya` | Silence/vowel acoustic energy between `y` and `a` | `akeya` |
| `test_intrusive_glottal_stop` | `tsanvsv` | High acoustic energy for `'` between `a` and `n` | `tsa'nvsv` |
| `test_mutual_exclusivity` | `akeya` | Competing acoustic energy for both `h` and `'` | `akeyha` OR `akey'a` (Never `akeyh'a`) |
| `test_combined_syncope_and_intrusion` | `adalenisgv` | Vowel `a` dropped, coda `h` inserted before `s` | `adalenihskv` |

---

## 8. Summary of Benefits

1. **True Reconciled Phonetics:** Eliminates the need for post-hoc heuristic regexes or secondary string reconciliation passes. The backtracked CTC path **is** the reconciled phonetic transcription.
2. **Robust to Imperfect Transcripts:** Canonical transliterations lacking underlying phonological markers (`h`, `'`) are automatically enriched directly from the speech audio.
3. **Purely Data-Driven & Generic:** Works across any CTC acoustic model and language simply by supplying the token IDs of intrusive candidates.

# Technical Specification: Blank-Constrained Syncope Transitions for `ctc-segmentation`

**Version:** 2.1  
**Target Package:** `ctc-segmentation` (`ctc_segmentation_dyn.pyx`, `ctc_segmentation.py`)  
**Status:** Ready for Implementation  

---

## 1. Executive Summary & Problem Analysis

### 1.1 Background
The `ctc-segmentation` engine supports **syncope transitions**, allowing phonological vowel omission (e.g., deletion of unstressed vowels `/a, e, i, o, u, v/` in fast or connected speech) without failing the dynamic programming trellis alignment.

In the trellis diagram, a syncope transition allows the DP path to skip past an optional vowel token (`is_syncope_token == 1`) by jumping over the column corresponding to that vowel, incurring a calibrated `syncope_penalty`.

### 1.2 The Collateral Consonant Deletion Bug
In the current implementation of `cython_fill_table` (`ctc_segmentation_dyn.pyx`) and Python backtracking (`ctc_segmentation.py`), syncope skip transitions are evaluated using a broad window:

```cython
# Current implementation in ctc_segmentation_dyn.pyx
if is_syncope_token[c - 1] == 1:
    for c_prev in range(max(0, c - 3), c - 1):
        # Evaluates both c_prev = c - 2 (1-step skip) AND c_prev = c - 3 (2-step skip)
        ...
```

The jump from `c_prev = c - 3` was originally intended to handle representations where inter-character blanks (`[PAD]`) separate every letter (e.g., `Consonant - [PAD] - Vowel - [PAD] - Consonant`).

However, when text is prepared with **contiguous character packing** (`replace_spaces_with_blanks=False`), character tokens inside a word are stored consecutively without intervening blanks:

```text
Column Index:    c=13      c=14      c=15      c=16      c=17      c=18      c=19      c=20
Token:          [PAD]        y         i         h         s         t         v      [PAD]
Syncope Token:    0          0         1         0         0         0         1        0
```

### 1.3 Case Study: Accidental Consonant Dropping in Verse 020101 (`yihstv`)
In Bible verse 020101 (Mark 1:1), the printed ground truth text contains a typographical error: **`ᏱᏍᏛ` (`yihstv`)** was printed instead of **`ᎣᏍᏛ` (`ohstv`)**. The spoken audio contains `ohsta`.

When aligning state $c = 16$ (`h`):
1. The aligner inspects column $c - 1 = 15$ (`i`), observing `is_syncope_token[15] == 1`.
2. The loop iterates over `c_prev \in [13, 14]`.
3. When evaluating **`c_prev = 13` (`[PAD]`)**, the trellis creates a direct jump from $c = 13$ (`[PAD]`) to $c = 16$ (`h`):
   $$[\text{PAD}]_{c=13} \xrightarrow{\text{syncope jump}} [\text{h}]_{c=16}$$
4. **Result**: Both the consonant `y` ($c=14$) and the vowel `i` ($c=15$) are bypassed together in a single transition for only the cost of one vowel syncope penalty ($2.0$).
5. The remaining characters `h-s-t-v` then align against the audio for `o-h-s-t-a`, achieving an artificially high confidence ($28.2\%$) and completely evading typo/anomaly detection.

---

## 2. Mathematical Formulation & Invariants

### 2.1 Invariants
1. **Consonant Preservation Invariant**: A syncope transition MUST NOT omit any character token whose `is_syncope_token == 0`.
2. **Blank-Only Stride Invariant**: A 2-column trellis jump ($c - 3 \to c$) over a syncope vowel at $c - 1$ is valid **if and only if** the intermediate column $c - 2$ is a CTC blank / PAD token ($L(c - 2) = \text{blank}$ or $L(c - 2) = -1$).

### 2.2 Updated Forward Trellis Recurrence

Let:
* $t \in [0, T-1]$ be the time frame index.
* $c \in [0, |S|-1]$ be the ground-truth column index.
* $L(c, s)$ be the token ID at column $c$, multi-label slot $s$.
* $\text{is\_syncope\_token}[c] \in \{0, 1\}$ indicate if column $c$ is an eligible syncope token.
* $\lambda_{\text{syncope}} \ge 0$ be the syncope penalty.
* $\text{blank}$ be the CTC blank / PAD token ID.

For any state arrival $(t, c)$ with $L(c, s) \ne -1$:

#### Case A: Immediate Syncope Vowel ($c - 1$ is syncope token)
1. **Direct 1-step character skip ($c - 2 \to c$):**
   $$P_{\text{sync}, 1}(t, c) = \text{table}[t - 1 + \text{offset}(c, c - 2), c - 2] + \text{lpz}[t + \text{offsets}[c], L(c, s)] - \lambda_{\text{syncope}}$$

2. **Blank-mediated 2-step skip ($c - 3 \to c$):**  
   *Permitted ONLY IF $c \ge 3$ AND $L(c - 2, 0) \in \{\text{blank}, -1\}$:*
   $$P_{\text{sync}, 2}(t, c) = \begin{cases}
   \text{table}[t - 1 + \text{offset}(c, c - 3), c - 3] + \text{lpz}[t + \text{offsets}[c], L(c, s)] - \lambda_{\text{syncope}} & \text{if } L(c - 2, 0) \in \{\text{blank}, -1\} \\
   -\infty & \text{otherwise}
   \end{cases}$$

#### Case B: Space/Blank Following Syncope Vowel ($c - 2$ is syncope token and $c - 1$ is blank)
When an optional vowel is at a word boundary followed by a space/blank column ($c - 1$):
1. **Skip vowel + trailing blank ($c - 3 \to c$):**
   $$P_{\text{sync}, 3}(t, c) = \text{table}[t - 1 + \text{offset}(c, c - 3), c - 3] + \text{lpz}[t + \text{offsets}[c], L(c, s)] - \lambda_{\text{syncope}}$$

2. **Skip leading blank + vowel + trailing blank ($c - 4 \to c$):**  
   *Permitted ONLY IF $c \ge 4$ AND $L(c - 3, 0) \in \{\text{blank}, -1\}$:*
   $$P_{\text{sync}, 4}(t, c) = \begin{cases}
   \text{table}[t - 1 + \text{offset}(c, c - 4), c - 4] + \text{lpz}[t + \text{offsets}[c], L(c, s)] - \lambda_{\text{syncope}} & \text{if } L(c - 3, 0) \in \{\text{blank}, -1\} \\
   -\infty & \text{otherwise}
   \end{cases}$$

---

## 3. Cython Implementation (`ctc_segmentation_dyn.pyx`)

Replace the unconstrained loop in `cython_fill_table`:

```cython
# -------------------------------------------------------------------------
# Compute syncope skip probability with blank-constrained stride
# -------------------------------------------------------------------------
syncope_skip_prob = prob_max
if c >= 2:
    for s in range(ground_truth.shape[1]):
        if ground_truth[c, s] != -1:
            # Case A: c - 1 is a syncope token
            if is_syncope_token[c - 1] == 1:
                # 1. Direct 1-token skip over syncope vowel (c - 2 -> c)
                delta_offset = offset_sum - offsets[c - 2]
                t_prev = t - 1 + delta_offset
                if 0 <= t_prev < table.shape[0]:
                    v_prob = table[t_prev, c - 2] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                    if v_prob > syncope_skip_prob:
                        syncope_skip_prob = v_prob

                # 2. 2-token skip (c - 3 -> c) ONLY IF c - 2 is a blank/PAD
                if c >= 3 and (ground_truth[c - 2, 0] == blank or ground_truth[c - 2, 0] == -1):
                    delta_offset = offset_sum - offsets[c - 3]
                    t_prev = t - 1 + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                        if v_prob > syncope_skip_prob:
                            syncope_skip_prob = v_prob

            # Case B: c - 2 is syncope token and c - 1 is a blank/space
            elif c >= 3 and is_syncope_token[c - 2] == 1 and (ground_truth[c - 1, 0] == blank or ground_truth[c - 1, 0] == -1):
                # 1. Skip vowel + trailing blank (c - 3 -> c)
                delta_offset = offset_sum - offsets[c - 3]
                t_prev = t - 1 + delta_offset
                if 0 <= t_prev < table.shape[0]:
                    v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                    if v_prob > syncope_skip_prob:
                        syncope_skip_prob = v_prob

                # 2. Skip leading blank + vowel + trailing blank (c - 4 -> c) ONLY IF c - 3 is blank
                if c >= 4 and (ground_truth[c - 3, 0] == blank or ground_truth[c - 3, 0] == -1):
                    delta_offset = offset_sum - offsets[c - 4]
                    t_prev = t - 1 + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        v_prob = table[t_prev, c - 4] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                        if v_prob > syncope_skip_prob:
                            syncope_skip_prob = v_prob
```

---

## 4. Python Backtracking Implementation (`ctc_segmentation.py`)

Mirrored logic in `ctc_segmentation()` backtracking step:

```python
# Check syncope skip transitions with blank constraints
min_syncope_skip_delta = np.inf
best_syncope_c_prev = None
best_syncope_s = None
if c >= 2:
    for s in range(ground_truth.shape[1]):
        if ground_truth[c, s] != -1:
            if is_syncope_token_arr[c - 1] == 1:
                # Valid candidate predecessors
                candidates = [c - 2]
                if c >= 3 and (ground_truth[c - 2, 0] == blank or ground_truth[c - 2, 0] == -1):
                    candidates.append(c - 3)
                
                for c_prev in candidates:
                    delta_offset = offsets[c] - offsets[c_prev]
                    t_prev = t - 1 + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                        expected_v_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            - syncope_penalty
                        )
                        v_delta = abs(est_v_prob - expected_v_prob)
                        if v_delta < min_syncope_skip_delta:
                            min_syncope_skip_delta = v_delta
                            best_syncope_c_prev = c_prev
                            best_syncope_s = s
            elif (
                c >= 3
                and is_syncope_token_arr[c - 2] == 1
                and (
                    ground_truth[c - 1, 0] == blank
                    or ground_truth[c - 1, 0] == -1
                )
            ):
                candidates = [c - 3]
                if c >= 4 and (ground_truth[c - 3, 0] == blank or ground_truth[c - 3, 0] == -1):
                    candidates.append(c - 4)

                for c_prev in candidates:
                    delta_offset = offsets[c] - offsets[c_prev]
                    t_prev = t - 1 + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        est_v_prob = table[t, c] - table[t_prev, c_prev]
                        expected_v_prob = (
                            lpz[t + offsets[c], ground_truth[c, s]]
                            - syncope_penalty
                        )
                        v_delta = abs(est_v_prob - expected_v_prob)
                        if v_delta < min_syncope_skip_delta:
                            min_syncope_skip_delta = v_delta
                            best_syncope_c_prev = c_prev
                            best_syncope_s = s
```

---

## 5. Verification & Acceptance Criteria

### 5.1 Unit Tests
1. **Contiguous Character Word with Syncope**:
   - Word: `atalenihskv` (`a-t-a-l-e-n-i-h-s-k-v`).
   - Spoken: `atalenhskv` (dropped `i`).
   - Expectation: `i` ($c=8$) is skipped via $c=7 \to c=9$; `n` ($c=7$) and `h` ($c=9$) remain precisely aligned.
2. **Typo / Consonant Protection**:
   - Word: `yihstv` (`[PAD]-y-i-h-s-t-v-[PAD]`).
   - Spoken: `ohsta`.
   - Expectation: Jump from `[PAD]` ($c=13$) directly to `h` ($c=16$) is **rejected** because $c=14$ (`y`) is not a blank. The word cannot skip `y`, causing alignment failure or near-zero confidence ($\text{conf} < 0.01$), successfully setting `flagged=True`.
3. **Inter-Character Blanks Compatibility**:
   - Ground truth with `replace_spaces_with_blanks=True` (`C1 - [PAD] - V - [PAD] - C2`) continues to allow skipping `[PAD] - V` seamlessly because intermediate states are blanks.

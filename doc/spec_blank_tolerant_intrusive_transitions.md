# Technical Specification: Blank-Tolerant Intrusive Token Transitions for `ctc-segmentation`

**Version:** 2.0  
**Target Package:** `ctc-segmentation` (`ctc_segmentation_dyn.pyx`, `ctc_segmentation.py`)  
**Status:** Ready for Implementation  

---

## 1. Executive Summary & Problem Analysis

In v1.0 of the intrusive token transition subsystem, intrusive detours (e.g. dynamic insertion of aspiration `/h/` or glottal stop `/'/`) were modeled as an immediate, rigid 2-frame transition:
$$t_{\text{prev}} = t - 2 \longrightarrow t_{\text{intrusive}} = t - 1 \longrightarrow t_{\text{next}} = t$$

### The CTC Blank Gap Problem
CTC acoustic models emit spikes/peaks separated by blank (`[PAD]` / $\epsilon$) frames. In natural continuous speech:
```text
Frame 23:  'l'  (-0.04 logprob)  <-- Ground truth state c-1
Frame 24:  'h'  (-0.01 logprob)  <-- Intrusive token peak
Frame 25:  [PAD]                 <-- Intervening CTC blank
Frame 26:  [PAD]                 <-- Intervening CTC blank
Frame 27:  'u'  (-0.02 logprob)  <-- Ground truth state c
```

When evaluating the forward trellis at state $c$ (`u`) at frame $t = 27$:
* The v1.0 trellis strictly evaluated $t-1 = 26$ as the intrusive frame.
* Frame 26 has $P(\text{h}) \approx 0$ because it is a blank frame (`[PAD]`).
* Consequently, the trellis **missed the intrusive `/h/` at frame 24**, discarding valid phonetic surface realizations (e.g., failing to emit `uwelhuhka` from citation `uweluka`).

---

## 2. Mathematical Formulation

Let:
* $t \in [0, T-1]$ be the time frame index.
* $c \in [0, |S|-1]$ be the ground-truth character column index.
* $L(c)$ be the vocabulary token ID of ground-truth state $c$.
* $J \subset \{0, \dots, V-1\}$ be the set of intrusive token IDs (e.g., `['h', "'"]`).
* $\lambda_{\text{intrusive}} \ge 0$ be `intrusive_penalty`.
* $W \ge 1$ be `intrusive_max_stride` (maximum blank frame stride, default: $4$ frames / $80\text{ms}$).
* $\text{blank}$ be the CTC blank / pad token ID.
* $\text{lpz}[t, v]$ be the log-probability of token $v$ at time $t$.

### 2.1 Forward Trellis Recurrence

To allow intervening blanks between state $c-1$, the intrusive token $j$, and state $c$:

For any arrival at state $c$ at frame $t$, we allow an intrusive detour that:
1. Transitions from state $c-1$ at frame $t - 1 - \delta_1 - \delta_2$ (where $\delta_1, \delta_2 \ge 0, \delta_1 + \delta_2 \le W$).
2. Traverses $\delta_1$ optional blank frames.
3. Emits intrusive token $j \in J$ at frame $t_{\text{intrusive}} = t - 1 - \delta_2$.
4. Traverses $\delta_2$ optional blank frames.
5. Emits ground-truth token $L(c)$ at frame $t$.

The optimal intrusive detour score arriving at $(t, c)$ is:
$$P_{\text{intrusive}}(t, c) = \max_{j \in J} \max_{\substack{\delta_1, \delta_2 \ge 0 \\ \delta_1 + \delta_2 \le W}} \left[ \begin{aligned}
&\text{table}[t - 2 - \delta_1 - \delta_2 + \text{offset}, c - 1] \\
&+ \sum_{k=1}^{\delta_1} \text{lpz}[t - 1 - \delta_2 - k + \text{offset\_sum}, \text{blank}] \\
&+ \text{lpz}[t - 1 - \delta_2 + \text{offset\_sum}, j] - \lambda_{\text{intrusive}} \\
&+ \sum_{m=1}^{\delta_2} \text{lpz}[t - m + \text{offset\_sum}, \text{blank}] \\
&+ \text{lpz}[t + \text{offset\_sum}, L(c)]
\end{aligned} \right]$$

### 2.2 Efficient 1D Blank-Accumulation Optimization

Because $\text{lpz}[\tau, \text{blank}] \approx 0$ during blank regions, the score can be simplified with cumulative blank log-probabilities or a compact search over the intrusive frame offset $\delta \in [0, W]$:

$$P_{\text{intrusive}}(t, c) = \max_{j \in J} \max_{0 \le \delta \le W} \left( \begin{aligned}
&\text{table}[t - 2 - \delta + \text{offset}, c - 1] \\
&+ \text{lpz}[t - 1 - \delta + \text{offset\_sum}, j] - \lambda_{\text{intrusive}} \\
&+ \text{blank\_score}(t - \delta, t - 1) \\
&+ \text{lpz}[t + \text{offset\_sum}, L(c)]
\end{aligned} \right)$$

---

## 3. Cython Implementation (`ctc_segmentation_dyn.pyx`)

In `cython_fill_table`:

```cython
cdef int max_intrusive_stride = min(5, table.shape[0])
cdef double blank_sum, p_cand
cdef int delta, j_idx, j, s, t_prev

# Compute intrusive detour probability with blank-stride tolerance
intrusive_prob = prob_max
if num_intrusive_tokens > 0 and c >= 1 and t >= 2:
    for j_idx in range(num_intrusive_tokens):
        j = intrusive_token_ids[j_idx]
        for s in range(ground_truth.shape[1]):
            if ground_truth[c, s] != -1:
                delta_offset = offset_sum - (offsets[c - 1 - s] if c - 1 - s >= 0 else 0)
                for delta in range(0, min(max_intrusive_stride, t - 1)):
                    t_prev = t - 2 - delta + delta_offset
                    if 0 <= t_prev < table.shape[0]:
                        # Optional: accumulate blank logprobs between t - 1 - delta and t
                        blank_sum = 0.0
                        for b_t in range(t - delta, t):
                            blank_sum += lpz[b_t + offset_sum, blank]
                        
                        p_cand = (
                            table[t_prev, c - 1 - s]
                            + lpz[t - 1 - delta + offset_sum, j]
                            - intrusive_penalty
                            + blank_sum
                            + lpz[t + offset_sum, ground_truth[c, s]]
                        )
                        if p_cand > intrusive_prob:
                            intrusive_prob = p_cand
```

---

## 4. Backtracking & State List Emission (`ctc_segmentation.py`)

During reverse dynamic programming traversal:
1. When an intrusive detour is selected at $(t, c)$:
   * Identify the winning intrusive token $j^*$ and frame offset $\delta^*$.
   * Emit ground truth character $L(c)$ at frame $t$: `state_list[offsets[c] + t] = char_list[L(c)]`.
   * Fill intervening blank frames: `state_list[offsets[c] + b_t] = "[PAD]"` for $b_t \in [t - \delta^*, t - 1]$.
   * Emit intrusive character $j^*$ at frame $t - 1 - \delta^*$: `state_list[offsets[c] + t - 1 - \delta^*] = char_list[j^*]`.
   * Step backwards: $c \leftarrow c - 1 - s$, $t \leftarrow t - (2 + \delta^*) + \text{offset}$.

---

## 5. Configuration & Recommended Defaults

In `CtcSegmentationParameters`:
* `syncope_tokens = ["a", "e", "i", "o", "u", "v"]`
* `syncope_penalty = 2.0` (prevents over-deletion of legitimate vowels while allowing true acoustic drops)
* `intrusive_tokens = ["h", "'"]`
* `intrusive_penalty = 0.1` (low threshold for inserting acoustic aspirations and glottal stops)
* `intrusive_max_stride = 4` (allows up to 4 blank frames / $80\text{ms}$ between phoneme peaks)

---

## 6. Verification Criteria

* Citation `uweluka` (Mark 1:3) aligned against spoken audio emits **`uwelhuhka`** with $t_{\text{dur}} \approx 0.62\text{s}$ and confidence $> 70\%$.
* Citation `hi-a` / `hia` (Mark 1:7) aligned against spoken audio emits **`hi'a`** with glottal stop.
* Benchmarked across 100 verses without regression in latency or accuracy.

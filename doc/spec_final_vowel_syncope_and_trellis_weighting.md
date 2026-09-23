# Specification: Word- and Phrase-Final Vowel Syncope Trellis Reform (Zero Hyperparameters)

**Status:** Implemented  
**Target Subsystems:** `ctc_segmentation_dyn.pyx`, `ctc_segmentation.py`, `partitioning.py`  
**Related PRs / Specs:** `doc/spec_no_hyperparams.md` (Relative Contrastive Acoustic Gating)

---

## 1. Context & Motivation

In conversational and connected speech, vowel syncope frequently occurs not only word-medially ($C_1 V C_2 \rightarrow C_1 C_2$), but also at **word boundaries** ($C_{\text{prev}} V_{\text{final}} \text{ [PAD] } C_{\text{next}} \rightarrow C_{\text{prev}} C_{\text{next}}$) and **phrase terminals** ($C_{\text{prev}} V_{\text{final}} \text{ [PAD\_end]} \rightarrow C_{\text{prev}} \text{ [PAD\_end]}$).

However, in benchmark evaluations on the relative contrastive gating trellis, final vowel syncope caused **26.94% of all character alignment errors** (929 spurious vowel deletions). When analyzing the dynamic programming path, two root causes emerge:

1. **The "Blank Anchor Trap"**:
   The relative contrastive gate is defined as:
   $$\text{Gate}(t, c) = \begin{cases} \text{OPEN}, & \text{if } \text{lpz}[t, \text{anchor}] > \text{lpz}[t, V] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$
   When $V_{\text{final}}$ is at a word or phrase boundary, the immediate subsequent token in ground truth is `PAD` (blank, token index `0`). In any pause, inter-word gap, or trailing silence, $\text{lpz}[t, \text{blank}] \approx 0.0$ while $\text{lpz}[t, V] \approx -10.0$. Thus, **the gate is trivially and unconditionally OPEN at every single silent frame**, even when the speaker articulated the vowel clearly.
2. **Consonant State Blank-Riding**:
   In standard CTC segmentation, `stay_prob` on character states uses:
   $$\text{stay\_prob}(C, t) = \text{table}[t-1, C] + \max(\text{lpz}[t, \text{blank}], \text{lpz}[t, C])$$
   This allows a consonant state $C_{\text{prev}}$ to linger across subsequent vowel frames with minimal penalty, after which the path executes a single-frame skip into `PAD` when silence starts, completely bypassing $V_{\text{final}}$.

This specification models final vowel syncope so that it is **naturally and acoustically weighted** without adding any tuning hyperparameters.

---

## 2. Mathematical Formulation & Structural Invariants

### A. Subsystem 1: Inter-Word Final Syncope ($C_{\text{prev}} V_{\text{final}} \text{ [PAD] } C_{\text{next}}$)

When a word-final vowel is dropped before the next word, the speaker transitions directly from $C_{\text{prev}}$ to the next word's onset $C_{\text{next}}$ (with optional inter-word acoustic silence). 

**The Invariant:** The non-acoustic `PAD` token must **never** serve as the contrastive gating anchor. The true acoustic anchor is the **onset consonant of the following word ($C_{\text{next}}$)**.

```
Ground Truth:  ... [C_prev]  [V_final]  [PAD]  [C_next] ...
                     |            \        /       |
                     +------------( SKIP )-------->+  (Gated against C_next)
```

#### Mathematical Transition:
1. **Target Anchor**: $A = C_{\text{next}} = \text{ground\_truth}[c, s]$ where $c$ is the column of $C_{\text{next}}$.
2. **Skipped Tokens**: $V_{\text{final}} = \text{ground\_truth}[c - 2, 0]$ (or $c - 3$ if leading blank exists) and intermediate inter-word `PAD` at $c - 1$.
3. **Contrastive Arrival Gate**:
   $$\text{Gate}_{\text{word\_final}}(t, c) = \begin{cases} \text{OPEN}, & \text{if } \text{lpz}[t + \text{offset}, C_{\text{next}}] > \text{lpz}[t + \text{offset}, V_{\text{final}}] \\ \text{CLOSED}, & \text{otherwise} \end{cases}$$
4. **Transition Probability (Case B)**:
   $$P_{\text{sync\_interword}}(t, c) = \text{table}[t - 1 + \Delta_{\text{offset}}, c - 3] + \text{lpz}[t + \text{offset}, C_{\text{next}}]$$
   *(With blank stride accumulation if inter-word silence intervened before $C_{\text{next}}$).*

---

### B. Subsystem 2: Phrase/Utterance-Terminal Syncope ($C_{\text{prev}} V_{\text{final}} \text{ [PAD\_end]}$)

At the end of an utterance, there is no subsequent word anchor $C_{\text{next}}$—only terminal silence (`PAD`).

To prevent the "Blank Anchor Trap" from erasing spoken terminal vowels:

1. **Immediate Acoustic Boundary Condition**:
   - The transition from $C_{\text{prev}}$ across $V_{\text{final}}$ into `PAD` must be anchored to the frame $t_{\text{end}}$ where $C_{\text{prev}}$ actually ends ($t_{\text{prev}} = t - 1$).
   - $C_{\text{prev}}$ is not permitted to stay active across frames where $C_{\text{prev}}$ has no acoustic support.
2. **Pure Likelihood Competition**:
   With tight duration transitions, the dynamic programming trellis compares two paths:
   - **Path A (Vowel Pronounced)**: Transitions $C_{\text{prev}} \to V_{\text{final}} \to \text{PAD}$.
     Accumulates $\sum_{t \in \text{vowel}} \text{lpz}[t, V_{\text{final}}] + \sum_{t \in \text{silence}} \text{lpz}[t, \text{blank}]$.
   - **Path B (Vowel Dropped)**: Transitions $C_{\text{prev}} \to \text{PAD}$ directly.
     Accumulates $\sum_{t \in \text{vowel}} \text{lpz}[t, \text{blank}] + \sum_{t \in \text{silence}} \text{lpz}[t, \text{blank}]$.
   - **Result**:
     - When vowel formants are present: In Path B, $\text{lpz}[t, \text{blank}]$ is strongly negative ($\le -8.0$), making Path A win easily.
     - When vowel was deleted: In Path A, $\text{lpz}[t, V_{\text{final}}]$ in silence is strongly negative ($\le -10.0$), making Path B win naturally.

---

## 3. Implementation Tasks

### Task 1: Trellis Anchor Resolution (`ctc_segmentation_dyn.pyx`)
- Refactor the syncope candidate selector in `cython_fill_table`:
  - When column $c$ is a `blank` (`ground_truth[c, 0] == blank`):
    - Do **NOT** open Case A or Case B skip transitions if the skip is gated against `blank` during an inter-word boundary.
    - Instead, defer the syncope skip evaluation until column $c$ advances to the next phoneme anchor $C_{\text{next}}$, skipping both $V_{\text{final}}$ and the intervening `PAD`.
  - For true phrase-terminal `PAD` (where $c = \text{table.shape}[1] - 1$):
    - Enforce immediate offset boundary ($t_{\text{prev}} = t - 1$) and evaluate direct likelihood competition without intermediate blank-farming.

### Task 2: Python Trellis & Backtracking Alignment (`ctc_segmentation.py`)
- Mirror the Cython anchor resolution in Python `fill_table`:
  - Ensure backtracking pointers correctly trace:
    - $C_{\text{next}} \leftarrow C_{\text{prev}}$ for inter-word syncope skips across $(V_{\text{final}} + \text{PAD})$.
    - $\text{PAD\_end} \leftarrow C_{\text{prev}}$ for phrase-terminal skips.

### Task 3: Ground Truth Syncope Masking (`partitioning.py` / `prepare_token_list`)
- Ensure `is_syncope_token` marks word-final vowels as syncope-eligible while setting the structural anchor to the following word's initial token.

### Task 4: Test Suite & Verification Matrix (`tests/`)
- Add unit tests verifying:
  1. **Inter-word syncope pronounced**: Audio containing `...ka no he tv tsi sa...` preserves `tv` when $C_{\text{next}} = \text{ts}$ is preceded by vowel formants.
  2. **Inter-word syncope dropped**: Audio containing `...ka no he t tsi sa...` cleanly skips `v` and aligns `t` directly to `ts`.
  3. **Phrase-terminal vowel pronounced**: Terminal vowel is preserved in the presence of vowel formants before trailing silence.
  4. **Phrase-terminal vowel dropped**: Terminal vowel is dropped when silence follows immediately after the consonant.

---

## 4. Acceptance Criteria

- [x] `blank` is never used as a contrastive gating anchor for inter-word syncope.
- [x] Inter-word syncope evaluates contrastive arrival directly against the following word's onset consonant $C_{\text{next}}$.
- [x] Spurious terminal vowel truncation into trailing silence is eliminated without introducing any threshold or penalty hyperparameters.
- [x] 100% of test suite passes (`pytest`).

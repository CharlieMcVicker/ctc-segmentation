**Feature: Token Class Probability Pooling for Contrastive Syncope Gating**

### Summary

Prevent spurious syncope transitions and alignment misrouting caused by ASR vowel misclassification by pooling emission probabilities across token classes (e.g., vocalic sets) exclusively for relative contrastive syncope gating, while preserving canonical ground-truth path transitions and low-confidence scoring during trellis evaluation and backtracking.

---

### Problem Context

Currently, the relative contrastive syncope gate evaluates whether frame $t$ exhibits higher acoustic evidence for the incoming anchor token than for the canonical skipped token:


$$\text{lpz}[t, \text{anchor}] > \text{lpz}[t, \text{canonical\_token}]$$

When the ASR acoustic model misclassifies a vowel (e.g., predicts $/u/$ where the citation text has $/o/$), $\text{lpz}[t, \text{canonical\_token}]$ drops significantly. Consequently:

1. The contrastive gate erroneously opens because the incoming anchor easily outscores the near-zero posterior of the target vowel.


2. The aligner takes an unintended syncope skip ($\epsilon$-skip), deleting a segment that was acoustically present.


3. If forced to retain the character, confidence drops to extreme outliers without an explicit acoustic gating threshold.

---

### Proposed Solution

Decouple the **acoustic gating evidence** from the **trellis transition cost**:

1. **Pre-DP Class Pooling:** Ahead of the core Cython DP pass, compute an additive pooled log-posterior row (using $\text{LogSumExp}$ over configured class members) and append it to the contiguous runtime posterior matrix.
2. **Index Remapping Table:** Provide a 1D mapping array (`syncope_gate_token_map`) of size `len(vocab)` that maps token IDs belonging to a syncope class to their corresponding pooled row index, passing unmapped tokens through via identity mapping.
3. **Contrastive Gating Target:** During syncope evaluation in `cython_fill_table`, query the mapped index in `lpz` for gating evidence:

$$\text{Gate}_{\text{syncope}}(t, c) = \left( \text{lpz}[t_{\text{anchor}}, L(c, s)] > \text{lpz}[t_{\text{anchor}}, \text{syncope\_gate\_token\_map}[L(c_{\text{vowel}}, 0)]] \right)$$



4. **Canonical Path Preservation:** When the gate is closed (sufficient vocalic energy is present), the DP path must proceed through the ground-truth token index $L(c, s)$ with its original log probability. The state list and character confidence scoring must record the actual ground-truth token, allowing character-level confidence to reflect the acoustic substitution.



---

### Key Changes & Architecture

* **`CtcSegmentationParameters` (`ctc_segmentation/ctc_segmentation.py`)**:


* Add configuration parameter `syncope_token_classes: Optional[List[List[Union[str, int]]]] = None`.
* Update `prepare_token_list`, `prepare_text`, and runtime initialization to map class token strings to vocabulary IDs.




* **`TrellisRuntimeContext` & Matrix Preparation (`ctc_segmentation/ctc_segmentation.py`)**:


* Calculate pooled rows for `probs` / `lpz` across class token indices via $\text{LogSumExp}$ vectorized across time frames.
* Append pooled rows to `lpz`, ensuring the matrix remains C-contiguous.
* Construct a 1D `int64` array `syncope_gate_token_map` of length `vocab_size` mapping token $i \mapsto \text{pooled\_row\_index}$ (or $i \mapsto i$ for identity/unclassed tokens).


* **Cython DP Engine (`ctc_segmentation/ctc_segmentation_dyn.pyx`)**:


* Accept `syncope_gate_token_map` in `cython_fill_table` and `backtrack`.


* Update syncope gating checks to resolve vowel index through `syncope_gate_token_map[c_vowel]` before querying `lpz`.
* Ensure backtracking and state list emission retain the canonical token ID from `ground_truth_mat` without modifying emission labels.





---

### Acceptance Criteria

* [ ] `syncope_token_classes` parameter is exposed in `CtcSegmentationParameters` and validated during initialization.


* [ ] `LogSumExp` pooling over classes correctly appends to `lpz` while maintaining C-contiguity without memory leaks.
* [ ] Contrastive syncope gate checks the pooled token probability while the non-skipped path consumes the canonical token log-posterior.


* [ ] Backtracking emits canonical token IDs into `state_list`, preserving character confidence degradation on acoustic substitutions.


* [ ] Unit tests verify:
* Vowel substitution (e.g., spoken $/u/$ for written $/o/$) does not trigger spurious syncope if total vowel class probability exceeds the anchor.
* Absence of vocalic energy still allows valid syncope skips as before.


* Identity passthrough holds for unclassed tokens.


* [ ] PR target is configured to fork repository `CharlieMcVicker/ctc-segmentation` on branch `dev`.
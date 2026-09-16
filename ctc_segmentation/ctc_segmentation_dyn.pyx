#!/usr/bin/env false
# encoding: utf-8
#cython: language_level=3

# Copyright 2020, Technische Universität München; Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""CTC segmentation.

This file is part of CTC segmentation to extract utterance alignments
within an audio file using dynamic programming.
For a description, see https://arxiv.org/abs/2007.09127
"""

import logging
import numpy as np
cimport numpy as np


def cython_fill_table(np.ndarray[np.float32_t, ndim=2] table,
                      np.ndarray[np.float32_t, ndim=2] lpz,
                      np.ndarray[np.int64_t, ndim=2] ground_truth,
                      np.ndarray[np.int64_t, ndim=1] offsets,
                      np.ndarray[np.int8_t, ndim=1] is_syncope_token,
                      np.ndarray[np.int64_t, ndim=1] intrusive_token_ids,
                      np.ndarray[np.int8_t, ndim=1] is_intrusive_site,
                      int intrusive_max_stride,
                      int blank,
                      int flags):
    """Fill the table of transition probabilities.

    Supports blank-constrained syncope skip transitions (enforcing the Consonant
    Preservation Invariant, Non-Acoustic Blank Rejection Invariant, and onset anchor
    gating for inter-word syncope and phrase-terminal likelihood competition)
    and blank-tolerant intrusive token detours with unified Relative Contrastive
    Acoustic Gating (zero hyperparameter runtime).

    :param table: table filled with maximum joint probabilities k_{t,j}
    :param lpz: character probabilities of each time frame
    :param ground_truth: label sequence
    :param offsets: window offsets per character (given as array of zeros)
    :param is_syncope_token: 1D mask array marking optional syncope token positions
        for blank-constrained syncope transitions (enforces the blank-only stride
        and non-acoustic blank rejection invariants so non-syncope characters/consonants
        are preserved).
    :param intrusive_token_ids: 1D array of token IDs eligible for intrusive insertion
    :param is_intrusive_site: 1D mask array gating allowed intrusive detour transition sites
    :param intrusive_max_stride: maximum blank frame stride for intrusive transitions
    :param blank: label ID of the blank symbol, usually 0
    :param flags: configuration options, default 0
    :return: Tuple (t, c) containing the frame index and column index of the maximum probability state in the terminal column.
    """
    cdef int c
    cdef int t
    cdef int offset = 0
    cdef float mean_offset
    cdef int offset_sum = 0
    cdef int lower_offset
    cdef int higher_offset
    cdef float switch_prob, stay_prob, skip_prob, syncope_skip_prob, intrusive_prob
    cdef float prob_floor = -1000000000
    cdef float last_max
    cdef int last_arg_max
    cdef np.ndarray[np.int64_t, ndim=1] cur_offset = np.zeros([ground_truth.shape[1]], np.int64) - 1
    cdef float max_lpz_prob
    cdef float p, v_prob, p_cand, blank_sum, base_prob
    cdef int s, delta_offset, t_prev, j_idx, j, delta, b_t, t_detour, has_candidate, t_departure
    cdef int anchor_tok, vowel_tok
    cdef int num_intrusive_tokens = intrusive_token_ids.shape[0]
    cdef int stay_transition_cost_zero
    cdef int preamble_transition_cost_zero

    # Compute the mean offset between two window positions
    mean_offset = (lpz.shape[0] - table.shape[0]) / float(table.shape[1])
    logging.debug(f"Average character duration: {mean_offset} (indices)")
    lower_offset = int(mean_offset)
    higher_offset = lower_offset + 1
    # calculation of the trellis diagram table
    table[0, 0] = 0
    for c in range(table.shape[1]):
        if c > 0:
            # Compute next window offset
            offset = min(max(0, last_arg_max - table.shape[0] // 2),
                         min(higher_offset, (lpz.shape[0] - table.shape[0]) - offset_sum))
            # Compute relative offset to previous columns
            for s in range(ground_truth.shape[1] - 1):
                cur_offset[s + 1] = cur_offset[s] + offset
            cur_offset[0] = offset
            # Apply offset and move window one step further
            offset_sum += offset
        # Log offset
        offsets[c] = offset_sum
        last_arg_max = -1
        last_max = 0
        # flag for setting stay transition cost to zero for blank
        stay_transition_cost_zero = (flags & 1) * int(np.any(ground_truth[c, :] == 0))
        # flag for zero transition cost at preamble
        preamble_transition_cost_zero = (flags & 2) * int(c==0)
        # Go through all rows of the current column
        for t in range((1 if c == 0 else 0), table.shape[0]):
            # Compute max switch probability
            switch_prob = prob_floor
            max_lpz_prob = prob_floor
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] != -1:
                    if t >= table.shape[0] - (cur_offset[s] - 1) or t - 1 + cur_offset[s] < 0 or c == 0:
                        p = prob_floor
                    else:
                        p = table[t - 1 + cur_offset[s], c - (s + 1)] + lpz[t + offset_sum, ground_truth[c, s]]
                    switch_prob = max(switch_prob, p)
                    max_lpz_prob = max(max_lpz_prob, lpz[t + offset_sum, ground_truth[c, s]])

            # Compute syncope skip probability with relative contrastive gating
            syncope_skip_prob = prob_floor
            if is_syncope_token.shape[0] > 0 and c >= 2:
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        anchor_tok = ground_truth[c, s]
                        if anchor_tok != blank:
                            # Case A: c - 1 is a syncope token
                            if is_syncope_token[c - 1] == 1:
                                vowel_tok = ground_truth[c - 1, 0]
                                # Gate: acoustics at arrival frame must favor anchor over skipped vowel
                                if lpz[t + offset_sum, anchor_tok] > lpz[t + offset_sum, vowel_tok]:
                                    # 1. Direct 1-token skip over syncope vowel (c - 2 -> c)
                                    delta_offset = offset_sum - offsets[c - 2]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        v_prob = table[t_prev, c - 2] + lpz[t + offset_sum, anchor_tok]
                                        if v_prob > syncope_skip_prob:
                                            syncope_skip_prob = v_prob

                                    # 2. 2-token skip (c - 3 -> c) ONLY IF c - 2 is a blank/PAD
                                    if c >= 3 and (ground_truth[c - 2, 0] == blank or ground_truth[c - 2, 0] == -1):
                                        delta_offset = offset_sum - offsets[c - 3]
                                        t_prev = t - 1 + delta_offset
                                        if 0 <= t_prev < table.shape[0]:
                                            v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, anchor_tok]
                                            if v_prob > syncope_skip_prob:
                                                syncope_skip_prob = v_prob

                            # Case B: c - 2 is syncope token and c - 1 is a blank/space (inter-word syncope)
                            elif c >= 3 and is_syncope_token[c - 2] == 1 and (ground_truth[c - 1, 0] == blank or ground_truth[c - 1, 0] == -1):
                                vowel_tok = ground_truth[c - 2, 0]
                                # Gate: acoustics at arrival frame must favor anchor over skipped vowel
                                if lpz[t + offset_sum, anchor_tok] > lpz[t + offset_sum, vowel_tok]:
                                    # 1. Skip vowel + trailing blank (c - 3 -> c)
                                    delta_offset = offset_sum - offsets[c - 3]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, anchor_tok]
                                        if v_prob > syncope_skip_prob:
                                            syncope_skip_prob = v_prob

                                    # 2. Skip leading blank + vowel + trailing blank (c - 4 -> c) ONLY IF c - 3 is blank
                                    if c >= 4 and (ground_truth[c - 3, 0] == blank or ground_truth[c - 3, 0] == -1):
                                        delta_offset = offset_sum - offsets[c - 4]
                                        t_prev = t - 1 + delta_offset
                                        if 0 <= t_prev < table.shape[0]:
                                            v_prob = table[t_prev, c - 4] + lpz[t + offset_sum, anchor_tok]
                                            if v_prob > syncope_skip_prob:
                                                syncope_skip_prob = v_prob

                        elif c == table.shape[1] - 1 or anchor_tok == blank:
                            # Syncope directly into blank (phrase-terminal or inter-word) with departure-boundary contrastive gating
                            if is_syncope_token[c - 1] == 1:
                                vowel_tok = ground_truth[c - 1, 0]
                                # 1. Direct skip over terminal vowel into blank (c - 2 -> c)
                                delta_offset = offset_sum - offsets[c - 2]
                                t_prev = t - 1 + delta_offset
                                if 0 <= t_prev < table.shape[0]:
                                    t_departure = t_prev + 1 + offsets[c - 2]
                                    if (
                                        0 <= t_departure < lpz.shape[0]
                                        and lpz[t_departure, blank] > lpz[t_departure, vowel_tok]
                                    ):
                                        v_prob = table[t_prev, c - 2] + lpz[t + offset_sum, anchor_tok]
                                        if v_prob > syncope_skip_prob:
                                            syncope_skip_prob = v_prob

                                # 2. 2-token skip (c - 3 -> c) ONLY IF c - 2 is blank
                                if c >= 3 and (ground_truth[c - 2, 0] == blank or ground_truth[c - 2, 0] == -1):
                                    delta_offset = offset_sum - offsets[c - 3]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        t_departure = t_prev + 1 + offsets[c - 3]
                                        if (
                                            0 <= t_departure < lpz.shape[0]
                                            and lpz[t_departure, blank] > lpz[t_departure, vowel_tok]
                                        ):
                                            v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, anchor_tok]
                                            if v_prob > syncope_skip_prob:
                                                syncope_skip_prob = v_prob

                            elif c >= 3 and (ground_truth[c - 1, 0] == blank or ground_truth[c - 1, 0] == -1) and is_syncope_token[c - 2] == 1:
                                vowel_tok = ground_truth[c - 2, 0]
                                delta_offset = offset_sum - offsets[c - 3]
                                t_prev = t - 1 + delta_offset
                                if 0 <= t_prev < table.shape[0]:
                                    t_departure = t_prev + 1 + offsets[c - 3]
                                    if (
                                        0 <= t_departure < lpz.shape[0]
                                        and lpz[t_departure, blank] > lpz[t_departure, vowel_tok]
                                    ):
                                        v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, anchor_tok]
                                        if v_prob > syncope_skip_prob:
                                            syncope_skip_prob = v_prob

                                if c >= 4 and (ground_truth[c - 3, 0] == blank or ground_truth[c - 3, 0] == -1):
                                    delta_offset = offset_sum - offsets[c - 4]
                                    t_prev = t - 1 + delta_offset
                                    if 0 <= t_prev < table.shape[0]:
                                        t_departure = t_prev + 1 + offsets[c - 4]
                                        if (
                                            0 <= t_departure < lpz.shape[0]
                                            and lpz[t_departure, blank] > lpz[t_departure, vowel_tok]
                                        ):
                                            v_prob = table[t_prev, c - 4] + lpz[t + offset_sum, anchor_tok]
                                            if v_prob > syncope_skip_prob:
                                                syncope_skip_prob = v_prob

            # Compute intrusive detour probability with blank-stride tolerance and relative contrastive gating
            intrusive_prob = prob_floor
            if (
                num_intrusive_tokens > 0
                and c >= 1
                and t >= 2
                and (is_intrusive_site.shape[0] == 0 or is_intrusive_site[c] == 1)
            ):
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] == -1 or c - 1 - s < 0:
                        continue
                    anchor_tok = ground_truth[c, s]
                    delta_offset = offset_sum - offsets[c - 1 - s]
                    for delta in range(0, min(intrusive_max_stride + 1, t - 1)):
                        t_prev = t - 2 - delta + delta_offset
                        if 0 <= t_prev < table.shape[0]:
                            t_detour = t - 1 - delta + offset_sum

                            # Fast early-exit: check if any candidate has relative contrast advantage over target anchor
                            has_candidate = 0
                            for j_idx in range(num_intrusive_tokens):
                                if lpz[t_detour, intrusive_token_ids[j_idx]] > lpz[t_detour, anchor_tok]:
                                    has_candidate = 1
                                    break
                            if not has_candidate:
                                continue

                            blank_sum = 0.0
                            for b_t in range(t - delta, t):
                                blank_sum += lpz[b_t + offset_sum, blank]
                            base_prob = (
                                table[t_prev, c - 1 - s]
                                + blank_sum
                                + lpz[t + offset_sum, anchor_tok]
                            )
                            for j_idx in range(num_intrusive_tokens):
                                j = intrusive_token_ids[j_idx]
                                if lpz[t_detour, j] > lpz[t_detour, anchor_tok]:
                                    p_cand = base_prob + lpz[t_detour, j]
                                    if p_cand > intrusive_prob:
                                        intrusive_prob = p_cand

            # Compute stay probability
            if t - 1 < 0:
                stay_prob = prob_floor
            elif preamble_transition_cost_zero:
                stay_prob = 0
            elif stay_transition_cost_zero:
                stay_prob = table[t - 1, c]
            else:
                stay_prob = table[t - 1, c] + max(lpz[t + offset_sum, blank], max_lpz_prob)
            # Use max of stay, switch, syncope skip, and intrusive detour prob
            table[t, c] = max(max(max(switch_prob, stay_prob), syncope_skip_prob), intrusive_prob)
            # Remember the row with the max prob
            if last_arg_max == -1 or last_max < table[t, c]:
                last_max = table[t, c]
                last_arg_max = t
    # Return cell index with max prob in last column
    c = table.shape[1] - 1
    t = table[:, c].argmax()
    return t, c

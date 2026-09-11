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
                      float syncope_penalty,
                      int blank,
                      int flags):
    """Fill the table of transition probabilities.

    :param table: table filled with maximum joint probabilities k_{t,j}
    :param lpz: character probabilities of each time frame
    :param ground_truth: label sequence
    :param offsets: window offsets per character (given as array of zeros)
    :param is_syncope_token: 1D mask array marking optional syncope token positions
    :param syncope_penalty: penalty subtracted for syncope skip transition
    :param blank: label ID of the blank symbol, usually 0
    :param flags: configuration options, default 0
    :return:
    """
    cdef int c
    cdef int t
    cdef int offset = 0
    cdef float mean_offset
    cdef int offset_sum = 0
    cdef int lower_offset
    cdef int higher_offset
    cdef float switch_prob, stay_prob, skip_prob, syncope_skip_prob
    cdef float prob_max = -1000000000
    cdef float last_max
    cdef int last_arg_max
    cdef np.ndarray[np.int64_t, ndim=1] cur_offset = np.zeros([ground_truth.shape[1]], np.int64) - 1
    cdef float max_lpz_prob
    cdef float p, v_prob
    cdef int s, c_prev, delta_offset, t_prev
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
            switch_prob = prob_max
            max_lpz_prob = prob_max
            for s in range(ground_truth.shape[1]):
                if ground_truth[c, s] != -1:
                    if t >= table.shape[0] - (cur_offset[s] - 1) or t - 1 + cur_offset[s] < 0 or c == 0:
                        p = prob_max
                    else:
                        p = table[t - 1 + cur_offset[s], c - (s + 1)] + lpz[t + offset_sum, ground_truth[c, s]]
                    switch_prob = max(switch_prob, p)
                    max_lpz_prob = max(max_lpz_prob, lpz[t + offset_sum, ground_truth[c, s]])

            # Compute syncope skip probability
            syncope_skip_prob = prob_max
            if c >= 2:
                for s in range(ground_truth.shape[1]):
                    if ground_truth[c, s] != -1:
                        # If c - 1 is a syncope token, jump from c - 2 (or c - 3)
                        if is_syncope_token[c - 1] == 1:
                            for c_prev in range(max(0, c - 3), c - 1):
                                delta_offset = offset_sum - offsets[c_prev]
                                t_prev = t - 1 + delta_offset
                                if 0 <= t_prev < table.shape[0]:
                                    v_prob = table[t_prev, c_prev] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                                    if v_prob > syncope_skip_prob:
                                        syncope_skip_prob = v_prob
                        # If c - 2 is a syncope token and c - 1 is a blank/space, jump from c - 3 (or c - 4)
                        elif c >= 3 and is_syncope_token[c - 2] == 1 and (ground_truth[c - 1, 0] == blank or ground_truth[c - 1, 0] == -1):
                            for c_prev in range(max(0, c - 4), c - 2):
                                delta_offset = offset_sum - offsets[c_prev]
                                t_prev = t - 1 + delta_offset
                                if 0 <= t_prev < table.shape[0]:
                                    v_prob = table[t_prev, c_prev] + lpz[t + offset_sum, ground_truth[c, s]] - syncope_penalty
                                    if v_prob > syncope_skip_prob:
                                        syncope_skip_prob = v_prob

            # Compute stay probability
            if t - 1 < 0:
                stay_prob = prob_max
            elif preamble_transition_cost_zero:
                stay_prob = 0
            elif stay_transition_cost_zero:
                stay_prob = table[t - 1, c]
            else:
                stay_prob = table[t - 1, c] + max(lpz[t + offset_sum, blank], max_lpz_prob)
            # Use max of stay, switch, and syncope skip prob
            table[t, c] = max(max(switch_prob, stay_prob), syncope_skip_prob)
            # Remember the row with the max prob
            if last_arg_max == -1 or last_max < table[t, c]:
                last_max = table[t, c]
                last_arg_max = t
    # Return cell index with max prob in last column
    c = table.shape[1] - 1
    t = table[:, c].argmax()
    return t, c

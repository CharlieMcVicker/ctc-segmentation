 [Bug] Syncope Contrastive Gate Only Inspects Arrival Consonant Frame, Permitting
  Spurious Vowel Deletions During Mismatched Vowels                                
                                                                                   
  ## Description                                                                   
                                                                                   
  When aligning speech where the speaker utters a substitute or dialectal vowel    
  (e.g. /a/ instead of citation /e/ in ale, or /v/ instead of citation /a/ in      
  nahski), the syncope-aware forward DP trellis erroneously drops the vowel        
  (emitting al or nhski), even when syncope token class pooling (syncope_tokens =  
  (("a", "e", "i", "o", "u", "v"), "t")) is enabled.                               
                                                                                   
  While syncope transitions are designed to be purely logprob-gated (zero-penalty, 
  opening only when acoustic evidence demonstrates the sound was deleted), the     
  current gating implementation in ctc_segmentation_dyn.pyx:124-168 and            
  ctc_segmentation.py:765-805 contains an arrival-frame gating blindspot.          
  ──────                                                                           
  ## Root Cause Analysis                                                           
                                                                                   
  In both Case A (intra-word syncope) and Case B (inter-word syncope across blank),
  the contrastive gate is evaluated only on the arrival frame t corresponding to   
  the next consonant anchor:                                                       
                                                                                   
    # ctc_segmentation_dyn.pyx: Case B (inter-word syncope)                        
    elif c >= 3 and is_syncope_token[c - 2] == 1 and (ground_truth[c - 1, 0] ==    
  blank):                                                                          
        vowel_tok = ground_truth[c - 2, 0]               # Target vowel e.g. 'e'   
        gate_tok = syncope_token_gates[vowel_tok]        # Pooled class {a, e, i, o,
  u, v}                                                                            
                                                                                   
        # Arrival-frame gate:                                                      
        if lpz[t + offset_sum, anchor_tok] > lpz[t + offset_sum, gate_tok]:        
            # Syncope skip transition is taken!                                    
            v_prob = table[t_prev, c - 3] + lpz[t + offset_sum, anchor_tok]        
            if v_prob > syncope_skip_prob:                                         
                syncope_skip_prob = v_prob                                         
                                                                                   
  ### Why Gating Fails:                                                            
                                                                                   
  1. At Intermediate Vowel Frames (t_{vowel} ∈ [t_{prev}, t - 1]):                 
      • The speaker uttered a vowel (e.g. a).                                      
      • The pooled vowel posterior is high: lpz[t_{vowel}, 𝒞_{vowels}] ≈ 0.0.      
      • However, the exact citation token (e.g. e) has a very low posterior:       
      lpz[t_{vowel}, 'e'] ≈ -18.0.                                                 
      • Taking the canonical emit path for e would accumulate -18.0. The trellis   
      avoids this by self-looping on the preceding consonant l or [PAD].           
  2. At the Consonant Arrival Frame (t):                                           
      • Frame t is the acoustic onset of the next word's consonant (e.g. 'n' in    
      nikatv).                                                                     
      • The gate checks:                                                           
                                                                                   
                                                                                   
    lpz[t, 'n'] > lpz⎡t, 𝒞      ⎤ ⟹ 0.0 > -16.0  (𝐓𝐑𝐔𝐄)                            
                     ⎣    vowels⎦                                                  
                                                                                   
  • Because frame t is indeed a consonant, the gate evaluates to OPEN.             
                                                                                   
  3. Result: The trellis never inspected whether vocalic evidence was present      
  during the skipped interval [t_{prev}, t - 1]. The DP path takes the syncope skip
  into 'n', discarding the uttered vowel a and clipping ale to al.                 
  ──────                                                                           
  ## Minimal Reproduction Case                                                     
                                                                                   
    import numpy as np                                                             
    import ctc_segmentation                                                        
                                                                                   
    vocab = {"[PAD]": 0, "a": 1, "e": 2, "i": 3, "o": 4, "u": 5, "v": 6, "t": 7,   
  "l": 8, "n": 9, "k": 10, "s": 11, "h": 12, "'": 13, "|": 14}                     
    inv_vocab = {v: k for k, v in vocab.items()}                                   
    char_list = [inv_vocab[i] for i in range(len(vocab))]                          
                                                                                   
    # Synthetic audio: 'a' (1-2), 'l' (3-4), 'a' (5-6) [spoken ala], 'n' (8-9), 'i'
  (10-11), 'k' (12-13), 'a' (14-15), 't' (16-17), 'v' (18-19)                      
    T = 21                                                                         
    lpz = np.full((T, len(vocab)), -20.0, dtype=np.float32)                        
    lpz[0, 0] = 0.0                                                                
    lpz[1:3, vocab["a"]] = 0.0                                                     
    lpz[3:5, vocab["l"]] = 0.0                                                     
    lpz[5:7, vocab["a"]] = 0.0  # Spoken vowel is 'a', target is 'e'               
    lpz[7, 0] = 0.0                                                                
    lpz[8:10, vocab["n"]] = 0.0                                                    
    lpz[10:12, vocab["i"]] = 0.0                                                   
    lpz[12:14, vocab["k"]] = 0.0                                                   
    lpz[14:16, vocab["a"]] = 0.0                                                   
    lpz[16:18, vocab["t"]] = 0.0                                                   
    lpz[18:20, vocab["v"]] = 0.0                                                   
    lpz[20, 0] = 0.0                                                               
                                                                                   
    params = ctc_segmentation.CtcSegmentationParameters(                           
        char_list=char_list,                                                       
        blank=0,                                                                   
        syncope_tokens=[["a", "e", "i", "o", "u", "v"], "t"],
        replace_spaces_with_blanks=False,
    )
    gt_mat, utt_indices = ctc_segmentation.prepare_text(params, ["ale", "nikatv"]) 
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(params, lpz,
  gt_mat)
  
    # Bug: State list outputs ['a', 'l', 'n', 'i', 'k', 'a', 't', 'v'], dropping   
  'e'
    assert "e" in state_list, f"Vowel was erroneously dropped! state_list:         
  {state_list}"
  ──────
  ## Proposed Solution
  
  1. Interval-Aware Syncope Gating:
      • When evaluating a syncope skip from predecessor state (t_{prev}, c - k) to 
      (t, c), the contrastive gate must verify that no vocalic peak occurred in the
      intermediate gap τ ∈ [t_{prev} + 1, t - 1]:
  

    Gate        = OPEN ⟺        max         lpz[τ, gate\_tok] <
  blank\_or\_silence\_threshold
        syncope          τ∈[t_{prev}+1,t-1]
  
  2. Strict Gate Closure on Vocalic Presence:
      • If any frame in the skipped window exhibits significant acoustic energy for
      the pooled vowel class 𝒞_{vowels}, the syncope gate must evaluate to CLOSED  
      (-∞), forcing the trellis along the canonical path to retain the syllable    
      nucleus.
  
  ──────
  ## Acceptance Criteria
  
  [ ] Synthetic tests reproducing ale (spoken ala) and nahski (spoken nvhski)      
  retain their canonical vowel states when class pooling is active.
  [ ] Real-world regression tests (test_syncope_class_prevents_kwo_vowel_clipping) 
  continue to pass without regressions on natural vowel syncope.
  [ ] Pure logprob-gated design (λ_{syncope} = 0.0) is preserved without           
  introducing arbitrary additive penalty hyperparameters.
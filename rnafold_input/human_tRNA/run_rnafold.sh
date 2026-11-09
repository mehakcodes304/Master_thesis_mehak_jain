#!/bin/bash
# Run RNAfold for all RNAs — one command per condition.
# Output files contain all RNAs concatenated.
# compare_rnafold.py parses them by header name.

# Condition A: unconstrained baseline (pure thermodynamics)
RNAfold --noPS \
  < /mnt/ceph/mjain/software/envs/bio/rnafold_input/human_tRNA/all_rnafold_unconstrained_input.txt \
  > /mnt/ceph/mjain/software/envs/bio/rnafold_output/human_tRNA/all_rnafold_unconstrained_output.txt

# Condition B: DDD hard-constrained
RNAfold --constraint --noPS \
  < /mnt/ceph/mjain/software/envs/bio/rnafold_input/human_tRNA/all_rnafold_constrained_input.txt \
  > /mnt/ceph/mjain/software/envs/bio/rnafold_output/human_tRNA/all_rnafold_constrained_output.txt


import pandas as pd


def parse_base_pairs(dot_bracket: str) -> list[tuple[int, int]]:
    """Parse dot-bracket → sorted list of (i, j) base-pair positions (1-indexed)."""
    stack = []
    pairs = []

    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            pairs.append((j + 1, i + 1))

    return sorted(pairs)


def group_into_stems(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """
    Group consecutive base pairs into stems.
    Two pairs (i1,j1) and (i2,j2) belong to the same stem if:
        i2 == i1 + 1  AND  j2 == j1 - 1
    """
    if not pairs:
        return []

    stems = []
    current_stem = [pairs[0]]

    for k in range(1, len(pairs)):
        i_prev, j_prev = pairs[k - 1]
        i_curr, j_curr = pairs[k]

        if i_curr == i_prev + 1 and j_curr == j_prev - 1:
            current_stem.append(pairs[k])
        else:
            stems.append(current_stem)
            current_stem = [pairs[k]]

    stems.append(current_stem)
    return stems


def gap_between(stem_a: list, stem_b: list) -> float:
    """
    Measure the structural gap between two stems by averaging both sides:
      - i_gap: how far apart the stems are on the left (i positions)
      - j_gap: how far apart the stems are on the right (j positions)
    """
    i_gap = abs(stem_b[0][0] - stem_a[-1][0])
    j_gap = abs(stem_a[-1][1] - stem_b[0][1])
    return (i_gap + j_gap) / 2


def cross_diff(stem_a: list, stem_b: list) -> float:
    """
    Returns abs(diff_i_to_j - diff_j_to_i) between two consecutive stems.
    Used to measure structural asymmetry between stem pairs.
    """
    diff_i_to_j = abs(stem_a[0][0] - stem_b[0][1])   # i_start(a) - j_end(b)
    diff_j_to_i = abs(stem_a[0][1] - stem_b[0][0])   # j_end(a)   - i_start(b)
    return abs(diff_i_to_j - diff_j_to_i)


def can_merge(stem_a: list, stem_b: list) -> bool:
    """
    Gate check before any merge is attempted.
    Only allow merging if j_start of stem_a < j_start of stem_b.

    j_start is the innermost j position of a stem → stem[-1][1]

    If j_start(a) >= j_start(b), the stems are likely two separate
    structural arms (e.g. a palindrome pattern) and must stay split.
    """
    j_start_a = stem_a[-1][1]   # innermost j of stem_a
    j_start_b = stem_b[-1][1]   # innermost j of stem_b
    return j_start_a < j_start_b


def stem_to_range(stem: list) -> str:
    """
    Format a stem as a human-readable range string.
    Example: [(1,72),(2,71),(3,70)] → "[(1-3), (70-72)]"
    """
    i_start = stem[0][0]
    i_end   = stem[-1][0]
    j_start = stem[-1][1]
    j_end   = stem[0][1]
    return f"[({i_start}-{i_end}), ({j_start}-{j_end})]"


# --- STEP 1: merge short stems by proximity ---

def merge_short_stems(stems: list, min_length: int = 3, max_i_gap: int = 10, max_j_gap: int = 10, min_stems: int = 0) -> list:
    """
    STEP 1: Find stems shorter than min_length and merge them into the
    nearest neighbor if within max_gap. If no neighbor is close enough,
    leave the stem in place — it will be caught by the final discard step.

    Parameters:
        min_length : stems strictly shorter than this are candidates for merging
        max_gap    : maximum allowed average (i_gap + j_gap) / 2 to merge
        min_stems  : never reduce total stem count below this number
    """
    changed = True
    while changed:
        changed = False
        i = 0

        while i < len(stems):
            if len(stems[i]) < min_length:

                # guard: don't drop below min_stems
                if len(stems) <= min_stems:
                    i += 1
                    continue

                has_prev = i > 0
                has_next = i < len(stems) - 1

                gap_to_prev = gap_between(stems[i - 1], stems[i]) if has_prev else float('inf')
                gap_to_next = gap_between(stems[i], stems[i + 1]) if has_next else float('inf')

                prev_reachable = has_prev and gap_to_prev <= max_i_gap and max_j_gap 
                next_reachable = has_next and gap_to_next <= max_i_gap and max_j_gap 

                if prev_reachable and next_reachable:
                    # both close — merge with the nearer one
                    if gap_to_prev <= gap_to_next:
                        stems[i - 1] = stems[i - 1] + stems[i]
                    else:
                        stems[i + 1] = stems[i] + stems[i + 1]
                    stems.pop(i)
                    changed = True

                elif prev_reachable:
                    stems[i - 1] = stems[i - 1] + stems[i]
                    stems.pop(i)
                    changed = True

                elif next_reachable:
                    stems[i + 1] = stems[i] + stems[i + 1]
                    stems.pop(i)
                    changed = True

                else:
                    # no neighbor close enough — leave it, discard step will handle it
                    i += 1

            else:
                i += 1

    return stems


# --- STEP 2: merge consecutive stems by cross difference ---

def merge_by_cross_diff(stems: list, max_cross_diff: int = 6) -> list:
    """
    STEP 2: Merge consecutive stem pairs where
    abs(diff_i_to_j - diff_j_to_i) <= max_cross_diff.

    Parameters:
        max_cross_diff : threshold — pairs with difference <= this get merged
    """
    changed = True
    while changed:
        changed = False
        i = 0

        while i < len(stems) - 1:
            if can_merge(stems[i], stems[i + 1]) and cross_diff(stems[i], stems[i + 1]) <= max_cross_diff:
                stems[i] = stems[i] + stems[i + 1]
                stems.pop(i + 1)
                changed = True          # merged stem may qualify again with its new neighbor
            else:
                i += 1

    return stems

def remove_nested_interactions(stems: list) -> list:
    """
    Remove any stem whose both arms are nested inside one arm of another stem.
    These can never produce unique chimeric reads.
    """
    def arm_ranges(stem):
        return (stem[0][0], stem[-1][0]), (stem[-1][1], stem[0][1])

    to_remove = set()
    for i, stem_a in enumerate(stems):
        a1, a2 = arm_ranges(stem_a)
        for j, stem_b in enumerate(stems):
            if i == j:
                continue
            b1, b2 = arm_ranges(stem_b)
            # Check if both arms of stem_a fall inside one arm of stem_b
            for b_arm in [b1, b2]:
                if (b_arm[0] <= a1[0] and a1[1] <= b_arm[1] and
                    b_arm[0] <= a2[0] and a2[1] <= b_arm[1]):
                    to_remove.add(i)
    return [s for k, s in enumerate(stems) if k not in to_remove]

# --- STEP 3: discard stems still below minimum length ---

def discard_short_stems(stems: list, min_length: int = 3) -> list:
    """
    STEP 3: Final cleanup — remove any stems that are still shorter than
    min_length after all merging is complete.

    Parameters:
        min_length : stems strictly shorter than this are discarded
    """
    return [stem for stem in stems if len(stem) >= min_length]


# --- Validation ---

def lengths_match(seq: str, sec_str: str, name: str) -> bool:
    """
    Check that the sequence and secondary structure have equal length.
    Prints a warning and returns False if they don't match.
    """
    if len(seq) != len(sec_str):
        print(f"WARNING: lengths do not match for {name} "
              f"(seq={len(seq)}, sec_str={len(sec_str)}) — skipping row.")
        return False
    return True


# --- Main pipeline ---

def extract_stems(df: pd.DataFrame,
                  min_length: int = 0,
                  max_i_gap: int = 8,
                  max_j_gap: int = 8, 
                  min_stems: int = 0,
                  max_cross_diff: int = 6) -> pd.DataFrame:
    """
    Full pipeline per tRNA row:
      0. Validate seq and sec_str lengths match
      1. Parse base pairs → group into stems
      2. Merge short stems by proximity (merge_short_stems)
      3. Merge consecutive stems by cross difference (merge_by_cross_diff)
      4. Discard any remaining stems below min_length (discard_short_stems)
      5. Build output table with range strings and cross-diff columns
    """
    records = []

    for _, row in df.iterrows():

        # --- Step 0: validate lengths ---
        if not lengths_match(row["seq"], row["sec_str"], row["name"]):
            continue                       # skip this row entirely

        # --- Step 1: parse and group ---
        pairs = parse_base_pairs(row["sec_str"])
        stems = group_into_stems(pairs)

        # --- Step 2: merge short stems by proximity ---
        stems = merge_short_stems(stems, min_length=4, max_i_gap=max_i_gap, max_j_gap=max_j_gap, min_stems=min_stems)

        # --- Step 3: merge by cross difference ---
        stems = merge_by_cross_diff(stems, max_cross_diff=max_cross_diff)

        # --- Step 4: discard anything still too short ---
        stems = discard_short_stems(stems, min_length=3)

        # --- Step 5: build records ---
        for interaction_number, stem in enumerate(stems, start=1):

            i_start_curr = stem[0][0]
            j_end_curr   = stem[0][1]

            if interaction_number < len(stems):
                next_stem             = stems[interaction_number]   # 1-indexed → next stem
                diff_i_curr_to_j_next = abs(i_start_curr - next_stem[0][1])
                diff_j_curr_to_i_next = abs(j_end_curr   - next_stem[0][0])
            else:
                diff_i_curr_to_j_next = None
                diff_j_curr_to_i_next = None

            records.append({
                "name":                   row["name"],
                "seq_len":                row["seq_len"],
                "interaction_number":     interaction_number,
                "sub_interaction_stem":   stem_to_range(stem),
                #"diff_i_curr_to_j_next":  diff_i_curr_to_j_next,
                #"diff_j_curr_to_i_next":  diff_j_curr_to_i_next,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = pd.read_csv("human_rRNA.csv")

    stems_df = extract_stems(df)
    stems_df.to_csv("stems2_human_rRNA.csv", index=False)

    print(stems_df.to_string(index=False))

if __name__ == "__main__":
    df = pd.read_csv("ecoli_rRNA.csv")

    stems_df = extract_stems(df)
    stems_df.to_csv("stems2_ecoli_rRNA.csv", index=False)

    print(stems_df.to_string(index=False))
 
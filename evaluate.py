import argparse
from argparse import RawDescriptionHelpFormatter
from collections import defaultdict


"""
EVALUATOR FOR TIMEX DETECTION AND NORMALIZATION

This program evaluates timex detection using CoNLL and TE3 evaluation metrics
and timex normalization using TE3 metrics.

For timex detection, evaluation can be done in strict (all tokens match) or
relaxed mode (at least a token matches) and considering type matches or not.
It works with the IOB2 tagging scheme.
    - CoNLL: --mode 'strict', --consider_type
    - rel-t: --mode 'relaxed', --consider_type
    - TE3 strict: --mode 'strict'
    - TE3 relaxed: --mode 'relaxed' (default)

For timex normalization, evaluation uses accuracy and TE3 F1 value metrics.

CoNLL evaluator adapted from https://github.com/sighsmile/conlleval.git.
"""


def main(args):
    # Recall the command-line arguments
    pred_file = args.pred_file               # token g-det p-det [g-norm p-norm]
    mode = args.mode                            # "strict"|"str"|"relaxed"|"rel"
    consider_type = args.consider_type                             # No argument
    all_metrics = args.all_metrics                                 # No argument

    mode = "strict" if mode in ["strict", "str"] else "relaxed"

    # Assign mode and type consideration for the four metrics
    metrics = {"CoNLL": ("strict", True), "rel-t": ("relaxed", True),
    "TE3 strict": ("strict", False), "TE3 relaxed": ("relaxed", False)}

    # Obtain data from pred_file
    with open(pred_file, "r", encoding='utf-8') as pred:
        pred_data = pred.readlines()
        tokens = get_column(pred_data, 0)
        gold_tags = get_column(pred_data, 1)
        pred_tags = get_column(pred_data, 2)
        # If normalization data is provided in the same file, obtain it
        try:
            gold_vals = get_column(pred_data, 3)
            pred_vals = get_column(pred_data, 4)
            norm_data = True
        except:
            gold_vals = ["-" if token != "" else "" for token in tokens]
            pred_vals = ["-" if token != "" else "" for token in tokens]
            norm_data = False

    # Obtain results for the four metric modes if requested
    if all_metrics:
        # Evaluate timex detection (and normalization if data in the same file)
        for k, v in metrics.items():
            print(f"\n{k.upper()} TIMEX EXTRACTION\n")
            # Compute detection metrics according to evaluation features (mode,
            # type) and normalization values if provided in the same file
            (correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
             correct_counts, gold_counts, pred_counts) = count_chunks(tokens,
             gold_tags, pred_tags, gold_vals, pred_vals, v[0], v[1])

            # Obtain timex detection results and print
            f1 = evaluate(correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
                          correct_counts, gold_counts, pred_counts, norm_data)

    else:
        # Compute detection metrics according to evaluation features (mode,
        # type) and normalization values if provided in the same file
        (correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
         correct_counts, gold_counts, pred_counts) = count_chunks(tokens,
         gold_tags, pred_tags, gold_vals, pred_vals, mode, consider_type)

        # Obtain timex detection results and print
        f1 = evaluate(correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
                      correct_counts, gold_counts, pred_counts, norm_data)


def get_column(file_data, col_n):
    """Obtain a concrete column of a file as a list."""
    sep = "\t"
    # If line is not empty, append the corresponding tag
    column = []
    for line in file_data:
        if line != "\n":
            column.append(line.strip().split(sep)[col_n])
        else:
            column.append("")
    return column


def split_tag(chunk_tag):
    """Splits chunk tags into IOBES prefix and chunk type."""
    # If chunk tag is "O" or an empty line, return ("O", None)
    if chunk_tag == "O" or chunk_tag == "":
        return ("O", None)
    # Otherwise, return the corresponding position and type
    return chunk_tag.split("-", maxsplit=1)


def is_chunk_end(prev_tag, tag, consider_type):
    """Checks if a chunk ended between the previous and current word."""
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == "O":
        return False
    elif prefix2 == "B":
        return True
    elif prefix2 == "O":
        return prefix1 != "O"
    elif consider_type and chunk_type1 != chunk_type2:
        return True
    else:
        return False


def is_chunk_start(prev_tag, tag, consider_type):
    """Checks if a new chunk started between the previous and current word."""
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == "O":
        return False
    elif prefix2 == "B":
        return True
    elif prefix1 == "O":
        return prefix2 != "O"
    elif consider_type and chunk_type1 != chunk_type2:
        return True
    else:
        return False


def count_chunks(tokens, gold_tags, pred_tags, gold_vals, pred_vals, mode, consider_type):
    """Counts chunks and tags according to evaluation features."""
    # Dicts for number of correct, gold and predicted chunks per type
    correct_chunks = defaultdict(int)
    gold_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)
    # Dicts for number of correct, gold and predicted type tags
    correct_counts = defaultdict(int)
    gold_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    # Dicts for number of correct types and values per type
    correct_dets = defaultdict(int)
    correct_norms = defaultdict(int)
    # Initialize variables
    prev_gold_tag, prev_pred_tag = "O", "O"
    prev_gold_val, prev_pred_val = "-", "-"
    chunk_tag, gold_chunk, pred_chunk = None, False, False

    data = list(zip(tokens, gold_tags, pred_tags, gold_vals, pred_vals))

    for i, (token, gold_tag, pred_tag, gold_val, pred_val) in enumerate(data):
        # Compute type tag occurrences if token is not an empty string
        if token != "":
            if gold_tag == pred_tag:
                correct_counts[gold_tag] += 1
            gold_counts[gold_tag] += 1
            pred_counts[pred_tag] += 1
        # Obtain gold and predicted token position and type tags
        gold_pos, gold_type = split_tag(gold_tag)
        pred_pos, pred_type = split_tag(pred_tag)
        # Check if this token is the end or the beginning of a timex
        gold_end = is_chunk_end(prev_gold_tag, gold_tag, consider_type)
        pred_end = is_chunk_end(prev_pred_tag, pred_tag, consider_type)
        gold_start = is_chunk_start(prev_gold_tag, gold_tag, consider_type)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag, consider_type)

        # Strict evaluation: spans match completely
        if mode == "strict":
            # If a timex has previously started:
            if chunk_tag is not None:
                # If the timex ends in gold and pred, count it as correct and
                # initialize chunk_tag
                if pred_end and gold_end:
                    correct_chunks[chunk_tag] += 1
                    # If value data is provided and coincides, count a match
                    if prev_gold_val == prev_pred_val:
                        correct_norms[chunk_tag] += 1
                    # If type data coincides, count a match
                    if split_tag(prev_gold_tag)[1] == split_tag(prev_pred_tag)[1]:
                        correct_dets[chunk_tag] += 1
                    chunk_tag = None
                # If timex end does not match, initialize chunk_tag
                if pred_end != gold_end:
                    chunk_tag = None
                # If type match is considered and does not match, initialize chunk_tag
                if consider_type and gold_type != pred_type:
                    chunk_tag = None
            # If timex start matches and, considering type, there is a type
            # match, or type is not considered, set chunk_tag to gold type
            if (gold_start and pred_start and
                ((consider_type and gold_type == pred_type) or not consider_type)):
                chunk_tag = gold_type
            # Count gold and pred chunks
            if gold_start:
                gold_chunks[gold_type] += 1
            if pred_start:
                pred_chunks[pred_type] += 1

        # Relaxed evaluation: at least 1 token matches
        if mode == "relaxed":
            # If a timex ends, set chunk to False
            if gold_end:
                gold_chunk = False
            if pred_end:
                pred_chunk = False
            # If a timex starts, count the chunk and set it to True
            if gold_start:
                gold_chunks[gold_type] += 1
                gold_chunk = True
            if pred_start:
                pred_chunks[pred_type] += 1
                pred_chunk = True

            # If a timex token matches (a prediction might match a started gold
            # timex) and, considering type, there is a type match, or type is
            # not considered:
            if ((gold_chunk and pred_chunk) and
                ((consider_type and gold_type == pred_type) or not consider_type)):
                    # Update chunk_tag, count it as correct and initialize variables
                    chunk_tag = gold_type
                    correct_chunks[chunk_tag] += 1
                    # If value data is provided and coincides, count a match
                    if gold_val == pred_val:
                        correct_norms[chunk_tag] += 1
                    # If type data coincides, count a match
                    if gold_type == pred_type:
                        correct_dets[chunk_tag] += 1
                    chunk_tag = None
                    gold_chunk, pred_chunk = False, False

        # Turn current tags values into previous tags and values
        prev_gold_tag, prev_pred_tag = gold_tag, pred_tag
        prev_gold_val, prev_pred_val = gold_val, pred_val

    # If the last token is a timex, count it as correct
    if chunk_tag is not None:
        correct_chunks[chunk_tag] += 1

    return (correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
            correct_counts, gold_counts, pred_counts)


def calculate_metrics(tp, t, p, percent=True):
    """Computes overall precision, recall and FB1.
    If percent is True, return 100 * original decimal value."""
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def evaluate(correct_chunks, gold_chunks, pred_chunks, correct_dets, correct_norms,
    correct_counts, gold_counts, pred_counts, norm_data):
    """Prints overall and chunk-type performance on precision, recall and
    FB1 score."""
    # Sum counts on chunks and tags
    sum_correct_chunks = sum(correct_chunks.values())
    sum_gold_chunks = sum(gold_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_gold_counts = sum(gold_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != "O")
    nonO_gold_counts = sum(v for k, v in gold_counts.items() if k != "O")

    # Get the sorted list of of chunk types
    chunk_types = sorted(list(set(list(gold_chunks) + list(pred_chunks))))

    # Compute overall precision, recall and FB1
    prec, rec, f1 = calculate_metrics(sum_correct_chunks, sum_gold_chunks, sum_pred_chunks)

    # Sum correct type counts and compute metrics
    sum_correct_dets = sum(correct_dets.values())
    prec_type, rec_type, f1_type = calculate_metrics(sum_correct_dets, sum_gold_chunks, sum_pred_chunks)
    acc_type = f1_type * 100 / f1

    # Sum correct norm counts and compute metrics for TE3 evaluation
    sum_correct_norms = sum(correct_norms.values())
    prec_val, rec_val, f1_val = calculate_metrics(sum_correct_norms, sum_gold_chunks, sum_pred_chunks)
    acc_val = f1_val * 100 / f1

    # Print overall performance and performance per chunk type
    print(f"Processed {sum_gold_counts} tokens with {sum_gold_chunks} phrases: "
    f"{sum_pred_chunks} phrases predicted, {sum_correct_chunks} correct.\n\n"

    f"{'Precision':>20}{'Recall':>10}{'F1':>10}{'Gold':>10}{'Pred':>10}{'Correct':>10}\n"
    f"Overall{prec:13.2f}{rec:10.2f}{f1:10.2f}{sum_gold_chunks:10}{sum_pred_chunks:10}{sum_correct_chunks:10}")

    # For each chunk type, compute precision, recall and FB1
    for t in chunk_types:
        prec_t, rec_t, f1_t = calculate_metrics(correct_chunks[t], gold_chunks[t], pred_chunks[t])
        print(f"{t:<10s}{prec_t:10.2f}{rec_t:10.2f}{f1_t:10.2f}{gold_chunks[t]:10}{pred_chunks[t]:10}{correct_chunks[t]:10}")

    print(f"\nNon-O accuracy: {(100 * nonO_correct_counts / nonO_gold_counts):8.2f}\n"
    f"Overall accuracy: {(100 * sum_correct_counts / sum_gold_counts):6.2f}\n")

    # Print overall type and value performance
    print(f"{'Precision':>20}{'Recall':>10}{'F1':>10}{'Gold':>10}{'Pred':>10}{'Correct':>10}{'Accuracy':>10}\n"
    f"Type{prec_type:16.2f}{rec_type:10.2f}{f1_type:10.2f}{sum_gold_chunks:10}{sum_pred_chunks:10}{sum_correct_dets:10}{acc_type:10.2f}")
    # If there is normalization data, output results
    if norm_data:
        print(f"Value{prec_val:15.2f}{rec_val:10.2f}{f1_val:10.2f}{sum_gold_chunks:10}{sum_pred_chunks:10}{sum_correct_norms:10}{acc_val:10.2f}")

    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program evaluates timex "
    "detection and normalization in strict or relaxed mode and considering type or not.\n"
    "These are the possible metrics:\n"
    "- CoNLL: --mode 'strict', --consider_type\n"
    "- rel-t: --mode 'relaxed', --consider_type\n"
    "- TE3 strict: --mode 'strict'\n"
    "- TE3 relaxed: --mode 'relaxed' (default)\n",
    formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("pred_file",
    help="To evaluate timex detection and, if provided, timex normalization. "
    "It must contain gold and prediction data in '{token}\t{gold-det}\t{pred-det}\t[{gold-norm}\t{pred-norm}]' tab-separated format.")
    parser.add_argument("-m", "--mode",
    default="relaxed", required=False, choices=["strict", "str", "relaxed", "rel"],
    help="Consider spans in strict or relaxed evaluation mode. Default: relaxed.")
    parser.add_argument("-t", "--consider_type",
    action="store_true", required=False,
    help="Consider type match or not. Default: False.")
    parser.add_argument("-a", "--all_metrics",
    action="store_true", required=False,
    help="Show results for all metrics (CoNLL, rel-t, TE3 strict, TE3 relaxed).")

    args = parser.parse_args()
    main(args)

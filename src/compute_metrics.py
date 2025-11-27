import string
import json
import os
import argparse
import logging
import collections
import math

from rouge import rouge_scorer
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
GPT2TOKENIZER = os.path.join(CURRENT_DIR, "../data/gpt2tokenizer")


# class GPTTokenizer:
#     gpt_tokenizer = AutoTokenizer.from_pretrained(GPT2TOKENIZER, max_length=1e5)

#     def tokenize(self, s):
#         tokens = self.gpt_tokenizer.tokenize(s)
#         # GPT2 uses Byte-level BPE, which will include space as part of the word. 
#         # But for the first word of a sentence, there is no space before it. 
#         # So, we remove all the added spaces ("Ġ"). 
#         tokens = [t.lstrip("Ġ") for t in tokens]
#         return tokens


# xlingual_tokenizer = GPTTokenizer()


class BleuScorer:
    """BLEU score computation class."""
    
    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams up to a given max_order from a token list."""
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu(self, reference_corpus, translation_corpus, max_order=4, smooth=False):
        """
        Computes BLEU score of translated segments against one or more references.

        reference_corpus: list of lists of references for each translation.
                        Each reference should be a tokenized list.
        translation_corpus: list of tokenized translations to score.
        """
        # Handle empty corpora
        if not reference_corpus or not translation_corpus:
            return 0.0
        
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0

        for (references, translation) in zip(reference_corpus, translation_corpus):
            # Handle empty references or translations
            if not references or not translation or not any(references):
                continue
                
            # references is a list of token lists; translation is a single token list
            reference_length += min(len(r) for r in references if r)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)

            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts

            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]

            for order in range(1, max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                    possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        # Handle zero reference length
        if reference_length == 0:
            return 0.0

        ratio = float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1.0
        else:
            if reference_length == 0:
                bp = 0.0
            else:
                bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp
        return bleu  # typically a float in [0..1]

import math
import re
import sys
import xml.sax.saxutils

class SmoothBLEU:
    def __init__(self, n=4, smooth=True, preserve_case=False, eff_ref_len="shortest"):
        self.n = n
        self.smooth = smooth
        self.preserve_case = preserve_case
        self.eff_ref_len = eff_ref_len
        self.nonorm = False

        self.normalize1 = [
            (re.compile('<skipped>'), ''), 
            (re.compile(r'-\n'), ''), 
            (re.compile(r'\n'), ' ')
        ]
        self.normalize2 = [
            (re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
            (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
            (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
            (re.compile(r'([0-9])(-)'), r'\1 \2 ')
        ]

    # ------------------------------
    # Core functions
    # ------------------------------
    def normalize(self, s):
        """Normalize and tokenize a sentence (like NIST mteval)."""
        if self.nonorm:
            return s.split()
        if isinstance(s, list):
            s = " ".join(s)
        for (pattern, replace) in self.normalize1:
            s = re.sub(pattern, replace, s)
        s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
        s = f" {s} "
        if not self.preserve_case:
            s = s.lower()
        for (pattern, replace) in self.normalize2:
            s = re.sub(pattern, replace, s)
        return s.split()

    def count_ngrams(self, words):
        counts = {}
        for k in range(1, self.n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    def cook_refs(self, refs):
        refs = [self.normalize(ref) for ref in refs]
        maxcounts = {}
        for ref in refs:
            counts = self.count_ngrams(ref)
            for ngram, count in counts.items():
                maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
        return [len(ref) for ref in refs], maxcounts

    def cook_test(self, test, cooked_refs):
        reflens, refmaxcounts = cooked_refs
        test = self.normalize(test)
        result = {"testlen": len(test), "reflen": 0, "guess": [0]*self.n, "correct": [0]*self.n}

        # effective reference length
        if self.eff_ref_len == "shortest":
            result["reflen"] = min(reflens)
        elif self.eff_ref_len == "average":
            result["reflen"] = float(sum(reflens)) / len(reflens)
        elif self.eff_ref_len == "closest":
            result["reflen"] = min(reflens, key=lambda r: abs(r - len(test)))

        result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, self.n + 1)]

        counts = self.count_ngrams(test)
        for ngram, count in counts.items():
            result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

        return result

    def score_cooked(self, allcomps):
        total = {'testlen': 0, 'reflen': 0, 'guess': [0]*self.n, 'correct': [0]*self.n}
        for comps in allcomps:
            total['testlen'] += comps['testlen']
            total['reflen'] += comps['reflen']
            for i in range(self.n):
                total['guess'][i] += comps['guess'][i]
                total['correct'][i] += comps['correct'][i]

        logbleu = 0.0
        for k in range(self.n):
            correct = total['correct'][k]
            guess = total['guess'][k]
            add_smooth = 1 if self.smooth and k > 0 else 0
            logbleu += math.log(correct + add_smooth + sys.float_info.min) - math.log(guess + add_smooth + sys.float_info.min)
        logbleu /= float(self.n)

        brevity_penalty = min(0, 1 - float(total['reflen'] + 1) / (total['testlen'] + 1))
        bleu_score = math.exp(logbleu + brevity_penalty)
        return bleu_score

    # ------------------------------
    # Public interface
    # ------------------------------
    def compute_bleu(self, refs, candidate):
        cooked_refs = self.cook_refs(refs)
        test = self.cook_test(candidate, cooked_refs)
        return self.score_cooked([test])


bleu_scorer = BleuScorer()
smooth_bleu_scorer = SmoothBLEU()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def bleu_score(prediction, ground_truth, xlingual=False):
    """Compute BLEU score between prediction and ground truth."""
    # Handle empty strings
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    
    if xlingual:
        tokenizer = xlingual_tokenizer
        pred_tokens = tokenizer.tokenize(prediction)
        ref_tokens = tokenizer.tokenize(ground_truth)
    else:
        # Simple whitespace tokenization for default case
        pred_tokens = prediction.split()
        ref_tokens = ground_truth.split()
    
    # Handle empty token lists
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # BLEU expects reference_corpus as list of lists and translation_corpus as list
    reference_corpus = [[ref_tokens]]  # Single reference wrapped in list
    translation_corpus = [pred_tokens]

    return bleu_scorer.compute_bleu(reference_corpus, translation_corpus)

def smooth_bleu_score(prediction, ground_truth, xlingual=False):
    """Compute BLEU score between prediction and ground truth."""
    # Handle empty strings
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    
    if xlingual:
        tokenizer = xlingual_tokenizer
        pred_tokens = tokenizer.tokenize(prediction)
        ref_tokens = tokenizer.tokenize(ground_truth)
    else:
        # Simple whitespace tokenization for default case
        pred_tokens = prediction.split()
        ref_tokens = ground_truth.split()
    
    # Handle empty token lists
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # BLEU expects reference_corpus as list of lists and translation_corpus as list
    reference_corpus = ref_tokens  # Single reference wrapped in list
    translation_corpus = pred_tokens
    print(reference_corpus)
    return smooth_bleu_scorer.compute_bleu(reference_corpus, translation_corpus)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL, bleu, smooth_bleu = 0, 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        bleu += metric_max_over_ground_truths(
            bleu_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        smooth_bleu += metric_max_over_ground_truths(
            smooth_bleu_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    bleu = 100.0 * bleu / len(references)
    smooth_bleu = 100.0 * smooth_bleu / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "eval_rougeL": rougeL, "bleu": bleu, "smooth_bleu": smooth_bleu}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results



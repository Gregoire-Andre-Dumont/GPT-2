import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')

# Function to calculate BLEU score
def calculate_bleu(references, hypotheses):
    smoothie = SmoothingFunction().method1

    # BLEU-1 to BLEU-4
    bleu_1 = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses], weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-4': bleu_4
    }

# Function to calculate ROUGE score
def calculate_rouge(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []}

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    return {key: sum(values) / len(values) for key, values in rouge_scores.items()}

# Example usage
references = ["The quick brown fox jumps over the lazy dog", "A fast brown fox leaps over a sleepy dog"]
hypotheses = ["The quick brown fox jumps over the lazy dog", "A swift brown fox jumps over the sleepy dog"]

bleu_scores = calculate_bleu(references, hypotheses)
# rouge_scores = calculate_rouge(references, hypotheses)

print("BLEU Scores:", bleu_scores)
# print("ROUGE Scores:", rouge_scores)
import nltk

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']

BLEUscore = nltk.translate.bleu_score.sentence_bleu(
    [reference],
    hypothesis,
    weights=[(1. / 2., 1. / 2.),
             # (1. / 3., 1. / 3., 1. / 3.),
             (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
             ]
)
print(BLEUscore)

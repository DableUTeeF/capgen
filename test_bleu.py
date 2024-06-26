import pandas as pd
import sentencepiece as spm
import json
import sacrebleu

sp = spm.SentencePieceProcessor()
sp.Load('/mnt/d/work//sacrebleu_tokenizer_spm.model')


def tokenize_gt(row):
    captions = json.loads(row.captions)
    outputs = [" ".join(sp.EncodeAsPieces(cap)) for cap in captions]
    return outputs


def tokenize_pd(row):
    caption = row.caption
    outputs = " ".join(sp.EncodeAsPieces(caption))
    return outputs


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> sol = pd.DataFrame({'id': [0, 1], 'captions': [json.dumps(['ลาไว้ 3 คนครับ', 'วันนี้', 'วันนี้']), json.dumps(['ลาไว้ 3 คนครับ', 'วันนี้', 'วันนี้'])]})
    >>> sub = pd.DataFrame({'id': [0, 1], 'caption': ['3 คน', 'วันนี้']})
    >>> score(sol, sub, 'id')
    100.0
    """
    labels = solution.apply(tokenize_gt, axis=1)
    preds = submission.apply(tokenize_pd, axis=1)
    transformed_references = [[refs[i] for refs in labels.tolist()] for i in range(3)]
    x = sacrebleu.corpus_bleu(
        preds.tolist(),
        transformed_references,
        smooth_method="exp",
        smooth_value=None,
    )

    return x.precisions[0]


if __name__ == '__main__':
    solution = pd.read_csv('kaggle/solution.csv')
    submission = pd.read_csv('kaggle/baseline.csv')
    print(score(solution, submission, 'image_id'))

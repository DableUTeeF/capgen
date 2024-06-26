import pandas as pd
import pandas.api.types
import nltk
import sentencepiece as spm
import json
import sacrebleu

sp = spm.SentencePieceProcessor()
sp.Load('/mnt/d/work/sacrebleu_tokenizer_spm.model')


def tokenize_gt(row):
    captions = json.loads(row.captions)
    print(captions, row)
    outputs = [" ".join(sp.EncodeAsPieces(cap)) for cap in captions]
    return outputs


if __name__ == '__main__':
    sol = pd.DataFrame({'id': [0, 1], 'captions': [json.dumps(['ลาไว้ 3 คนครับ', 'วันนี้']), json.dumps(['ลาไว้ 3 คนครับ', 'วันนี้'])]})
    sub = pd.DataFrame({'id': [0, 1], 'captions': [json.dumps('3 คน'), json.dumps('วันนี้')]})
    sol.apply(tokenize_gt, axis=1)

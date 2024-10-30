import pandas as pd
import json
import nltk
from transformers import AutoTokenizer
import evaluate
from sacrebleu.tokenizers.tokenizer_spm import TokenizerSPM
import sacrebleu
import os

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


if __name__ == '__main__':
    s = TokenizerSPM('flores101')
    # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    hf_sacrebleu = evaluate.load("sacrebleu")
    origins = json.load(open('/media/palm/data/coco/annotations/captions_val2017.json'))
    gt = {}
    json_file = json.load(open('/media/palm/BiggerData/capgen/annotations/capgen_v0.3_coco.json'))
    for ann in origins['annotations']:
        image_id = f"{ann['image_id']:012d}.jpg"
        if image_id not in gt:
            gt[image_id] = []
        gt[image_id].append(ann['caption'])

    ss = '/media/palm/BiggerData/capgen/results'
    # for filename in os.listdir(ss):
    for filename in sorted(os.listdir(ss)):
        if '_eng' not in filename:
            continue
        dic = json.load(open(os.path.join(ss, filename)))
        print('\n' + filename)
        labels = []
        preds = []
        for key in gt:
            if key not in dic:
                continue
            if len(gt[key]) < 2:
                continue
            elif len(gt[key]) > 2:
                gt[key] = gt[key][:2]
            labels.append(gt[key])
            # text = ' '.join([tokenizer.decode(x, skip_special_tokens=True) for x in tokenizer(dic[f'{key:012d}.jpg'][0]['generated_text'])['input_ids']])
            text = dic[key]
            preds.append(text)
        results = sacrebleu.corpus_bleu(preds, labels)
        print('sacrebleu_ori')
        print(results)
        results = hf_sacrebleu.compute(predictions=preds,
                                       references=labels)
        print('sacrebleu_hf')
        print(results)
        rouge_result = rouge.compute(predictions=preds,
                                     references=labels,
                                     use_stemmer=True)
        print('rouge_result')
        print(rouge_result)
        meteor_result = meteor.compute(predictions=preds,
                                       references=labels)
        print('meteor')
        print(meteor_result)


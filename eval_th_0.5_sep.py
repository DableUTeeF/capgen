import json
import nltk
from transformers import AutoTokenizer
import evaluate
from sacrebleu.tokenizers.tokenizer_spm import TokenizerSPM
import sacrebleu
import os
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import language_evaluation
import pandas as pd


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def tokenize(text):
    return ' '.join([tokenizer.decode(x) for x in tokenizer(text)['input_ids'][1:-1]])


if __name__ == '__main__':
    # s = TokenizerSPM('flores101')
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    evaluator = language_evaluation.CocoEvaluator(verbose=False, coco_types=["CIDEr"], )
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    hf_sacrebleu = evaluate.load("sacrebleu")
    gt = {}
    json_file = json.load(open('/mnt/d/work/capgen/capgen_v0.5/test.json'))
    for key in json_file:
        k = os.path.basename(key)+'.jpg'
        gt[k] = []
        for ann in json_file[key]:
            gt[k].append(tokenize(ann))

    outputs = {'filename': [], 'bleu': [], 'rougeL': [], 'meteor': [], 'cider': [], 'sacrebleu': [], 'sentence_sacrebleu': [],
               # 'preds': [], 'labels': [],
               'utokens_pd': []
               }
    ss = 'results-v0.5'
    for filename in os.listdir(ss):
        if 'eng' in filename or 'ling' in filename:
            continue

        dic = json.load(open(os.path.join(ss, filename)))
        print('\n' + filename)
        labels = []
        temp_label = []
        preds = []
        sentence_sacrebleu = []
        num = 0
        utokens_pd = set()
        for key in gt:
            num += 1
            # if num > 5:
            #     continue
            if filename not in ['phayathaibertnocross_2_1e-05_thai.json', 'blip-large.json']:
                continue
            if key not in dic:
                continue
            if len(gt[key]) < 2:
                continue
            elif len(gt[key]) > 2:
                gt[key] = gt[key][:2]
            labels.append(gt[key])
            temp_label.append(gt[key][0])
            # text = ' '.join([tokenizer.decode(x, skip_special_tokens=True) for x in tokenizer(dic[f'{key:012d}.jpg'][0]['generated_text'])['input_ids']])
            if 'zeroshot' in filename:
                dic[key] = dic[key][0]
            text = tokenize(dic[key])
            utokens_pd.update(text.split(' '))
            preds.append(text)
            results = sacrebleu.sentence_bleu(text, gt[key])
            sentence_sacrebleu.append(results.score)
            # print(key)

        # outputs['preds'].append([preds[i].replace(' ', '') for i in range(len(preds))])
        # outputs['labels'].append([[label.replace(' ', '') for label in labels[i]] for i in range(len(labels))])
        outputs['utokens_pd'].append(len(utokens_pd))
        outputs['filename'].append(filename)
        outputs['sentence_sacrebleu'].append(np.mean(sentence_sacrebleu))
        results = sacrebleu.corpus_bleu(preds, labels)
        outputs['sacrebleu'].append(results.score)
        print('sacrebleu_ori')
        print(results)
        results = hf_sacrebleu.compute(predictions=preds,
                                       references=labels)
        print('sacrebleu_hf')
        print(results)
        outputs['bleu'].append(results['score'])

        rouge_result = rouge.compute(predictions=preds,
                                     references=labels,
                                     use_stemmer=True)
        print('rouge_result')
        print(rouge_result)
        outputs['rougeL'].append(rouge_result['rougeL'])

        meteor_result = meteor.compute(predictions=preds,
                                       references=labels)
        print('meteor')
        print(meteor_result)
        outputs['meteor'].append(meteor_result['meteor'])

        results = evaluator.run_evaluation(preds, labels)
        print(results)
        # results = evaluator.run_evaluation(temp_label, labels)
        # print(results)
        # raise Exception()
        outputs['cider'].append(results['CIDEr'])
    df = pd.DataFrame(outputs)
    df.to_csv('csvs/results-v0.5_4.csv')

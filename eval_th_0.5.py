import json
from transformers import AutoTokenizer
import evaluate
import nltk
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
    old = pd.read_csv('csvs/results-v0.5_5.csv')
    filenames = list(old['filename'])
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    evaluator = language_evaluation.CocoEvaluator(verbose=False, coco_types=["CIDEr"], )
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    hf_sacrebleu = evaluate.load("sacrebleu")
    gt = {}
    json_file = json.load(open('/mnt/d/work/capgen/capgen_v0.5/test.json'))
    for key in json_file:
        k = os.path.basename(key) + '.jpg'
        gt[k] = []
        for ann in json_file[key]:
            gt[k].append(tokenize(ann))

    outputs = {'filename': [], 'bleu': [], 'sacrebleu': [], 'sentence_sacrebleu': [],
               # 'bleu2': [], 'bleu4': [],
               'rougeL': [], 'meteor': [], 'cider': [], 'utokens_pd': []}
    ss = 'results-v0.5'
    for filename in os.listdir(ss):
        # if filename not in ['blip_frozen_Meta-Llama-3-8B-Instruct_blip2-opt-2.7b-coco.json']:
        #     continue
        if 'resnet' not in filename:
            continue
        if 'eng' in filename or 'ling' in filename:
            continue
        dic = json.load(open(os.path.join(ss, filename)))
        print('\n' + filename)
        labels = []
        temp_label = []
        preds = []
        sentence_sacrebleu = []
        bleu2 = []
        bleu4 = []
        utokens_pd = set()
        for key in gt:
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
            preds.append(text)
            results = sacrebleu.sentence_bleu(text, gt[key])
            sentence_sacrebleu.append(results.score)
            utokens_pd.update(text.split(' '))
            # BLEUscore = nltk.translate.bleu_score.sentence_bleu(
            #     gt[key],
            #     text,
            #     weights=[
            #         (1. / 2., 1. / 2.),
            #         # (1. / 3., 1. / 3., 1. / 3.),
            #         # (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
            #     ]
            # )
            # bleu2.append(BLEUscore)

        outputs['utokens_pd'].append(len(utokens_pd))
        outputs['filename'].append(filename)
        outputs['sentence_sacrebleu'].append(np.mean(sentence_sacrebleu))
        # outputs['bleu2'].append(np.mean(bleu2))
        # BLEUscore = nltk.translate.bleu_score.sentence_bleu(
        #     labels,
        #     preds,
        #     weights=[
        #         # (1. / 2., 1. / 2.),
        #         # (1. / 3., 1. / 3., 1. / 3.),
        #         (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
        #     ]
        # )
        # outputs['bleu4'].append(BLEUscore)
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
        print('cider')
        results = evaluator.run_evaluation(preds, labels)
        print(results)
        # results = evaluator.run_evaluation(temp_label, labels)
        # print(results)
        # raise Exception()
        outputs['cider'].append(results['CIDEr'])
        df = pd.DataFrame(outputs)
        df.to_csv('csvs/results-v0.5_7.csv')

from datasets import load_dataset
import json
import os
from pyinflect import getInflection
import numpy as np

import collections.abc

class OrderedSet(collections.abc.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

def alter_eng(
        sentence,
        nlp,
        alter_lemma,
        p
):
    doc = nlp(sentence.lower())

    alt_idx = []
    doc_out = []
    for i, token in enumerate(doc):
        doc_out.append(token.text)
        if token.pos_ in ["NOUN", "VERB", "ADP"]:
            alt_idx.append(i)
    n = int(p * len(alt_idx))
    idx = np.random.choice(np.array(alt_idx), size=n, replace=False)
    for i in idx:
        token = doc[i]
        while True:
            t = np.random.choice(alter_lemma[token.pos_], size=1)[0]
            if t != token.lemma_:
                break
        inflect_t = getInflection(t, tag=token.tag_, inflect_oov=True)
        if inflect_t is None:
            inflect_t = t  # no inflection
        else:
            inflect_t = inflect_t[0]
        doc_out[i] = inflect_t
    return " ".join(doc_out)


class KarpathyDataset:
    def __init__(
            self,
            coco_json,
            image_dir
    ):
        self.image_map = {}
        for file in os.listdir(os.path.join(image_dir, 'train2017')):
            self.image_map[file] = 'train2017'
        for file in os.listdir(os.path.join(image_dir, 'val2017')):
            self.image_map[file] = 'val2017'
        self.data = []
        json_file = json.load(open(coco_json))
        for ann in json_file:
            ann = json_file[ann]
            filename = f"{ann['cocoid']:012d}.jpg"
            for sent in ann['sentences']:
                self.data.append((
                    os.path.join(
                        image_dir,
                        self.image_map[filename],
                        filename
                    ),
                    sent
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, text = self.data[idx]
        return image, text


if __name__ == '__main__':
    import spacy
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    nlp = spacy.load('en_core_web_sm')
    alter_lemma = {
        "NOUN": ['cat', 'dog', 'man'],
        "VERB": ['sit', 'run', 'walk'],
        "ADP": ['on', 'under']
    }
    text1 = "a cat rests on a dog's back as he lies on the sidewalk"
    text2 = alter_eng(
        text1,
        nlp,
        alter_lemma,
        0.5
    )
    print()

    # dset = KarpathyDataset(
    #     '/media/palm/BiggerData/capgen/annotations/karpathy_train.json',
    #     '/media/palm/data/coco/images'
    # )
    #
    # dataset = load_dataset("yerevann/coco-karpathy")
    # train = json.load(open('/media/palm/data/coco/annotations/captions_train2017.json'))
    # val = json.load(open('/media/palm/data/coco/annotations/captions_val2017.json'))
    #
    # k_test = set()
    # k_val = set()
    # c_train = set()
    # c_val = set()
    #
    # karpathy_train = {}
    # karpathy_restval = {}
    # karpathy_val = {}
    # karpathy_test = {}
    #
    # for d in dataset['train']:
    #     karpathy_train[d['cocoid']] = d
    # for d in dataset['validation']:
    #     karpathy_val[d['cocoid']] = d
    # for d in dataset['test']:
    #     karpathy_test[d['cocoid']] = d
    # for d in dataset['restval']:
    #     karpathy_restval[d['cocoid']] = d
    #
    # dst = '/media/palm/BiggerData/capgen/annotations'
    # json.dump(karpathy_train, open(f'{dst}/karpathy_train.json', 'w'))
    # json.dump(karpathy_restval, open(f'{dst}/karpathy_restval.json', 'w'))
    # json.dump(karpathy_val, open(f'{dst}/karpathy_val.json', 'w'))
    # json.dump(karpathy_test, open(f'{dst}/karpathy_test.json', 'w'))

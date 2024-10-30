from data import alter_eng
import spacy
from transformers import AutoTokenizer
from orderedset import OrderedSet
import numpy as np


def al1(a, b):
    diff_b = OrderedSet(b) - OrderedSet(a)
    diff_a = OrderedSet(a) - OrderedSet(b)

    no_index = []
    num_del = 0
    index = 0
    while b and a:
        # print(f'For {b=}')
        # print(f'For {a=}')
        if b[index] not in diff_b:
            if a[index] == b[index]:
                # print('del', a[index], b[index])
                del a[index], b[index]
                num_del += 1
                # print(f'{a=}')
                # print(f'{b=}')
            else:
                intersec = list(OrderedSet(a) & OrderedSet(b))
                # print(f'{list(intersec)=}')
                while intersec[0] != b[index]:
                    del b[index]
                    no_index.append(index + num_del)
                    num_del += 1
                while intersec[0] != a[index]:
                    del a[index]
                # print(f'{a=}')
                # print(f'{b=}')
            diff_b = OrderedSet(b) - OrderedSet(a)
            diff_a = OrderedSet(a) - OrderedSet(b)
        else:
            # print(b[index], index)
            no_index.append(index + num_del)
            del b[index]
            num_del += 1
            while a and (a[index] in diff_a):
                # print('del', a[index])
                del a[index]

    return no_index


def al2(codec_a, codec_b):
    diffs = []
    ai = 0
    bi = 0
    while bi < len(codec_b) - 1:
        token_a = codec_a[ai]
        token_b = codec_b[bi]
        if token_a == token_b:
            ai += 1
            bi += 1
            continue
        if codec_a[ai] == codec_b[bi+1]:
            diffs.append(bi)
            bi += 1
            continue
        if codec_a[ai+1] == codec_b[bi]:
            diffs.append(bi)
            ai += 1
            continue
        diffs.append(bi)
        ai += 1
        bi += 1
    if codec_a[ai] != codec_b[bi]:
        diffs.append(bi)
    return diffs


def al3(a, b):
    intersec = list(OrderedSet(a) & OrderedSet(b))
    index = 0
    num_del = 0
    no_index = []
    while b and a:
        if not intersec or b[index] != intersec[0]:
            del b[index]
            no_index.append(index + num_del)
            num_del += 1

        if not intersec or a[index] != intersec[0]:
            del a[index]

        if a and (a[index] == b[index]):
            del a[index], b[index]
            num_del += 1
        intersec = list(OrderedSet(a) & OrderedSet(b))
    return no_index


def al4(a, b):
    intersec = list(OrderedSet(a) & OrderedSet(b))
    index = 0
    num_del = 0
    no_index = []
    while b and a:
        try:
            ranks = [b.index(i) + a.index(i) for i in intersec]
            min_intersec = intersec[np.argmin(ranks)]
            intersec[0] = min_intersec
        except:
            pass
        if not intersec or b[index] != intersec[0]:
            del b[index]
            no_index.append(index + num_del)
            num_del += 1
        if not intersec or a[index] != intersec[0]:
            del a[index]
        if a and (a[index] == b[index]):
            del a[index], b[index]
            num_del += 1
        intersec = list(OrderedSet(a) & OrderedSet(b))
    return no_index


if __name__ == '__main__':
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

    # codec_a = tokenizer(text1).input_ids
    # codec_b = tokenizer(text2).input_ids
    codec_a = [5, 10, 3, 33897, 25064, 27616, 42480, 102418, 110, 25053, 26135, 2877, 11681, 110, 35814, 27952, 27616, 42321, 11]
    codec_b = [5, 25005, 33897, 25064, 25297, 42480, 102418, 110, 25986, 25026, 12413, 35814, 27952, 25026, 12413, 42321, 10, 11]


    a1 = al1(list(tuple(codec_a)), list(tuple(codec_b)))
    a2 = al2(list(tuple(codec_a)), list(tuple(codec_b)))
    a3 = al4(list(tuple(codec_a)), list(tuple(codec_b)))
    print(a1, a2, a3)
    if a2 != a3:
        print(codec_a)
        print(codec_b)

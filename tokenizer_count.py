from attacut import tokenize, Tokenizer
import json
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # lengths = {}
    # for stage in ['train', 'val', 'test']:
    #     lengths[stage] = []
    #     output = {}
    #     data = json.load(open(f'/media/palm/Data/capgen/capgen_v0.5/annotations/{stage}.json'))
    #     for key, values in data.items():
    #         output[key] = []
    #         for caption in values:
    #             tokens = tokenize(caption)
    #             lengths[stage].append(len(tokens))
    #             output[key].append(tokens)
    #     json.dump(output, open(f'/media/palm/Data/capgen/capgen_v0.5/annotations/{stage}_attacut.json', 'w'))
    # json.dump(lengths, open(f'/media/palm/Data/capgen/capgen_v0.5/annotations/lengths.json', 'w'))
    # words = tokenize('คนสวมเสื้อแขนยาวสีดำสวมกางเกงขายาวสีน้ำเงินกำลังยืนอยู่บนสกีหิมะ')
    # print(words)
    lengths = json.load(open('/media/palm/Data/capgen/capgen_v0.5/annotations/lengths.json'))
    plt.title('Train')
    plt.hist(np.clip(lengths['train'], 0, 50), bins=10)
    plt.figure()
    plt.title('Val')
    plt.hist(np.clip(lengths['val'], 0, 50), bins=10)
    plt.figure()
    plt.title('Test')
    plt.hist(np.clip(lengths['test'], 0, 50), bins=10)
    plt.show()

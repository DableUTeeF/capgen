import os
import json
import pandas as pd


if __name__ == '__main__':
    cul = json.load(open('results-kaggle/mgpt_scratch_2.json'))
    coc = json.load(open('results-kaggle/mgpt_scratch_2-coco.json'))

    sam = pd.read_csv('kaggle/sample_submission.csv')
    sam = sam.fillna('')
    outputs = []
    for i, row in sam.iterrows():
        if row['caption'] != '':
            outputs.append(row['caption'])
        else:
            ids = (row['image_id'] + '.jpg').replace('food', '').replace('travel', '').replace('//', '/')
            if ids in cul:
                outputs.append(cul[ids])
            elif row['image_id'] + '.jpg' in coc:
                outputs.append(coc[row['image_id'] + '.jpg'])
            else:
                raise Exception('WTF')
    sam['caption'] = outputs
    sam.to_csv('kaggle/baseline.csv', index=False)

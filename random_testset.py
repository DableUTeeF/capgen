import pandas as pd
import uuid
from PIL import Image
import os
import json


if __name__ == '__main__':
    srcs = {
        'coco': '/mnt/d/work/coco/images',
        'cult': '/mnt/d/work/capgen/capgen_v0.4',
    }
    dst = '/mnt/d/work/capgen/capgen_v0.5/test'
    df = pd.read_csv('kaggle/sample_submission.csv')
    mapping = {}
    for i, row in df.iterrows():
        name = str(uuid.uuid4())
        while name in mapping:
            name = str(uuid.uuid4())
        mapping[name] = row['image_id']
        if row['image_id'].startswith('test2017'):
            src = srcs['coco']
        else:
            src = srcs['cult']
        im = Image.open(os.path.join(src, row['image_id']+'.jpg'))
        im.save(os.path.join(dst, name+'.jpg'))
    json.dump(mapping, open('/mnt/d/work/capgen/capgen_v0.5/mapping.json', 'w'))


import json
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    src = '/mnt/d/work/capgen/capgen_v0.3'
    dst = '/mnt/d/work/capgen/capgen_v0.4'

    train_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_train.json'))
    train_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_train.json'))
    test_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_test.json'))
    test_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_test.json'))
    val_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_val.json'))
    val_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_val.json'))

    # os.makedirs(os.path.join(dst, 'val/travel'), exist_ok=True)
    # os.makedirs(os.path.join(dst, 'val/food'), exist_ok=True)
    # val_map = {}
    # for k03, v03 in val_03.items():
    #     for k04, v04 in val_04.items():
    #         if v03 == v04:
    #             val_map[k03] = k04
    #             im = Image.open(os.path.join(src, k03))
    #             im.save(os.path.join(dst, k04.replace('png', 'jpg')))
    #             break

    # os.makedirs(os.path.join(dst, 'test/travel'), exist_ok=True)
    # os.makedirs(os.path.join(dst, 'test/food'), exist_ok=True)
    test_map = {}
    for k03, v03 in test_03.items():
        for k04, v04 in test_04.items():
            if v03 == v04:
                if not k03.endswith('png'):
                    k03 = k03 + '.jpg'
                test_map[k03] = k04.replace('.png', '').replace('.jpg', '')
    #             im = Image.open(os.path.join(src, k03)).convert('RGB')
    #             im.save(os.path.join(dst, k04.replace('png', 'jpg')))
    #             break
    json.dump(test_map, open('/mnt/d/work/capgen/capgen_v0.4/annotations/v0.3-v0.4_test_map.json', 'w'))

    # os.makedirs(os.path.join(dst, 'train/travel'), exist_ok=True)
    # os.makedirs(os.path.join(dst, 'train/food'), exist_ok=True)
    # for k03, v03 in train_03.items():
    #     for k04, v04 in train_04.items():
    #         if v03 == v04:
    #             im = Image.open(os.path.join(src, k03)).convert('RGB')
    #             im.save(os.path.join(dst, k04.replace('png', 'jpg')))
    #             break

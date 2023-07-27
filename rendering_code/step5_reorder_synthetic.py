import os
import re
import shutil
import tqdm

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


anno_path = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/xinchuan_200w/oriannot/'
outanno_path = '/mnt/workspace/dataset/interhand_5fps/interhand_data/train/xinchuan_200w/order_oriannot/'
img_path = '/mnt/workspace/workgroup/lijun/hand_dataset/xinchuan_200wimg/'
outimg_path = '/mnt/workspace/workgroup/lijun/hand_dataset/order_xinchuan_200wimg/'
annots = natural_sort(os.listdir(anno_path))
imgnames = natural_sort(os.listdir(img_path))
print('len annot', len(annots), flush=True)### 597611
ori = 366368 #1015851
start = 366358 #1015851

for idx in tqdm.tqdm(range(0, 1020000)):
    try:
        oneanno = annots[idx]
    except:
        continue
    cur_num = oneanno.split('.pkl')[0]

    imgname = imgnames[idx]
    img_num = imgname.split('.png')[0].split('0_')[1]
    assert cur_num == img_num

    outannot = str(start) + '.pkl'
    outimg = '0_' + str(start) + '.png'
    shutil.copy(anno_path + oneanno, outanno_path + outannot)
    shutil.copy(img_path + imgname, outimg_path + outimg)
    start = idx + 1 + ori
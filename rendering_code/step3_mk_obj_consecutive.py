import os
import re
import shutil
import tqdm

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

mesh_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/camera_pose/'
annot_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/oriannot/'
outanno_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/nodup_annot/'
outmesh_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/nodup_pose/'
meshs = natural_sort(os.listdir(mesh_path))
annots = natural_sort(os.listdir(annot_path))
print('len annot', len(annots), flush=True)### 649493
ori = 366358
start = 366358

print(' start idx ', start + 0, flush=True)
print(' ebnd idx ', start + 1245000, flush=True)
error_file = open('error.txt', 'w')
valid_idx = 0
for idx in tqdm.tqdm(range(0, 1247260)):
    start = ori + valid_idx#idx
    oneanno = annots[idx]
    cur_num = oneanno.split('.pkl')[0]
    cur_meshleft = str(cur_num) + '_left.obj'
    if not os.path.exists(mesh_path + cur_meshleft):
        error_file.write(str(cur_num))
        error_file.write('\n')
        continue
    cur_meshright = str(cur_num) + '_right.obj'
    outannot = str(start) + '.pkl'
    outmesh_left = str(start) + '_left.obj'
    outmesh_right = str(start) + '_right.obj'
    shutil.copy(annot_path + oneanno, outanno_path + outannot)
    shutil.copy(mesh_path + cur_meshleft, outmesh_path + outmesh_left)
    shutil.copy(mesh_path + cur_meshright, outmesh_path + outmesh_right)
    valid_idx += 1
    # start = idx + 1 + ori
error_file.close()
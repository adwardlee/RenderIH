import cv2
import os
dir_path = '/home/tallery/code/Hand/ego3d/test/seq_00000'
sub_dir_list = os.listdir(dir_path)
sub_dir_list.sort()
for sub_dir_name in sub_dir_list:
    name = os.path.join(dir_path, sub_dir_name)
    pic_name = os.path.join(name, 'color.png')
    pic = cv2.imread(pic_name)
    cv2.imshow('1', pic)
    cv2.waitKey(10)
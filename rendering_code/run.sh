for((i=999815;i<=1006358;i++));
do /mnt/workspace/software/blender/blender-3.2.2-linux-x64/blender -b '0826.blend' -P 'load_mano_diffbg.py' -noaudio -o "/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/render_img/#_$i" -f 0 -F 'PNG' "$i";
done

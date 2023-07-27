## Steps to generate synthetic data
1.get augmented pose from X
2.step2 remove duplicate pose and select camera view
3. make filename consecutive
4.use load_mano_diffbg to generate imgs
5.reorder generate img and step1 annot
6. change image size and annotation-camera to 256x256
7. generate hms/mask/dense or gen ori_handdict

As I provide the cropped imgs and annotations, we directly run step 7 to generate ori_handdict. 
put imgs to train/color_img/ directory and annotations to train/color_annot/ directory, change `data_path` to the right path

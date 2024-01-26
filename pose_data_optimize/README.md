# USAGE
 We provide the template pose data in ```data/template_renderih_data_sample.pkl```, which is the an annotation file in RenderIH dataset.
 
 The processing pipeline consists of several Python scripts that should be executed in sequence. Follow the steps below to process your pose data:

1. `python template_0_pose_convert.py` to convert the original pose representation.
2. `python template_1_pose_argue.py`  to do the pose augmentation.
3. `python batch_optimize_mocap_origin.py --vis` to do the main optimization.
4. `python template_3_pose_reconvert.py` to convert the optimized pose to original representation.


 
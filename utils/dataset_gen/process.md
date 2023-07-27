## Process Dataset

1. process tzionas dataset. Download from `https://files.is.tue.mpg.de/dtzionas/Hand-Object-Capture/` 01~07, then 
`python utils/dataset_gen/tzionas_generation.py`
2. process h2o3d dataset. Download hand data from `https://www.tugraz.at/index.php?id=57823`, object data from
`https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing`, change hand_data, object_data path to the dataset
then `python utils/dataset_gen/h2o3d_dataloader.py`
3. process ego3d dataset. Download data from `https://github.com/AlextheEngineer/Ego3DHands`, change path, then run `python utils/dataset_gen/ego3dhand_dataloader.py`
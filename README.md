# Requirement
```
pip install -r requirement.txt,
```

# evaluation
HPatches Sequences / Image Pairs Matching Benchmark

1.Download the HPatches dataset：

```
cd evaluation_hpatch/hpatches_sequences
bash download.sh
```
2.Extract local descriptors：
```
cd evaluation_hpatch
CUDA_VISIBLE_DEVICES=0 python export.py  --tag [Descriptor_suffix_name] 
```
3.Evaluation
```
cd evaluation_hpatch/hpatches_sequences
jupyter-notebook
```
run HPatches-Sequences-Matching-Benchmark.ipynb


## Training
Set the dataset path in the configuration file configs/MTLDesc_train.yaml

```
mega_image_dir:  /data/Mega_train/image   #images
mega_keypoint_dir:  /data/Mega_train/keypoint #keypoints
mega_despoint_dir:  /data/Mega_train/despoint #descriptor correspondence points
```
```
python train.py --gpus 0 --configs configs/MTLDesc_train.yaml --indicator mtldesc_0
```

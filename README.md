# Requirement
Pytorch版本比较关键。
```
Pytorch == 1.2.0
OpenCV >= 3.4
numpy == 3.6
```

# 快速开始
HPatches Sequences / Image Pairs Matching Benchmark

1.下载HPatches数据集：

```
cd evaluation_hpatch/hpatches_sequences
bash download.sh
```
2.提取描述子：
```
cd evaluation_hpatch
CUDA_VISIBLE_DEVICES=0 python export.py  --tag [Descriptor_suffix_name] 
```
3.测评
对比方法已保存在：evaluation_hpatch/hpatches_sequences/cache 目录下
启动jupyter
```
cd evaluation_hpatch/hpatches_sequences
jupyter-notebook
```
运行HPatches-Sequences-Matching-Benchmark.ipynb


## 训练模型
在配置文件configs/MTLDesc_train.yaml中设置数据集路径

```
mega_image_dir:  /data/Mega_train/image   #图片
mega_keypoint_dir:  /data/Mega_train/keypoint #关键点
mega_despoint_dir:  /data/Mega_train/despoint #描述子
```
```
python train.py --gpus 0 --configs configs/MTLDesc_train.yaml --indicator mtldesc_0
```
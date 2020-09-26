# Requirement
```
Pytorch == 1.2.0
OpenCV >= 3.4
numpy == 3.6
```

# 训练MagicPoint
## 构造合成数据集
```
python synthetic_dataset_generation.py \ 
--data_root=/path/to/the/synthesized_dataset
```

## 训练MagicPoint
```
# 模型默认存储在/project_root/magicpoint_ckpt/[prefix]中，log文件默认存储于/project_root/magcipoint_log/[prefix]中，prefix在命令行中指定
python magicpoint_synthetic_main.py \
--synthetic_dataset_dir=/path/to/the/synthesized_dataset \
--prefix=synthetic
```

# 准备COCO原始数据
从 http://cocodataset.org/#download 下载COCO原始数据，并整理成如下结构：  
```
coco
 -train2014
   --images
 -val2014
   --images
```
# 准备MegDepth数据集
根据D2-Net https://github.com/mihaidusmanu/d2-net 下载相关MegDepth的数据集，整理成如下结构：
```
MegaDepthOrder
 -phoenix
 -scene_info
 -Undistored_SfM
 -train_scenes.txt # 来自d2net
 -val_scenes.txt # 来自d2net
```

# 标注COCO及MegaDepth数据集的伪标签  
一共进行两轮标签标注,第二轮才标注MegaDepth数据集：  
第一轮：由在合成数据集上训练的magicpoint进行标注，得到[COCO+第一轮伪标签]  
```
# 第一次标注COCO,在COCO文件夹下得到标注的点文件夹
python adaption_dataset_generation.py \
--ckpt_file=/path/to/ckpt_of_magicpoint_trained_on_synthesized_data \
--dataset_type=coco \
--dataset_dir=/path/to/the/coco_dataset \
--output_root=/path/to/save/the/labeled_coco_dataset_0 \
--detection_threshold=0.005 
```

第二轮：  
首先用[COCO+第一轮伪标签]重新训练magicpoint
```
# 利用第一轮标注的COCO数据重新训练magicpoint
# 模型默认存储在/project_root/magicpoint_ckpt/[prefix]中，log文件默认存储于/project_root/magcipoint_log/[prefix]中，prefix在命令行中指定
python magicpoint_coco_main.py \
--coco_dataset_dir=/path/to/the/coco_dataset \
--prefix=coco
```
然后用新的magicpoint对COCO数据进行第二轮标注，该次得到的标注数据作为MegPoint的训练数据  
```
# 第二次标注COCO,在COCO文件夹下得到第二次标注的点文件夹
python adaption_dataset_generation.py \
--ckpt_file=/path/to/magicpoint_trained_on_coco_data \
--dataset_type=coco \
--dataset_dir=/path/to/the/coco_dataset \
--output_root=/path/to/save/the/labeled_coco_dataset_1 \
--detection_threshold=0.04 
```

在标注MegaDepth之前，首先要对MegaDepth进行预处理得到可以直接输入的图像对以及图像对上的点对（用于训练描述子）。
```
# 预处理MegaDepth数据集，得到preprocessed_train_dataset，包含图像对及对应点对
python create_megadepth_dataset.py \
--base_path=/path/to/MegaDepthOrder \
--secen_info_path=/path/to/MegaDepthOrder/scene_info \
```

然后标注MegaDepth数据集
```
python adaption_dataset_generation.py \
--ckpt_file=/path/to/magicpoint_trained_on_coco_data \
--dataset_type=megadepth \
--dataset_dir=/path/to/the/MegaDepthOrder/preprocessed_train_dataset \
--output_root=/path/to/the/MegaDepthOrder/preprocessed_train_label \
--detection_threshold=0.04 
```


# 训练MegPoint
利用[COCO+MegaDepth]训练MegPoint
```
# 模型默认存储在/project_root/megpoint_ckpt/[prefix]中，log文件默认存储于/project_root/megpoint_log/[prefix]中，prefix在命令行中指定
python megpoint_main.py 
--gpus=1 \
--dataset_type=megacoco \
--dataset_dir=/path/to/coco/train2014 \
--megadepth_dataset_dir=/path/to/MegaDepthOrder/preprocessed_train_dataset \
--megadepth_label_dir=/path/to/MegaDepthOrder/preprocessed_train_label \
--model_type=SuperPointBackbone128 \
--batch_size=24 \
--adjust_lr=True \
--epoch_num=30 \
--prefix=megacoco_megpoint128
```


# 预训练的模型
```
magicpoint_synthesized.pt 在合成数据上预训练的模型
magicpoint_coco.pt 用第一次标注的结果训练的模型
megpoint.pt + megpoint_extractor.pt 用coco+megadepth训练的模型
```

# 在HPatches上测试MegPoint
HPatches Sequences / Image Pairs Matching Benchmark
1.下载HPatches数据集：

```
cd evaluation_hpatch/hpatches_sequences
bash download.sh
```
2.在HPatches上提取MegPoint特征：

```python
CUDA_VISIBLE_DEVICES=0 python export_hpatch_prediction.py  --tag [Descriptor_suffix_name] --top-k [key_point_numbers]
例如：
CUDA_VISIBLE_DEVICES=1 python export_hpatch_prediction.py  --tag megpoint_10k_ms1 --top-k 10000 --max-scale 1
```
3.在Benchmark上评测：

```
#对比方法已保存在：
evaluation_hpatch/hpatches_sequences/cache 目录下
#启动jupyter
jupyter-notebook
#进入evaluation_hpatch/hpatches_sequences
#运行HPatches-Sequences-Matching-Benchmark.ipynb
```
![avatar](imgs/hpatch1.jpg)
![avatar](imgs/hpatch2.jpg)

# 在Aachen上测试MegPoint

1.准备Aachen数据集：

This code currently supports **only the Aachen Day-Night** dataset. Further datasets might be supported in the future. For the dataset, we provide two files `database.db` and `image_pairs_to_match.txt`, which are in the `data/aachen-day-night/` sub-directory of this repository. You will need to move them to directory where you are storing the Aachen Day-Night dataset. In order for the script to function properly, the directory should have the following structure:

```
.
├── database.db
├── image_pairs_to_match.txt
├── images
│  └── images_upright
├── 3D-models
│  ├── database_intrinsics.txt
│  └── aachen_cvpr2018_db.nvm
└── queries/night_time_queries_with_intrinsics.txt
```

2.在Aachen上提取MegPoint特征：

```
cd evaluation_aachen
CUDA_VISIBLE_DEVICES=0 python export_predictions_evaluation.py --config=./megpoint128.yaml --output_root=/data/localization/aachen/Aachen-Day-Night --export_name=megpoint/aachen --weights=../pretrained_models --tag megpoint
--config是配置文件
--output_root是输出文件的根目录路径，--export_name是子目录路径，这里输提取的特征点和描述子将以.megpoint(--tag)格式储存在/data/localization/aachen/Aachen-Day-Night/megpoint/aachen
```

3.使用官方pipeline在Aachen Benchmark上评测：

```
python reconstruction_pipelines.py --dataset_path /data/localization/aachen/Aachen-Day-Night --colmap_path  /usr/local/bin --method_name bb10  --feature_path  /data/localization/aachen/Aachen-Day-Night/megpoint/aachen
```

4.使用加OANet的pipeline在Aachen Benchmark上评测：

```
python oanet_pipelines.py --dataset_path /data/localization/aachen/Aachen-Day-Night --colmap_path  /usr/local/bin --method_name megpoint --feature_path  /data/localization/aachen/Aachen-Day-Night/megpoint/aachen
```
5.上传Benchmark
https://www.visuallocalization.net/



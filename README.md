# Requirement
```
Pytorch >= 1.2.0
OpenCV >= 3.4
numpy == 3.6
```

# 训练MagicPoint
## 构造合成数据集
```
python synthetic_dataset_generation.py --data_root=/path/to/the/synthesized_dataset
```

## 训练MagicPoint
```
# 模型默认存储在/project_root/magicpoint_ckpt/[prefix]中，log文件默认存储于/project_root/magcipoint_log/[prefix]中，prefix在命令行中指定
python magicpoint_synthetic_main.py --synthetic_dataset_dir=/path/to/the/synthesized_dataset --prefix=synthetic
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

# 标注COCO数据集的伪标签  
一共进行两轮标签标注：  
第一轮：由在合成数据集上训练的magicpoint进行标注，得到[COCO+第一轮伪标签]  
```
# 标注好的数据存储在与原始COCO数据相同的位置
python adaption_dataset_generation.py --round=0 --first_ckpt_file=/path/to/magicpoint_trained_on_synthesized_data --coco_dataset_dir=/path/to/the/coco_dataset
```  

第二轮：  
首先用[COCO+第一轮伪标签]重新训练magicpoint
```
# 利用第一轮标注的COCO数据重新训练magicpoint
# 模型默认存储在/project_root/magicpoint_ckpt/[prefix]中，log文件默认存储于/project_root/magcipoint_log/[prefix]中，prefix在命令行中指定
python magicpoint_coco_main.py --coco_dataset_dir=/path/to/the/coco_dataset --prefix=coco
```  
然后用新的magicpoint对COCO数据进行第二轮标注，该次得到的标注数据作为MegPoint的训练数据  
```
# 标注好的数据存储在与原始COCO数据相同的位置
python adaption_dataset_generation.py --round=1 --second_ckpt_file=/path/to/magicpoint_trained_on_coco_data --coco_dataset_dir=/path/to/the/coco_dataset
```

# 训练MegPoint
利用[COCO+第二轮伪标签]训练MegPoint
```
# 模型默认存储在/project_root/megpoint_ckpt/[prefix]中，log文件默认存储于/project_root/megpoint_log/[prefix]中，prefix在命令行中指定
# hpatch_dataset_dir指定hpatches数据集的位置，hpatches数据集用来对每个epoch模型进行验证
python megpoint_main.py --dataset_dir=/path/to/coco_pseudo_label_1 --hpatch_dataset_dir=/path/to/hpatch_dataset --prefix=exp
```

# 在HPatches上测试MegPoint
```
# 训练好的MegPoint由两部分构成
# tmp_ckpt_file指定第一部分，extractor_ckpt_file指定第二部分
python megpoint_main.py --hpatch_dataset_dir=/path/to/hpatch_dataset  --tmp_ckpt_file=[project_root]/pretrained_models/megpoint.pt --extractor_ckpt_file=[project_root]/pretrained_models/megpoint_extractor.pt
```


    
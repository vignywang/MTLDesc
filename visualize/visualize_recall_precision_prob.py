# 
# Created by ZhangYuyang on 2019/8/20
#
import numpy as np
import matplotlib.pyplot as plt


test_data_dir = "/home/zhangyuyang/project/development/MegPoint/magicpoint_ckpt/good_results/adam_0.0010_64/model_59_test_data.npy"
test_data = np.load(test_data_dir)
recall = test_data[0, 1:-1]
precision = test_data[1, 1:-1]
prob = test_data[2, 1:-1]

tmp_idx = np.where(prob <= 0.15)
recall = recall[tmp_idx]
precision = precision[tmp_idx]
prob = prob[tmp_idx]
title = "model_59" + " on the synthetic validation dataset"

plt.figure(figsize=(10, 5))
x_ticks = np.arange(0, 1, 0.01)
y_ticks = np.arange(0, 1, 0.05)
plt.title(title)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel('probability threshold')
plt.plot(prob, recall, label='recall')
plt.plot(prob, precision, label='precision')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('./tmp.png')
# plt.show()






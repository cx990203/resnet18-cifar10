# resnet18-cifar10
最基础的resnet18进行cifar10数据集分类处理。</br>
工程文件说明：</br>
resnet.py--resnet18模型存放文件</br>
train.py--训练脚本，默认导入cifar-10数据集</br>
utils.py--工具函数存放脚本</br>
white_balance_val.py--白平衡测试脚本（应付作业用的），用于测试图像在经过色调调整后的推理准确率，以及再加上白平衡处理之后的准确率。当然由于cifar-10数据集图像像素较小，有一定效果但并不是很好。</br>

## white_balance_val.py文档说明
图像色调调整主要在HSV空间当中进行</br>
共使用五种常见白平衡方法对图像进行处理：均值、完美反射、灰度世界假设、基于图像分析的偏色检测及颜色校正、动态阈值法。</br>
白平衡源码放置于utils.py文件当中。</br>
算法源码摘自博客：https://blog.csdn.net/weixin_34910922/article/details/109106029（如有侵权，请联系删除）</br>


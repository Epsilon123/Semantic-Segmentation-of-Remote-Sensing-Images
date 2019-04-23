内容：

遥感图像的语义分割，分别使用Deeplab V3+(Xception 和mobilenet V2 backbone)和unet模型
Deeplab V3+模型代码来自https://github.com/Epsilon123/keras-deeplab-v3-plus
unet模型代码来自https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
数据来自Kaggle竞赛https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
可以根据使用需要自定义输入、输出、激活函数、网络层

环境：

Ubuntu 16.04
keras 2.15
tensorflow-gpu 1.10
cuda 9.0
opencv 3.4
tifffile
shapely 1.6

结果：

https://github.com/Epsilon123/zhu/blob/master/img/6120_2_2_10.png
https://github.com/Epsilon123/zhu/blob/master/img/6120_2_2raw.jpg
https://github.com/Epsilon123/zhu/blob/master/img/612022_x_0.png
https://github.com/Epsilon123/zhu/blob/master/img/612022_x_1.png
https://github.com/Epsilon123/zhu/blob/master/img/612022_x_2.png
https://github.com/Epsilon123/zhu/blob/master/img/612022_x_5.png

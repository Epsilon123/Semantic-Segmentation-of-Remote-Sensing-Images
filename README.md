# 内容：

遥感图像的语义分割，分别使用Deeplab V3+(Xception 和mobilenet V2 backbone)和unet模型

Deeplab V3+模型代码来自https://github.com/Epsilon123/keras-deeplab-v3-plus

unet模型代码来自https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras

数据来自Kaggle竞赛https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
翻墙注册，这里有数据，教程和代码，能搞懂unet就能理解所有的语义分割原理。

可以根据使用需要自定义输入、输出、激活函数、网络层

# 环境：

Ubuntu 16.04

keras 2.15

tensorflow-gpu 1.10

cuda 9.0

opencv 3.4

tifffile

shapely 1.6

# 结果：

label：

![10个类别](https://github.com/Epsilon123/zhu/blob/master/img/6120_2_2_10.png)

原图：

![原图](https://github.com/Epsilon123/zhu/blob/master/img/6120_2_2raw.jpg)

类别1：建筑物

![class 0：建筑物](https://github.com/Epsilon123/zhu/blob/master/img/612022_x_0.png)

类别2：道路

![class 1：道路](https://github.com/Epsilon123/zhu/blob/master/img/612022_x_1.png)

类别3：树木

![class 2：树](https://github.com/Epsilon123/zhu/blob/master/img/612022_x_2.png)

类别6：汽车

![class 5：汽车](https://github.com/Epsilon123/zhu/blob/master/img/612022_x_3.png)

github是随便写的，有问题可以邮箱联系我853569053@qq.com
kaggle账号注册需要翻墙，数据量很大需要翻墙下载。建议先实现unet，在了解整个实验流程原理以及代码的基础上，再实现deeplab v3+。此项目不再维护，不要用QQ联系。

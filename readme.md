# 自动驾驶疲劳检测挑战赛第三名方案

## 第一步：数据处理
先将数据解压放置xfdata文件夹下。然后使用code文件夹中的npy2jpg.py，将数据集还原为jpg数据且符合torch.dataset的要求。npy2jpg.py还将划分训练集和验证集，并生成均衡样本。最终，训练集‘sleepy‘和‘non-sleepy‘两类各有5500个样本，验证集则为两类各550个样本。生成的训练使用的数据集位于xfdata/sleepy/train下。
数据处理主要是将原本.npy类型数据，属于[-128,127]之前转为8位整型之后保存为jpg图片。

## 第二步：模型训练
使用code中的train.py，调用预训练的efficientnetb7进行训练，详细的训练参数和采用的数据增强在train.py上半部分。

## 第三步：输出预测
使用code中的test.py，加载保存在user_data/model_data下已经训练好的模型权重文件并进行TTA（Test Time Augmentation）。最终获得提交的csv，存放于prediction_result下。

## 文件介绍
code文件夹下放置了loader文件夹和pskd_loss.py，训练过程中调用了算法PSKD[论文的arxiv链接](https://arxiv.org/abs/2006.12000)，这是其依赖文件。train.py和test.py分别为训练和预测的文件。npy2jpg.py为数据处理文件。
prediction_result里放置了获得最好成绩的csv文件。
user_data里放置了预测使用的模型权重 自行下载解压之后放在指定目录下\user_data\model_data\efficientnet-b7_balanced_color_1e-5\efficientnet-b7\ckpt\
xfdata可放置赛题数据集（解压后的）。
requirements为选手的python环境。


### 本机环境：
显卡：nvidia 3060
python==3.10.9
cuda==11.8，cudnn==8.7.0
使用的pytorch低于2.0也可以运行

### 相关资源
最佳模型：链接：https://pan.baidu.com/s/1NhXYp0uq8HqV3bJKs4DAdg?pwd=1234 提取码：1234

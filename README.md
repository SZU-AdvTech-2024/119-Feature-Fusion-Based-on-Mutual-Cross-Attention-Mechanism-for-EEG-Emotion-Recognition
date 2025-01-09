##### 1 数据集

使用DEAP_MNE_preprocessing预处理和特征提取方法。访问[Google Drive](https://drive.google.com/drive/folders/1jRQRbRgTIZEDByQYz41CuoyzPe45hxHv )获取提取的特征。

##### 2 环境创建

```
conda env create -f environment.yml
conda activate MCA-EEG
```

##### 3 训练测试

mca_experiment.py 原模型训练
mca_experiment_PCA.py 改进模型训练
mca_validations.py 原模型验证
mca_10kfold.py 原模型10折交叉验证范式下训练验证
mca_10kfold_kforest_PCA.py 改进模型10折交叉验证范式下训练验证
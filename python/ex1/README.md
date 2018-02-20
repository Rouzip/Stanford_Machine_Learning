## Tips
1. numpy矩阵的拼接，hstack用于拼接横向，vstack用于拼接纵向，其本质都是对于concatenate的包装
2. 当拼接的矩阵缺少维度的时候，类似于matlab，使用`[:,None]`可以添加缺省的维度（更一般化的，使用reshape来对于其进行调整，但是需要注意和matlab不同，python行优先
3. 画3d图的时候需要注意值是否已经转置
4. 画3d图或者contour这样的图的时候，需要变成紧密的点，所以需要meshgrid来转换x轴和y轴的值
5. sum或者mean的时候需要对于行和列敏感，指定axis最好


## 感受

对于numpy进行了初步的尝试，对于机器学习算法从octave到python进行了迁移。python数据以行优先，在画图计算等时候需要进行注意。

线性回归是通过数据对于合适的theta来进行决策，可以进行拟合和分类。但是对于某些数据效果不太好，可以使用kernel或者别的算法。
## Tips
1. 散点图使用scatter，如果在一个图中画多个，应该建立subplot，然后在其中画多个图
2. 正则化第一个由于是偏置项，所以不需要使用正则化，在theta之中需要将其置0
3. contour是热力图，其中xyz之后的参数决定画多少层
4. np之中有loadtxt，scipy.io之中具有loadmat

## 感受

scipy中的优化算法有很多，可以使用minimize，也可以自己指定grad和J，来进行选择。
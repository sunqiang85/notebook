
# 统计学习方法概论笔记
## 统计学习三要素
- 模型
- 策略
- 算法

## 统计学习的特点
统计学习是计算机运用数据及统计方法提高性能的机器学习
>Herbert A. Simon. 如果一个系统能够通过执行某个过程改进它的性能，这就是学习。

## 统计学习的对象
对象是数据：从数据出发，提取数据特征，抽象数据模型，发现数据知识，回到数据的分析与预测中。
前提假设：同类数据具有一定的统计规律性。比如用随机变量来描述数据特征，用概率分布来描述数据的统计规律

## 统计学习的目的
对数据进行分析和预测，性能提升，同时尽可能的提高学习效率

## 统计学习的方法
- 监督学习
- 非监督学习
- 半监督学习
- 强化学习

# 监督学习
从给定的，有限的，用于学习的训练数据（training data）集合出发，假设数据是独立同分布产生的；并假设要学习的模型属于某个函数的集合，称为假设空间（hypothesis space）；应用某个评价标准（evaluation criterion），从假设空间中选取一个最优的模型，使它对已知训练数据和未知测试数据(test data)在给定的评价标准中有最优的预测；最优模型的选取由算法实现。

## 基本概念
- 输入空间：输入所有可能取值的集合。
- 输出空间：输出所有可能取值的集合。
- 特征空间：特征向量存在的空间。（可以是有限元素的集合，也可以是欧氏空间）

输入变量写作$X$，输入变量的取值写作$x$
$$
x=(x^1,x^2,\ldots,x^n)^T
$$
$x^{i}$表示第$i$个特征
$x_{i}$表示第$i$个输入变量
$$
x_i=(x_i^1,x_i^2,\ldots,x_i^n)^T
$$
训练集通常表示为
$$
T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}
$$

### 联合概率分布
监督学习假设输入随机变量$X$和输出变量$Y$满足联合概率分布$P(X,Y)$。对于学习系统来说，联合分布的具体定义是未知的。

### 假设空间
模型属于由输入空间到输出空间的映射的集合，这个集合就是假设空间（hypothesis space）。  
监督学习的模型可以是概率模型或非概率模型，由条件概率分布$P(Y|X)$或决策函数（decision function）$Y=f(X)$表示。

## 问题形式化
监督学习利用训练数据集学习一个模型，再用模型对测试样本集进行预测（prediction）。
![model](https://github.com/sunqiang85/notebook/raw/master/statisticsMachineLearning/images/1_1.png)

通过学习得到的模型，表示为条件概率分布$\hat{P}(Y|X)$或决策函数$Y=\hat{f}(x)$  
在预测过程中由
$$
\DeclareMathOperator*{\argmax}{arg\,max}
y_{N+1}=\argmax_{y_{N+1}} P(Y_{N+1}|X_{N+1})
$$

## 模型
- 在非概率模型中，假设空间可以用$\mathcal{F}$来表示，定义为决策函数的集合
$$
\begin{eqnarray}
\mathcal{F}=\{f|Y=f(x)\}
\end{eqnarray}
$$
在机器学习中$\mathcal{F}$通常是由一个参数向量$\theta$决定的向量簇
$$
\begin{eqnarray}
\mathcal{F}=\{f|Y=f_{\theta}(X),\theta \in R^n\}
\end{eqnarray}
$$

- 在概率模型中，假设空间可以用$\mathcal{F}$来表示，定义为条件概率的集合
$$
\begin{eqnarray}
\mathcal{F}=\{|Y=P(Y|X)\}
\end{eqnarray}
$$
在机器学习中$\mathcal{F}$通常是由一个参数向量$\theta$决定的条件概率分布簇
$$
\begin{eqnarray}
\mathcal{F}=\{f|Y=P_{\theta}(Y|X),\theta \in R^n\}
\end{eqnarray}
$$

## 策略
为了判断假设空间中模型的好坏，引入损失函数和风险函数的概念
### 损失函数
损失函数是$f(X)$与$Y$的非负值函数，记作$L(Y,F(X))$。  
统计学习中常用的损失函数有:  
1. 0-1 损失函数（0-1 loss function）
$$
\begin{eqnarray}
L(Y,F(X)) =
\begin{cases} 
1,  & \mbox{if }Y\ne F(X) \\
0,  & \mbox{if }Y= F(X)
\end{cases}
\end{eqnarray}
$$
2. 平方损失函数（quadratic loss function）  
$$
\begin{equation}
L(Y,F(X))=(Y-F(X))^2
\end{equation}
$$
3. 绝对损失函数 (absolute loss function)
$$
\begin{equation}
L(Y,F(X))=|Y-F(X)|
\end{equation}
$$
4. 对手损失函数（logarithmic loss function）
$$
\begin{equation}
L(Y,F(X))=-\log{P(Y|X)}
\end{equation}
$$
模型$f(X)$关于联合分布P(X,Y)的平均意义下的损失，称为风险函数（risk function）或期望损失（expected loss）
$$
\begin{equation}
R_{exp}(f)=E_p[L(Y,F(X))]=\int_{\mathcal{X} \times \mathcal{Y}} \,L(y,f(x))P(x,y)dxdy
\end{equation}
$$
对于非概率模型来说
$$
\begin{eqnarray}
P(x,y) =
\begin{cases} 
0,  & \mbox{if }y\ne f(X) \\
P(x),  & \mbox{if }y= f(X)
\end{cases}
\end{eqnarray}
$$

风险函数依赖于联合分布概率$P(X,Y)$，如果联合概率分布知道了，就能直接求出条件概率分布$P(Y|X)$，所以监督学习是一个病态问题（ill-formed problem）
给定一个训练数据集
$$
T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}
$$
模型$f(X)$关于训练集的平均损失称为经验风险（empirical risk）或经验损失（empirical loss），记作$R_{emp}$
$$
\begin{equation}
R_{emp}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))
\end{equation}
$$
根据大数定理，当样本容量$N$趋于无穷时，$R_{emp}(f)$趋于$R_{exp}(f)$

### 经验风险最小化与结构风险最小化
经验风险化最小化策略（empirical risk minimization）：
$$
\begin{equation}
\min_{f \in \mathcal{F}}\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))
\end{equation}
$$
当样本容量较小时，会产生过度拟合的问题。
结构风险：
通过在经验风险的基础上增加了表示模型复杂度的正则化项（regularizer）或惩罚项（penalty term）来防止过度拟合
$$
\begin{equation}
R_{srm}=\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))+\lambda J(f)
\end{equation}
$$
结构风险最小化策略（structure risk minimization）：
$$
\begin{equation}
\min_{f \in \mathcal{F}}\frac{1}{N}\sum_{i=1}^NL(y_i,f(x_i))+\lambda J(f)
\end{equation}
$$

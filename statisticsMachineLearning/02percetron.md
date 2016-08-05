# 感知器笔记
## 感知机模型
定义2.1（感知机）假设输入空间（特征空间）是$x \in R^n$,输出空间是$\mathcal{Y}=\{+1,-1\}$
$$
\begin{equation}
f(x)=sign(w\cdot x+b)
\end{equation}
$$
$$
\begin{equation}
sign(x)=
\begin{cases}
+1, & x \ge 0 \\
-1, & x \lt 0
\end{cases}
\end{equation}
$$

## 感知机学习策略
### 数据的线性可分析
### 感知机学习策略
假设超平面$S$的误分类点集合为M，那么所有误分类点到超平面S的总距离为
$$
-\frac{1}{\Vert w \Vert}\sum_{x_i \in M}y_i(w \cdot x_i+b)
$$
感知机$sign(w\cdot x+b)$的损失函数定义为
$$
L(w,b)=-\sum_{x_i \in M}y_i(w \cdot x_i+b)
$$
### 感知器学习算法
随机梯度下降法（stochastic gradient descent)
输入：训练数据集$T=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}$，其中$\mathcal{Y}=\{-1,+1\}$,学习率$\eta(0\lt \eta \le 1)$  
输出：$w,b$;感知机模型$f(x)=sign(w\cdot x+b)$
1. 选取初值$w_0,b_0$
2. 在训练集中选取数据$(x_i,y_i)$
3. 如果$y_i(w\cdot x_i+b) \le 0$
4. 转至第2步，直到训练集中没有误分类点。
$$
w \gets w+\eta y_i x_i \\
b \gets b+\eta y_i
$$

直观感觉为什么可行？不妨设$(x_i,y_i)$为误分类点
$$
\begin{eqnarray*}
y_i(w_{t+1}x_i+b_{t+1}) & = &y_i((w_t+\eta y_i x_i)x_i+b_t+\eta y_i) \\
 & = &y_i(w_t+b_t)+\eta y_i^2 x_i^2+ \eta y_i^2 \\
 & \ge & y_i(w_t+b_t)
\end{eqnarray*}
$$
可见$y_i(w_{t+1}x_i+b_{t+1})$是递增的，最终会大于0，即正确分类。
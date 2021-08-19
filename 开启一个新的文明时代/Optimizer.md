## 优化器的简单介绍
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;平时用的函数，比如:y = ax²+b，我们是怎么找最值？是求导，求导可以找到一个函数变化的方向，当这个变化为0时，通常情况下这就是一个极值(不排除 y = 3^2的方法通过查找全局最优（全局最小点）来解决问题。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;深度学习网络是通用的函数逼近器，不可能对神经网络拟合的结果做出类似函数的假设(现在研究神经网络的解释也是一门大学问)。所以单纯的求导解决不了问题,所以我们要考虑其他的方法去解决问题，而且从几何上看当维数很高的时候，这就是一个n维的问题，会遇到维数诅咒的问题。 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;那如何解决神经网络模型中的寻找最优值的方法？既然求导求不到直接的结果，那我们可以用一个评估函数，去判断这个模型的好坏，然后再跟据求出的随机函数修改模型的参数，最终让模型接近最优解。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如何；来寻找最优的模型?我们可以采取两种方式:
- 随机扰动：我们可以对当前的参数集应用一个随机扰动ΔW，并计算获得的新参数集的损失值Ws=Ws–1+ΔWs–1。如果训练步骤s的损失值小于前一个步骤的损失值，我们就可以接受找到的解，并继续对新参数集应用新的随机扰动。否则，我们必须重复随机扰动，直到找到更好的解。


- 更新方向估计：这种方法不是随机生成一组新的参数，而是将局部最优研究过程引导到函数的最大下降方向上。第二种方法是参数训练型机器学习模型的实际标准，这些模型表示为可微函数。

## 具体使用的方法
### 梯度下降法的概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;函数求导就是函数的变化趋势，也可以理解为在以x为中心的无限小区域中，函数相对于变量x的变化量。(无限小并不是等于)。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;知道一阶求导的变化后就明白了，如果一个值是极值，那么这个值的导数一定等于0，而且这个导函数的两边是异号的。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如何找一阶导数等于0的点呢？要沿着反梯度的方向移动，去寻找这个点。打个比方，寻找y = x<sup>2</sup>的值，求导得到y<sup>‘</sup> = 2x 当y<sup>‘</sup> = 0的时候x的值就是极值，我们训练模型的时候是寻找比如:y=ax<sup>2</sup>+b是寻找这个a的值,所以我们可以根据得到的y<sup>'</sup> = 0的值去求a的值，这时候把a与b看做已知。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果用梯度下降的方法，就是利用导数有关的一个损失函数去一步步去逼近结果:W<sub>s</sub> = W<sub>s-1</sub>-η▽L(（x,y);W<sub>s-1</sub>)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该参数η为学习率，是梯度下降训练阶段的超参数。为学习率选择正确的值与其说是一门科学，不如说是一门艺术，我们唯一能做的就是通过直觉来选择一个适合模型和数据集的值。关于学习率的应用:
- 过高的学习率会使训练阶段不稳定，这是由局部极小值附近的跳跃所致。这会引起损失函数值的振荡。为了记住这一点，我们可以考虑一个U形曲面。如果学习率太高，在接下来的更新步骤中，则从U的左边跳到右边，反之亦然，不会下降谷值（因为U的两个峰值的距离大于η）。
- 过低的学习率会使训练阶段不理想，处于次优状态，因为我们永远不会跳出一个并非全局最小值的谷值。因此，存在陷入局部最小值的风险。此外，学习率太低的另一个风险是永远找不到一个好的解决方案——不是因为我们被困在一个局部的最小值中，而是因为我们朝着当前方向前进得太慢了。由于这是一个迭代的过程，研究可能会花费太长时间。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目前我们为了面对这个问题，从而使用一些策略去控制损失函数的生成。通过使用完整数据集计算出的损失函数来一次性更新参数。这种方法称为批量梯度下降法。但是有些数据集的数量非常大我们不能寄希望于一次性执行。

#### 随机梯度下降法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随机梯度下降法为训练数据集的每个元素更新模型参数:W<sub>s</sub> = W<sub>s-1</sub>-η▽L(（x<sub>i</sub>,y<sub>i</sub>);W<sub>s-1</sub>)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果数据集的平方差大，那么随机梯度下降会导致训练阶段损失值浮动很大。这可以让我们更好的去考虑更好的最小值的解空间探索区域。但是收敛速度慢，很难找一个合适的最小值。

<font color='red'>代码请看./example/SGD_test.py</font>

#### 小批量梯度下降
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每次用一小部分数据集为模型更新参数:W<sub>s</sub> = W<sub>s-1</sub>-η▽L(（x<sub>[i,i+b]</sub>,y<sub>[i,i+b]</sub>);W<sub>s-1</sub>)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;优势:
- 使用小批量减少了参数的更新方差，因此训练过程收敛更快。
- 使用一定基数的小批量，可以重复使用相同的方法进行在线训练。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面介绍的算法只考虑当前参数的值和通过应用定义计算的反梯度，且特征值固定，我们在寻找最佳最小值时可能遇到的问题还有:
- 选择学习率:使用相同的学习率来更新每个参数，不可能适应所有情况。
- 鞍点和停滞区：用于训练神经网络的损失函数是大量参数的函数，因此是非凸函数。在优化过程中，可能会遇到鞍点（函数值沿一个维度增加，而沿其他维度减少的点）或停滞区（损失曲面局部恒定不变的区域）。在这种情况下我们遇到的维度的梯度几乎为零，所以反梯度指向的方向几乎为零。这时候我们会以为到达最优点了，例如:y=x<sup>3</sup>,事实上我们找的这个点没有意义。

### 梯度下降优化算法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了提高普通优化的效率，我们提出一些优化的算法:
#### 动量算法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;展示了损失曲面的物理解释如何导出成功的结果,于损失曲面的物理解释。让我们把损失曲面想象成一个混乱的场景，其中一个粒子在四处移动，目的是找到全局的最小值。在传统的物理学中，存在动量定理，在零时间内将一个粒子从一个点传送到一个新的点，而且没有能量损失的系统是不存在的。由于外力以及速度随时间变化，系统的初始能量会损失。我们可以使用一个物体（粒子）的类比，该物体在一个表面（损失曲面）上滑动，受到动能摩擦力的影响，该摩擦力的能量和速度会随着时间的推移而降低。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;动量算法的更新规则:v<sub>s</sub> = ηv<sub>s-1</sub>-η▽L(W<sub>s-1</sub>);W<sub>s</sub> =W<sub>s-1</sub>+v<sub>s</sub> 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v是粒子的向量速度,该方法考虑了粒子在前一步所达到的向量速度，并再后续更新中，对于不同方向的分量减小向量速度，对于相同方向的分量则增加向量速度。这样，系统的总能量降低了，反过来减少了振荡并获得了更快的收敛.

#### ADAM算法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;早先的算法更新的步数都是固定的，为了解决这个问题，提出一套自适应学习率优化方法，将不同的学习率与网络的每个参数相关联，从而使用一个自适应于神经元专门提取的特征类型的学习率来更新这些参数。自适应矩估计ADAM算法是现在最常用的算法之一。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;作为一种自适应方法，Adam存储过去平方梯度的衰减平均值和每个参数的动态变化。它为模型中的每个参数创建一个学习率：

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Adam存储每一时刻梯度衰减指数mt的加权平均值:

- η,i = 0,1,....,|W|

- <sub>s</sub>=β<sub>1</sub>m<sub>s-1</sub>-1+(1-β1)η▽L(W<sub>s-1</sub>)

- v<sub>s</sub>=β1v<sub>s-1</sub>-1+(1-β<sub>2</sub>)η▽L(W<sub>s-1</sub>)<sup>2</sup>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第一项是梯度的指数移动平均（一阶动量估计），第二项是梯度平方的指数移动平均（二阶动量估计）。ms和vs都是带有|w|分量的向量，并且初始值均为0。β1和β2指数移动平均的衰减因子，是该算法的超参数。ms和vs向量的零初始化使它们的值接近于0，特别是当衰减因子接近于1时（因此衰减率较低）。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Adam算法的设计者利用偏差校正第一时刻和第二时刻的估计值来抵消这些偏差。

- m<sup>^</sup><sub>s</sup> = <sup>m<sub>s</sub></sup>&frasl;<sub>1-β<sup>s</sup><sub>1</sub></sub>
- v<sup>^</sup><sub>s</sup> = <sup>v<sub>s</sub></sup>&frasl;<sub>1-β<sup>s</sup><sub>2</sub></sub>
- W<sub>s</sub> = W<sub>s-1</sub>-<sup>-η</sup>&frasl;(v<sup>^</sup><sub>s</sub>+&epsilon;)<sup>1&frasl;2<sup>

<font color='red'>代码请看./example/adam_test.py</font>

#### AdaDelta算法

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AdaDelta是AdaGrad的改进，减缓了学习率的下降速率。AdaDelta不是累积所有过去的平方梯度，而是将累积过去梯度的窗口限制为固定大小。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AdaDelta不是低效地存储w大小的过去平方梯度，而是将梯度的总和递归地定义为所有过去的平方梯度的衰减平均值。时间步长t的运行平均值E[g<sup>2</sup>]<sub>t</sub>仅依赖于先前的平均值和当前梯度（γ作为系数，类似于动量项）:
E[g<sup>2</sup>]<sub>t</sub>=γE[g<sup>2</sup>]<sub>t-1</sub>-1+(1-γ)g<sup>2</sup><sub>t</sub>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中E[g<sup>2</sup>]是时间t时的梯度的平方和，γE[g<sup>2</sup>]<sub>t-1</sub>是时间t-1的梯度平方和的γ倍。

- ∆W<sub>t</sub>=-η·g<sub>t,i</sub>
- W<sub>t</sub>+1=W<sub>t</sub>+∆W<sub>t</sub>
- ∆W<sub>t</sub> = <sup>-η</sup>&frasl;(E[g<sup>2</sup>]<sub>t</sub>+&epsilon;)<sup>1&frasl;2</sup>*g<sub>t</sub>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型的使用:模型的加速效果很不错，最终在局部最小值周围浮动。

<font color='red'>代码请看./example/adadelta_test.py</font>

#### RMSprop

- E[g<sup>2</sup>]<sub>t</sub>=γE[g<sup>2</sup>]<sub>t-1</sub>-1+(1-γ)g<sup>2</sup><sub>t</sub>就变为了求梯度平方和的平均数。
- ∆W<sub>t</sub> = <sup>-η</sup>&frasl;(E[g<sup>2</sup>]<sub>t</sub>+&epsilon;)<sup>1&frasl;2</sup>×g<sub>t</sub>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;模型的使用:适合处理非平稳目标

<font color='red'>代码请看./example/rmsprop_test.py</font>

#### 自动微分
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;自动微分是将复合函数分解为输出量（根节点）和一系列的输入量（叶子节点）及基本函数（中间节点），构成一个计算图，并以此计算任意两个节点间的梯度。这其实是高等数学讲的知识点。
- 加法法则：任意两个节点间的梯度为它们两节点之间所有路径的偏微分之和；
- 链式法则：一条路径的偏微分为路径上各相邻节点间偏微分的连乘。

  (有时间在高等数学中找个简单的例子讲解一下)
#### 反向传播算法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;神经网络中只要各个的部件的函数是可微的那么整个神经网络也可以说是可微的，就可以用自动微分去计算梯度，从而在复杂的环境中使用梯度下降法。

<font size=20 color='blue'>2021.9.2结束优化器的笔记</font>
# 第一次作业 VAE 逐题解析

对应原题：[Assignment1.pdf](../Assignment1.pdf)。以下保留原题编号；所有对数均为自然对数。

题干按所附 PDF 原文转写，合并排版断行并重排数学公式；共同题设放在小问之前，每个小问紧接独立解析。

原题课程标记：Peking University Generative Models，Module 1: VAE；Assignment 1: Variational Autoencoders。

## 第1题 ELBO 的推导与等号条件

**原题（PDF第1页）**

Let $`p_\theta(x,z)=p(z)p_\theta(x\mid z)`$ and let $`q_\phi(z\mid x)`$ be an approximate posterior. Derive

```math
\log p_\theta(x)\geq
\underbrace{\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\rm KL}(q_\phi(z\mid x)\Vert p(z))}_{\mathcal L(\theta,\phi;x)},
```

and identify when equality holds. You may assume common support and finite required expectations.

**详细解析**

**题意。** 用近似后验 $`q_\phi(z\mid x)`$ 推导对数边缘似然的下界，并说明何时取等号。

固定观测 $`x`$，简写 $`q=q_\phi(z\mid x)`$。利用贝叶斯公式，

```math
\begin{aligned}
D_{\rm KL}(q\Vert p_\theta(z\mid x))
&=\mathbb E_q[\log q-\log p_\theta(x,z)+\log p_\theta(x)]\\
&=\log p_\theta(x)-\mathbb E_q\!\left[\log\frac{p_\theta(x,z)}{q}\right].
\end{aligned}
```

把联合分布 $`p_\theta(x,z)=p(z)p_\theta(x\mid z)`$ 代入，得到恒等式

```math
\boxed{
\log p_\theta(x)
=\underbrace{\mathbb E_q\log p_\theta(x\mid z)-D_{\rm KL}(q\Vert p(z))}_{\mathcal L(\theta,\phi;x)}
+D_{\rm KL}(q\Vert p_\theta(z\mid x)).}
```

最后一项非负，因此 $`\log p_\theta(x)\geq\mathcal L`$。**等号当且仅当 $`q_\phi(z\mid x)=p_\theta(z\mid x)`$ 几乎处处成立**。这里需要匹配的是真实后验，而不是先验 $`p(z)`$。

也可以直接用 Jensen 不等式：

```math
\log p_\theta(x)=\log\mathbb E_q\frac{p_\theta(x,z)}{q(z\mid x)}
\geq\mathbb E_q\log\frac{p_\theta(x,z)}{q(z\mid x)}.
```

直观上，重建项鼓励潜变量保留解释 $`x`$ 的信息，先验 KL 项约束潜变量的分布；它们共同构成一个可训练的下界。最大化下界通常有助于似然学习，但不保证每一步都使真实似然增大，因为后验近似的误差也会变化。


**推导细节与进一步理解**

**从 Bayes 公式到下界，中间没有省略的步骤。** Bayes 公式先给出

```math
p_\theta(z\mid x)=\frac{p(z)p_\theta(x\mid z)}{p_\theta(x)}.
```

取对数、移项，再加减 $`\log q_\phi(z\mid x)`$，得到逐个 $`z`$ 成立的关系

```math
\begin{aligned}
\log p_\theta(x)
&=\log p_\theta(x\mid z)+\log p(z)-\log p_\theta(z\mid x)\\
&=\log p_\theta(x\mid z)
-\log\frac{q_\phi(z\mid x)}{p(z)}
+\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}.
\end{aligned}
```

两边在 $`q_\phi(z\mid x)`$ 下取期望。左边不依赖 $`z`$，所以期望仍为自身；右边第二、三项分别恰好是两个 KL。这解释了为什么 ELBO 里的先验 KL 带负号，而下界差距里的后验 KL 带正号。

**用一个两点例子理解“下界”。** 假设固定观测 $`x`$ 后，联合概率在两个潜变量上为 $`p(x,z=1)=0.2`$、$`p(x,z=2)=0.3`$，则 $`p(x)=0.5`$，真实后验是 $`(0.4,0.6)`$。若近似后验取 $`(0.5,0.5)`$，

```math
\mathrm{ELBO}=\tfrac12\log(0.2/0.5)+\tfrac12\log(0.3/0.5)
\approx-0.71356<\log0.5\approx-0.69315.
```

若改为真实后验 $`(0.4,0.6)`$，两个联合概率除以后验都等于 $`0.5`$，Jensen 中的随机量变为常数，ELBO 就等于 $`\log0.5`$。因此“后验近似越准确，下界越紧”指的是固定生成模型和固定观测时，后验 KL 越小。

**为什么不直接最大化边缘似然。** $`p(z)`$ 通常很简单，但 $`\int p(z)p_\theta(x\mid z)dz`$ 穿过了非线性解码器，积分及真实后验往往难算。ELBO 引入可采样、可计算的 $`q_\phi`$，将训练变成重建期望加可处理的正则项，而不是声称真实似然不存在或不能定义。

## 第2题 一般多元高斯之间的 KL 散度

**原题（PDF第1页）**

For $`d`$-dimensional Gaussians $`q(z)=\mathcal N(\mu_q,\Sigma_q)`$ and $`p(z)=\mathcal N(\mu_p,\Sigma_p)`$ with positive definite covariance matrices, derive the closed form of $`D_{\rm KL}(q\Vert p)`$.

**详细解析**

**题意。** $`q=\mathcal N(\mu_q,\Sigma_q)`$，$`p=\mathcal N(\mu_p,\Sigma_p)`$，协方差正定。求 $`D_{\rm KL}(q\Vert p)`$。

从定义出发：

```math
\begin{aligned}
\log\frac{q(z)}{p(z)}
=\frac12\bigg[&\log\frac{\det\Sigma_p}{\det\Sigma_q}
-(z-\mu_q)^\top\Sigma_q^{-1}(z-\mu_q)\\
&+(z-\mu_p)^\top\Sigma_p^{-1}(z-\mu_p)\bigg].
\end{aligned}
```

令 $`u=z-\mu_q`$、$`\Delta=\mu_q-\mu_p`$。因为 $`\mathbb E_qu=0`$、$`\mathbb E_quu^\top=\Sigma_q`$，对任意固定矩阵 $`A`$ 有

```math
\mathbb E_q[u^\top Au]=\mathrm{tr}(A\Sigma_q).
```

于是

```math
\mathbb E_q[(z-\mu_q)^\top\Sigma_q^{-1}(z-\mu_q)]=d,
```

```math
\mathbb E_q[(z-\mu_p)^\top\Sigma_p^{-1}(z-\mu_p)]
=\mathrm{tr}(\Sigma_p^{-1}\Sigma_q)+\Delta^\top\Sigma_p^{-1}\Delta.
```

代回即得

```math
\boxed{
D_{\rm KL}(q\Vert p)=\frac12\left[
\mathrm{tr}(\Sigma_p^{-1}\Sigma_q)
+(\mu_p-\mu_q)^\top\Sigma_p^{-1}(\mu_p-\mu_q)
-d+\log\frac{\det\Sigma_p}{\det\Sigma_q}\right].}
```

常见特例 $`p=\mathcal N(0,I)`$，$`q=\mathcal N(\mu,\mathrm{diag}(e^r))`$：

```math
\boxed{D_{\rm KL}(q\Vert p)=\frac12\sum_{i=1}^d(e^{r_i}+\mu_i^2-1-r_i).}
```

**易错点。** KL 的方向决定使用谁的逆协方差；$`D_{\rm KL}(q\Vert p)`$ 中是 $`\Sigma_p^{-1}`$。若 $`p=q`$，答案应恰为 0，可用来检查符号。


**推导细节与进一步理解**

**把高斯密度的对数真正写开。** 两个高斯都含有 $`-\frac d2\log(2\pi)`$，做差后这一项抵消。剩下的行列式项来自归一化系数，二次型项来自指数；因此可以在求期望前就判断对数行列式的方向是 $`\log|\Sigma_p|-\log|\Sigma_q|`$。

二次型期望公式也可逐坐标推导：

```math
\mathbb E[u^\top Au]
=\sum_{i,j}A_{ij}\mathbb E[u_iu_j]
=\sum_{i,j}A_{ij}(\Sigma_q)_{ij}
=\mathrm{tr}(A\Sigma_q),
```

这里用到了协方差矩阵的对称性。再将 $`z-\mu_p=u+\Delta`$ 展开：

```math
(u+\Delta)^\top\Sigma_p^{-1}(u+\Delta)
=u^\top\Sigma_p^{-1}u+2\Delta^\top\Sigma_p^{-1}u
+\Delta^\top\Sigma_p^{-1}\Delta.
```

中间交叉项的期望为零，是因为 $`\mathbb E u=0`$，而不是因为每个样本上 $`u`$ 都等于零。

**检查一维特例。** 公式应退化为

```math
D_{\rm KL}(\mathcal N(\mu_q,\sigma_q^2)\Vert\mathcal N(\mu_p,\sigma_p^2))
=\frac12\left[\frac{\sigma_q^2+(\mu_q-\mu_p)^2}{\sigma_p^2}
-1+\log\frac{\sigma_p^2}{\sigma_q^2}\right].
```

若方差相同而均值不同，KL 就是均值差平方除以 $`2\sigma^2`$；若均值相同而方差比例为 $`r=\sigma_q^2/\sigma_p^2`$，KL 为 $`(r-1-\log r)/2`$，由 $`\log r\le r-1`$ 可确认非负。正定性保证逆矩阵和对数行列式存在，不能把这套普通密度公式不加说明地用于退化高斯。

## 第3题 输出对数方差时的重参数化梯度

**原题（PDF第1页）**

An encoder outputs $`\mu,r\in\mathbb R^d`$, where $`r_i=\log\sigma_i^2`$. Define

```math
F(\mu,r)=\mathbb E_{z\sim\mathcal N(\mu,\mathrm{diag}(e^r))}[f(z)].
```

Assume that $`f`$ is differentiable, and that differentiation and expectation may be interchanged. Using parameter-independent noise $`\epsilon\sim\mathcal N(0,I)`$, derive a reparameterized expression for $`\nabla_{(\mu,r)}F`$.

**详细解析**

**题意。** 编码器输出 $`\mu,r`$，其中 $`r_i=\log\sigma_i^2`$，求
$`F(\mu,r)=\mathbb E_{z\sim\mathcal N(\mu,\mathrm{diag}(e^r))}f(z)`$ 对两组参数的梯度。

取与参数无关的 $`\epsilon\sim\mathcal N(0,I)`$，定义

```math
\sigma=e^{r/2},\qquad z=\mu+e^{r/2}\odot\epsilon.
```

注意标准差是 $`e^{r/2}`$，不能写成 $`e^r`$。于是

```math
F(\mu,r)=\mathbb E_\epsilon f(\mu+e^{r/2}\odot\epsilon).
```

固定一次噪声后，$`\partial z_i/\partial\mu_j=\mathbf1_{i=j}`$，而
$`\partial z_i/\partial r_j=\tfrac12e^{r_i/2}\epsilon_i\mathbf1_{i=j}`$。在题目允许交换微分与期望的条件下，

```math
\boxed{\nabla_\mu F=\mathbb E_\epsilon[\nabla_zf(z)],\qquad
\nabla_r F=\frac12\mathbb E_\epsilon[\nabla_zf(z)\odot e^{r/2}\odot\epsilon].}
```

若使用 $`K`$ 个独立噪声样本，以样本平均替换期望就得到无偏的路径梯度估计。对上游编码器参数 $`\phi`$ 再使用链式法则：

```math
\nabla_\phi F=J_\mu(\phi)^\top\nabla_\mu F+J_r(\phi)^\top\nabla_r F.
```

这里题设的 $`f`$ 没有显式依赖 $`\mu,r`$；若它还显式依赖参数，需要额外加入对应的直接偏导项。重参数化解决的是如何对含采样的期望求梯度，而不是消除随机性。


**推导细节与进一步理解**

**为什么梯度可以通过采样。** 原表达式的积分权重由 $`\mu,r`$ 决定；换成标准高斯 $`\epsilon`$ 后，积分权重固定，参数只出现在 $`z(\mu,r,\epsilon)`$ 这个确定变换中。对一次固定噪声做微小参数扰动时，噪声不重抽，而样本位置会随参数连续移动，这就是路径梯度。

逐坐标链式法则为

```math
\frac{\partial}{\partial\mu_j}f(z)=\sum_i\frac{\partial f}{\partial z_i}\frac{\partial z_i}{\partial\mu_j}
=\frac{\partial f}{\partial z_j},
```

```math
\frac{\partial}{\partial r_j}f(z)=\sum_i\frac{\partial f}{\partial z_i}\frac{\partial z_i}{\partial r_j}
=\frac{\partial f}{\partial z_j}\cdot\frac12e^{r_j/2}\epsilon_j.
```

**用能手算的函数检查 $`1/2`$。** 一维取 $`f(z)=z^2`$，则 $`F=\mu^2+e^r`$，所以正确梯度是 $`\partial_\mu F=2\mu`$、$`\partial_rF=e^r`$。路径公式给出

```math
\mathbb E[2z]=2\mu,\qquad
\mathbb E[2z\cdot\tfrac12e^{r/2}\epsilon]
=\mu e^{r/2}\mathbb E\epsilon+e^r\mathbb E\epsilon^2=e^r.
```

如果把 $`r`$ 错当成 log standard deviation，就会漏掉或多出因子 2。实际反向传播时，编码器的 $`\mu,r`$ 两路都必须保留计算图；参数无关的是 $`\epsilon`$，不是变换后的 $`z`$。

## 第4题 Gumbel Max 与 Gumbel Softmax

**原题（PDF第1页）**

Paper: Eric Jang, Shixiang Gu, and Ben Poole. *Categorical Reparameterization with Gumbel-Softmax*. ICLR 2017. Citation 9500+.

Explain what is Gumbel-Max trick and prove it. Explain when and how to use Gumbel-Softmax trick.

**详细解析**

### Gumbel Max 的结论

设类别权重 $`w_i>0`$，概率 $`\pi_i=w_i/\sum_jw_j`$，且 $`g_i`$ 独立服从标准 Gumbel 分布，累积分布函数为 $`F(g)=\exp(-e^{-g})`$。则

```math
\boxed{k=\arg\max_i(\log w_i+g_i)\quad\Longrightarrow\quad\Pr(k=i)=\pi_i.}
```

可以用 $`u_i\sim\mathrm{Uniform}(0,1)`$ 和 $`g_i=-\log(-\log u_i)`$ 生成 Gumbel 噪声。使用 $`\log\pi_i`$ 或未归一化的 logit 都可以，因为共同的加法常数不影响最大值。

### 用指数竞赛证明

令 $`E_i=e^{-g_i}=-\log u_i`$，则 $`E_i\sim\mathrm{Exp}(1)`$。再令 $`T_i=E_i/w_i`$，则 $`T_i\sim\mathrm{Exp}(w_i)`$ 且独立。注意

```math
\log w_i+g_i=-\log(E_i/w_i)=-\log T_i.
```

因此取最大值等价于取 $`T_i`$ 的最小值。独立连续分布的平局概率为零，故

```math
\begin{aligned}
\Pr(k=i)
&=\int_0^\infty \underbrace{w_i e^{-w_it}}_{T_i\text{ 的密度}}
\prod_{j\ne i}\underbrace{e^{-w_jt}}_{\Pr(T_j>t)}\,dt\\
&=w_i\int_0^\infty e^{-(\sum_jw_j)t}\,dt
=\frac{w_i}{\sum_jw_j}.
\end{aligned}
```

零概率类别可删除，或把其 logit 视为 $`-\infty`$。

### 何时以及怎样使用 Gumbel Softmax

神经网络需要从类别分布采样、并让梯度通过该离散选择时，可把不可导的 one hot 最大值选择替换为温度为 $`\tau>0`$ 的连续向量：

```math
\boxed{y_i=\frac{\exp((\log w_i+g_i)/\tau)}{\sum_j\exp((\log w_j+g_j)/\tau)}.}
```

对 logit $`a_j=\log w_j`$，固定噪声后有

```math
\frac{\partial y_i}{\partial a_j}=\frac1\tau y_i(\mathbf1_{i=j}-y_j).
```

训练时先采样 Gumbel 噪声，再把 $`y`$ 输入后续网络，按普通反向传播优化。$`\tau\to0^+`$ 时 $`y`$ 几乎必然收敛到 Gumbel Max 的 one hot 样本；$`\tau\to\infty`$ 时趋于均匀向量。有限温度优化的是连续松弛目标，其梯度通常不是原离散目标梯度的无偏估计。

需要前向严格离散时，可用直通估计：前向取 $`y_{\rm hard}=\mathrm{onehot}(\arg\max_i y_i)`$，反向使用 $`y`$ 的导数，实现为 $`y_{\rm hard}-\mathrm{stopgrad}(y)+y`$。它同样是有偏估计。过低温度使输出更接近离散样本，也可能使梯度不稳定，实践中需权衡。

论文来源：[Jang 等，ICLR 2017，第 2 节](https://arxiv.org/html/1611.01144v5#S2)。上面的指数竞赛展开给出了题目所需的采样正确性证明。

**推导细节与进一步理解**

**为什么要给 log 概率加噪声。** 直接选择概率最大的类别只会得到众数，不会按整个分布抽样。Gumbel Max 让每个类别独立竞争，且竞争获胜率恰好与 $`w_i`$ 成正比。若用网络 logits $`a_i`$，取 $`w_i=e^{a_i}`$，便可直接计算 $`\arg\max_i(a_i+g_i)`$，不用先算 softmax；这有利于数值稳定。

指数竞赛里用到的分布变换可补充为：对 $`t\ge0`$，

```math
\Pr(E_i>t)=\Pr(-\log U_i>t)=\Pr(U_i<e^{-t})=e^{-t},
```

所以 $`E_i`$ 的确是率为 1 的指数变量；同理 $`\Pr(T_i>t)=e^{-w_it}`$。获胜概率积分中“自己的密度乘其他人的生存概率”，就是要求自己在 $`t`$ 时到达、其他所有类别都比 $`t`$ 晚。

**Gumbel Softmax 的梯度并没有穿过 argmax。** 它把离散选择换成概率单纯形内部的向量，后续网络因而接收连续值。对同一个 Gumbel 样本，温度逐渐减小时最大分量占据的比例越来越大；由于连续噪声下最大值几乎必然唯一，极限才是 one hot。温度很小时，大部分样本已经饱和，少数接近平局的样本又可能有较大的 $`1/\tau`$ 因子，所以不能简单认为“温度越小梯度越好”。

**直通估计的前后向分别是什么。** 写 $`y_{\rm ST}=y_{\rm hard}-\mathrm{stopgrad}(y)+y`$：前向两项 $`-y+y`$ 抵消，实际送入后续网络的是 $`y_{\rm hard}`$；反向时 stop gradient 的导数为零，$`y_{\rm hard}`$ 也不通过 argmax 求导，只留下连续 $`y`$ 的导数。因此它是一种人为设计的替代梯度，不能把“能反传”误说成“等于原离散目标的精确梯度”。

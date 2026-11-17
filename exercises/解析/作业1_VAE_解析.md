# 第一次作业 VAE 逐题解析

对应原题：[Assignment1.pdf](../Assignment1.pdf)。以下保留原题编号；所有对数均为自然对数。

## 第1题 ELBO 的推导与等号条件

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

## 第2题 一般多元高斯之间的 KL 散度

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

## 第3题 输出对数方差时的重参数化梯度

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

## 第4题 Gumbel Max 与 Gumbel Softmax

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

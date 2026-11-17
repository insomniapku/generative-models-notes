# 第三次作业 GAN 逐题解析

对应原题：[GenModelExercise2026.pdf](../GenModelExercise2026.pdf)。该文件首页明确标注 Assignment 3。下文记 $`p=p_{\rm data}`$、$`g=p_g`$；两者均相对于题设的同一参考测度取密度。

## 第1题 GAN 的极小极大博弈

### 第1题a 完整目标与双方任务

```math
\boxed{\min_G\max_D V(D,G)
=\mathbb E_{x\sim p_{\rm data}}\log D(x)
+\mathbb E_{z\sim p_z}\log(1-D(G(z))).}
```

判别器最大化这个目标，让真实样本的 $`D(x)`$ 接近 1、生成样本的 $`D(G(z))`$ 接近 0。生成器最小化目标中与自己有关的第二项，让生成样本更容易被判为真。训练一般交替更新两者；不是让两者同时最小化同一个 $`V`$。

### 第1题b 最优判别器

固定生成器后，

```math
V(D,G)=\int\big[p(x)\log D(x)+g(x)\log(1-D(x))\big]\,dx.
```

由于题目允许所有可测判别器，可以逐点优化。对 $`p(x)>0,g(x)>0`$ 的点，设 $`u=D(x)`$：

```math
\frac{d}{du}[p\log u+g\log(1-u)]=\frac pu-\frac g{1-u}=0
\quad\Longrightarrow\quad u=\frac p{p+g}.
```

二阶导数为 $`-p/u^2-g/(1-u)^2<0`$，所以它是唯一最大值。边界情况也符合公式：$`p>0,g=0`$ 时取 $`u=1`$；$`p=0,g>0`$ 时取 $`u=0`$。若 $`p=g=0`$，被积函数恒为 0，任意 $`u\in[0,1]`$ 都最优。

故在 $`p(x)+g(x)>0`$ 处，

```math
\boxed{D^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_g(x)}.}
```

这是对真实/生成来源使用相等先验概率时的后验概率。实际神经网络受容量与优化限制，不一定达到这个无限容量最优解。

### 第1题c 饱和损失与非饱和损失的梯度

写 $`D=\sigma(a)`$，则 $`dD/da=D(1-D)`$。两种待最小化的生成器损失满足

```math
\boxed{\frac{d\ell_{\rm sat}}{da}
=\frac{d\log(1-D)}{da}=-D,}
```

```math
\boxed{\frac{d\ell_{\rm ns}}{da}
=\frac{d(-\log D)}{da}=D-1.}
```

更新生成器时判别器参数固定，但 $`a(G_\theta(z))`$ 对生成器输出的导数仍须保留。链式法则给出

```math
\nabla_\theta\ell_{\rm sat}=-D(G_\theta(z))\,\nabla_\theta a(G_\theta(z)),
```

```math
\nabla_\theta\ell_{\rm ns}=(D(G_\theta(z))-1)\,\nabla_\theta a(G_\theta(z)).
```

当 $`D(G_\theta(z))\approx0`$ 时，原损失的前置系数约为 0，判别器越容易识别假样本，生成器反而越难得到训练信号。非饱和损失的系数约为 $`-1`$，消除了这一个饱和因素。它不保证整个梯度永不消失：如果 $`\nabla_\theta a(G_\theta(z))`$ 本身很小，梯度仍可能很小。

## 第2题 最优 GAN 与 JS 散度

**题意。** 代入最优判别器，证明 $`C(G)=2\mathrm{JSD}(p_{\rm data}\Vert p_g)-2\log2`$，再求全局最小值和等号条件。

令 $`m=(p+g)/2`$。由第1题，

```math
\begin{aligned}
C(G)
&=\int p\log\frac p{p+g}+g\log\frac g{p+g}\,dx\\
&=\int p\log\frac p m\,dx+\int g\log\frac g m\,dx
-(\log2)\left(\int p\,dx+\int g\,dx\right)\\
&=D_{\rm KL}(p\Vert m)+D_{\rm KL}(g\Vert m)-2\log2\\
&=\boxed{2\mathrm{JSD}(p\Vert g)-2\log2}.
\end{aligned}
```

由于两个 KL 均非负，

```math
\boxed{\min_{p_g}C(G)=-2\log2=-\log4.}
```

要取等号，两个非负 KL 必须同时为 0，因此 $`p=m`$ 且 $`g=m`$ 几乎处处，即 $`p_g=p_{\rm data}`$ 几乎处处。反过来若两分布相等，$`m=p=g`$，JS 为 0，所以确实取到最小值。

在匹配后的分布支持上，$`D^*=1/2`$。最优的是**生成分布**，不一定只有一套生成器参数能实现它。若限定生成器网络无法表达真实分布，则该理论下界不一定可达。

**数值检查。** 当两分布支持完全分离时，JS 为 $`\log2`$、$`C(G)=0`$；当两者相等时，JS 为 0、$`C(G)=-2\log2`$。这也解释了为什么支持分离时原始 GAN 距离可能不提供有用的连续变化信号。

## 第3题 GigaGAN 与原始 GAN 的技术区别

原始 GAN 给出了 $`z\to G(z)`$ 与真假判别的通用框架，没有规定现代文本条件生成架构。GigaGAN 保留对抗训练，重新设计了可扩展的文本生成器和判别器。下表概括原论文第 3 节的设计。[论文方法部分](https://arxiv.org/html/2303.05511v2#S3)

| 方面 | 原始 GAN 框架 | GigaGAN |
|---|---|---|
| 条件输入 | 基本形式仅输入噪声 | CLIP 文本特征与噪声共同控制生成 |
| 生成器 | 通用网络 $`G(z)`$ | 风格映射、样本自适应卷积核选择、自注意力和文本交叉注意力 |
| 判别器 | 真实/生成样本判别 | 文本条件判别，多尺度输入、多尺度输出 |
| 训练信号 | 基础对抗目标 | 加入文本匹配、CLIP 对比与视觉辅助对抗信号 |
| 高分辨率 | 框架本身无专门方案 | 另训练 GAN 上采样器 |

理解时把握两条因果链即可：文本交叉注意力让局部图像特征对齐词语；多尺度判别把监督传到不同分辨率，帮助大模型训练。它仍然通过前馈生成器合成图像，基础生成器与上采样器可以分阶段运行；无需像 DDPM 那样执行一长串去噪步骤。

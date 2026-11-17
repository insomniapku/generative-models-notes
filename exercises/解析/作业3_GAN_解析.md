# 第三次作业 GAN 逐题解析

对应原题：[GenModelExercise2026.pdf](../GenModelExercise2026.pdf)。该文件首页明确标注 Assignment 3。下文记 $`p=p_{\rm data}`$、$`g=p_g`$；两者均相对于题设的同一参考测度取密度。

题干按所附 PDF 原文转写，合并排版断行并重排数学公式；共同题设放在小问之前，每个小问紧接独立解析。

原题课程标记：Peking University Generative Models，Module 3: GAN；Assignment 3: Generative Adversarial Networks。

## 第1题 GAN 的极小极大博弈

**原题（PDF第1页）**

Let $`z\sim p_z`$ and let $`G_\theta(z)`$ induce the generator distribution $`p_g`$. Let $`p_{\rm data}`$ be the real data distribution, and let $`D(x)\in[0,1]`$ estimate the probability that $`x`$ is real. Assume that $`p_{\rm data}`$ and $`p_g`$ admit densities with respect to a common reference measure, and use the convention $`0\log0=0`$.

### 第1题a 完整目标与双方任务

**原题（PDF第1页）**

(a) Write down the complete minimax objective $`\min_G\max_D V(D,G)`$ for the original GAN. Explain the goals of the generator and the discriminator.

**详细解析**

```math
\boxed{\min_G\max_D V(D,G)
=\mathbb E_{x\sim p_{\rm data}}\log D(x)
+\mathbb E_{z\sim p_z}\log(1-D(G(z))).}
```

判别器最大化这个目标，让真实样本的 $`D(x)`$ 接近 1、生成样本的 $`D(G(z))`$ 接近 0。生成器最小化目标中与自己有关的第二项，让生成样本更容易被判为真。训练一般交替更新两者；不是让两者同时最小化同一个 $`V`$。


**推导细节与进一步理解**

**为什么一边最大化、一边最小化。** 把真实样本标签设为 1、生成样本标签设为 0，判别器的二元交叉熵损失是 $`L_D=-V(D,G)`$。最小化这个分类损失等价于最大化 $`V`$。固定判别器之后，第一项只包含真实数据和判别器，不含生成器参数，因此生成器真正优化的只有第二项。

在一个大小为 $`m`$ 的 minibatch 上，可以写成

```math
L_D=-\frac1m\sum_{i=1}^m\left[\log D(x_i)+\log(1-D(G(z_i)))\right],
\qquad
L_G^{\rm sat}=\frac1m\sum_{i=1}^m\log(1-D(G(z_i))).
```

判别器更新时，生成样本应当对生成器停止梯度；生成器更新时，则固定判别器参数，但继续保留“生成图像到判别器输出”的导数。二者的冻结位置不同，是对抗训练实现里常见的混淆点。

目标比较的是分布而非单个样本配对：标准 GAN 没有要求某个噪声 $`z_i`$ 必须重建某张指定的 $`x_i`$。因此不能把生成器的目标误写成逐样本重建误差。

### 第1题b 最优判别器

**原题（PDF第1页）**

(b) For a fixed generator $`G`$, prove that an optimal discriminator is

```math
D^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_g(x)}
```

wherever $`p_{\rm data}(x)+p_g(x)\gt 0`$.

**详细解析**

固定生成器后，

```math
V(D,G)=\int\big[p(x)\log D(x)+g(x)\log(1-D(x))\big]\,dx.
```

由于题目允许所有可测判别器，可以逐点优化。对 $`p(x)\gt 0,g(x)\gt 0`$ 的点，设 $`u=D(x)`$：

```math
\frac{d}{du}[p\log u+g\log(1-u)]=\frac pu-\frac g{1-u}=0
\quad\Longrightarrow\quad u=\frac p{p+g}.
```

二阶导数为 $`-p/u^2-g/(1-u)^2\lt 0`$，所以它是唯一最大值。边界情况也符合公式：$`p\gt 0,g=0`$ 时取 $`u=1`$；$`p=0,g\gt 0`$ 时取 $`u=0`$。若 $`p=g=0`$，被积函数恒为 0，任意 $`u\in[0,1]`$ 都最优。

故在 $`p(x)+g(x)\gt 0`$ 处，

```math
\boxed{D^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_g(x)}.}
```

这是对真实/生成来源使用相等先验概率时的后验概率。实际神经网络受容量与优化限制，不一定达到这个无限容量最优解。


**推导细节与进一步理解**

**为什么允许逐点最大化。** 在题目给出的无限判别器函数类中，不同 $`x`$ 的 $`D(x)`$ 没有参数共享约束，改变一个位置的输出不会迫使另一个位置也改变，因此可把积分优化分解成每个 $`x`$ 上的一元凹函数优化。实际神经网络有共享参数，所以这个推导给的是理论最优基准。

令一阶导为零后，代数过程是

```math
\frac pu=\frac g{1-u}
\quad\Longrightarrow\quad p(1-u)=gu
\quad\Longrightarrow\quad p=(p+g)u.
```

除以 $`p+g`$ 之前要检查其为正；这就是题目特意说明支持条件的原因。比如某点真实密度为 3、生成密度为 1，则最优输出是 $`3/4`$，反映该点更常来自真实数据，而不是只要它位于真实支持中就必须输出 1。

**相等先验从哪里来。** 两个期望在目标中使用同等权重，相当于真假来源先验各一半。若目标改为 $`\rho\mathbb E_p\log D+(1-\rho)\mathbb E_g\log(1-D)`$，则相同推导会给出 $`D^*=\rho p/(\rho p+(1-\rho)g)`$；因此分母中简单相加不是无条件成立的。

### 第1题c 饱和损失与非饱和损失的梯度

**原题（PDF第1页）**

(c) During a generator update, hold the discriminator fixed and write $`D(x)=\sigma(a(x))`$, where $`a(x)`$ is its logit and $`\sigma(a)=(1+e^{-a})^{-1}`$. For one generated sample, consider

```math
\ell_{\rm sat}(a)=\log(1-\sigma(a)),\qquad
\ell_{\rm ns}(a)=-\log\sigma(a),
```

both of which are minimized by the generator. Derive their derivatives with respect to $`a`$ and use the chain rule to explain why the original loss can yield vanishing generator gradients when $`D(G_\theta(z))\approx0`$, and how the non-saturating loss addresses this.

**详细解析**

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


**推导细节与进一步理解**

**把 sigmoid 的导数逐项代入。** 对饱和损失，外层对 $`D`$ 的导数为 $`-1/(1-D)`$，而内层为 $`D(1-D)`$，两者相乘是 $`-D`$。对非饱和损失，外层为 $`-1/D`$，乘以内层得到 $`-(1-D)=D-1`$。

再把生成器到 logit 的路径分开：

```math
\nabla_\theta a(G_\theta(z))
=J_{G_\theta}(z)^\top\nabla_xa(x)\big|_{x=G_\theta(z)}.
```

整个梯度由“损失对 logit 的系数”“判别器对输入的导数”“生成器对参数的导数”共同组成。非饱和损失主要修复第一部分，而不是保证后两部分也良好。

例如 $`a=-8`$ 时，$`D\approx0.000335`$，饱和损失的导数约为 $`-0.000335`$，非饱和损失的导数约为 $`-0.999665`$。在同一条后续梯度路径上，后者不会因为判别器过于确信“这是假的”就额外乘上一个接近零的因子。两者都希望增大假样本的 logit，但提供的训练信号强度不同。

## 第2题 最优 GAN 与 JS 散度

**原题（PDF第1页）**

Let $`C(G)=\max_D V(D,G)`$, where the maximization is over all measurable discriminators $`D:\mathcal X\to[0,1]`$. Define the mixture density $`m=(p_{\rm data}+p_g)/2`$ and recall

```math
D_{\rm KL}(p\Vert q)=\mathbb E_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right],
```

```math
\mathrm{JSD}(p\Vert q)=\frac12D_{\rm KL}\left(p\middle\Vert\frac{p+q}{2}\right)
+\frac12D_{\rm KL}\left(q\middle\Vert\frac{p+q}{2}\right).
```

Using the optimal discriminator from Question 1, prove that

```math
C(G)=2\mathrm{JSD}(p_{\rm data}\Vert p_g)-2\log2.
```

Over all generator distributions, show that its global minimum is $`-2\log2`$, attained if and only if $`p_g=p_{\rm data}`$ almost everywhere. All logarithms are natural.

**详细解析**

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


**推导细节与进一步理解**

**常数为什么恰好是 $`-2\log2`$。** 由 $`p+g=2m`$，每个对数都拆成 $`\log(p/m)-\log2`$ 或 $`\log(g/m)-\log2`$。对前者用 $`p`$ 积分，常数贡献 $`-\log2\int p=-\log2`$；后者又贡献一次。因此不是 $`-\log2`$，也不是 $`-\log4`$ 再乘 2。

**两个 KL 为什么总是有定义。** $`m`$ 覆盖了两分布支持的并集；在 $`p\gt 0`$ 的地方有 $`m\ge p/2\gt 0`$，所以不会出现把正的 $`p`$ 除以零的问题。$`g`$ 也同理。即使 $`p`$ 与 $`g`$ 的支持互不相交，JS 仍然有限。

**达到最优时发生什么。** 当 $`p=g`$，判别器在有质量的地方只能输出 $`1/2`$，两项期望各为 $`\log(1/2)`$，总和为 $`-2\log2`$。当支持完全分离，判别器可把真样本判成 1、假样本判成 0，两项 log 都为 0，总和反而是 0。由于生成器在最小化，这说明分布匹配时目标更小，符号方向是对的。

这里先对 $`D`$ 完全优化再讨论 $`C(G)`$；实际训练只交替做有限次梯度更新，不能据此断言每次迭代的 JS 都单调下降。全局最优分布的结论也不能排除有限网络训练中的模式坍塌。

## 第3题 GigaGAN 与原始 GAN 的技术区别

**原题（PDF第1页）**

Paper: Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. *Scaling up GANs for Text-to-Image Synthesis*. CVPR 2023.

Describe the technical differences between GigaGAN and the original GAN.

**详细解析**

原始 GAN 给出了 $`z\to G(z)`$ 与真假判别的通用框架，没有规定现代文本条件生成架构。GigaGAN 保留对抗训练，重新设计了可扩展的文本生成器和判别器。下表概括原论文第 3 节的设计。[论文方法部分](https://arxiv.org/html/2303.05511v2#S3)

| 方面 | 原始 GAN 框架 | GigaGAN |
|---|---|---|
| 条件输入 | 基本形式仅输入噪声 | CLIP 文本特征与噪声共同控制生成 |
| 生成器 | 通用网络 $`G(z)`$ | 风格映射、样本自适应卷积核选择、自注意力和文本交叉注意力 |
| 判别器 | 真实/生成样本判别 | 文本条件判别，多尺度输入、多尺度输出 |
| 训练信号 | 基础对抗目标 | 加入文本匹配、CLIP 对比与视觉辅助对抗信号 |
| 高分辨率 | 框架本身无专门方案 | 另训练 GAN 上采样器 |

理解时把握两条因果链即可：文本交叉注意力让局部图像特征对齐词语；多尺度判别把监督传到不同分辨率，帮助大模型训练。它仍然通过前馈生成器合成图像，基础生成器与上采样器可以分阶段运行；无需像 DDPM 那样执行一长串去噪步骤。

**推导细节与进一步理解**

**怎样把表格组织成解释。** 输入端先利用文本特征与噪声得到风格控制，同时保留词级表示供交叉注意力使用。生成器于是既有整体语义与风格控制，也能把不同空间位置关联到不同词语。自适应卷积核选择可以理解为根据当前样本在多个候选滤波器之间加权，而非所有提示使用完全相同的卷积变换。

判别器除了检查“像不像真实图像”，还要检查“图像与这段文本是否匹配”。多尺度输入和输出把监督传到粗结构及细节；错误配对的图文可作为不匹配信号，CLIP 与视觉辅助信号则引入预训练特征。这解释了为什么仅把原始 GAN 的参数数量放大，并不能自动得到相同能力。

最后区分**对抗框架**与**具体架构**：原始 GAN 是训练原则，GigaGAN 是在这一原则下为大规模文本生成图像设计的系统。两者共有生成器与判别器博弈；文本条件、网络结构、辅助损失与上采样方案才是回答技术区别时应着重比较的部分。

# 第二次作业 AR 与 Transformer 逐题解析

对应原题：[Assignment 2.pdf](../Assignment%202.pdf)。题目约定位置 $`i`$ 的输入包含 $`x_i`$，用来预测 $`x_{i+1}`$；下文的掩码据此允许看当前位。

题干按所附 PDF 原文转写，合并排版断行并重排数学公式；共同题设放在小问之前，每个小问紧接独立解析。

原题课程标记：Peking University Generative Models，Module 2: AR + Transformer；Assignment 2: AR and Transformers。

## 第1题 自回归 Transformer 的因果掩码

**原题（PDF第1页）**

In an autoregressive language model, position $`i`$ uses tokens $`x_1,\ldots,x_i`$ to predict $`x_{i+1}`$. Consider the unmasked attention operation

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V,
\qquad Q,K,V\in\mathbb R^{n\times d},
```

where softmax is applied row-wise.

### 第1题a 无掩码注意力为什么不符合自回归建模

**原题（PDF第1页）**

(a) Explain why unmasked attention operation fails in autoregressive modeling.

**详细解析**

自回归分解要求

```math
p(x_{1:n})=\prod_{i=1}^n p(x_i\mid x_{1:i-1}).
```

训练时整段序列都在内存中。若第 $`i`$ 行注意力可以读取 $`j>i`$ 的 $`K_j,V_j`$，其输出就可能包含 $`x_{i+1}`$，甚至更远处的信息。它在预测 $`x_{i+1}`$ 时看到了答案，造成未来信息泄漏；推理时这些未来 token 尚未生成，训练和推理的条件不一致。


**推导细节与进一步理解**

**一个具体的泄漏例子。** 训练序列是“我／喜欢／数学”，位置 1 用“我”预测“喜欢”。如果第 1 行能关注第 2 位，那么第 2 位的值向量已经由“喜欢”的嵌入计算得到，模型可以把目标信息直接抄回第 1 位。这样的训练损失再低，也不能说明模型学会了仅根据“我”预测下一词。

并行训练本身没有问题：所有真实 token 可以一起放进张量，但计算图必须保证位置 $`i`$ 的输出只依赖 $`x_{1:i}`$。推理时模型手里也恰好只有这个前缀，所以因果性要求是对**信息依赖**的要求，而不是要求训练程序真的一词一词地运行。

### 第1题b 怎样应用掩码以及为什么有效

**原题（PDF第1页）**

(b) Explain how the causal mask is applied and why it prevents attention to future tokens.

**详细解析**

在 softmax **之前**加矩阵 $`M`$：

```math
M_{ij}=\begin{cases}0,&j\leq i,\\-\infty,&j>i,\end{cases}\qquad
O=\mathrm{softmax}_{\rm row}\!\left(\frac{QK^\top}{\sqrt d}+M\right)V.
```

因为 $`e^{-\infty}=0`$，未来位置的权重恰为 0；对合法位置重新归一化后，

```math
O_i=\sum_{j\leq i}\frac{e^{q_i^\top k_j/\sqrt d}}{\sum_{r\leq i}e^{q_i^\top k_r/\sqrt d}}v_j.
```

四个位置的掩码是

```math
M=\begin{pmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{pmatrix}.
```

每一层都使用此掩码，逐 token 的 FFN、LayerNorm 和残差也不跨位置混入未来信息，于是可按层归纳证明第 $`i`$ 位仅依赖 $`x_{1:i}`$。因果掩码和 padding 掩码解决不同问题，需要时应同时使用。仅在 softmax 后乘零而不重新归一化，不等于上述标准注意力。


**推导细节与进一步理解**

**逐步观察 softmax。** 先设分数为 $`s_{ij}`$，加掩码后分子变为 $`e^{s_{ij}+M_{ij}}`$。合法位置乘 $`e^0=1`$，未来位置乘 $`e^{-\infty}=0`$；分母则只剩合法位置分子的和。这样未来位置既不能贡献值向量，也不能通过分母改变合法位置的相对归一化。

**为什么对角线要保留。** 本题是输入到 $`x_i`$、预测 $`x_{i+1}`$，所以允许 $`j=i`$。若另一种数据对齐是第 $`i`$ 位直接预测输入中的 $`x_i`$，必须相应移位或使用严格下三角掩码，不能不检查输入目标对齐方式就照抄掩码。

**为什么不能只屏蔽最后一层。** 如果较早层的第 $`j`$ 个隐藏向量已经读取未来信息，后层即使只看过去位置，也可能从这些被污染的向量间接取得未来内容。必须从输入开始保证每一层的依赖都合法。实现中用足够小的有限数近似 $`-\infty`$ 时还要留意数据类型；此外，不应制造整行全为 $`-\infty`$ 的有效查询，否则 softmax 可能出现未定义值。

## 第2题 自注意力计算与复杂度

**原题（PDF第1页）**

Let $`X\in\mathbb R^{n\times d}`$ be a sequence of $`n`$ token representations. Consider a single-head self-attention layer with projection matrices $`W_Q,W_K,W_V\in\mathbb R^{d\times d}`$. Ignore biases and the output projection.

### 第2题a 公式与维度

**原题（PDF第1页）**

(a) Write down the computation of $`Q,K,V`$ and the scaled dot-product attention output. Specify the dimension of each intermediate matrix and the axis along which softmax is applied.

**详细解析**

给定 $`X\in\mathbb R^{n\times d}`$，$`W_Q,W_K,W_V\in\mathbb R^{d\times d}`$：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
```

```math
S=\frac{QK^\top}{\sqrt d},\qquad A=\mathrm{softmax}_{\rm row}(S),\qquad O=AV.
```

| 矩阵 | 维度 | 意义 |
|---|---|---|
| $`Q,K,V`$ | $`n\times d`$ | 每个位置的查询、键、值 |
| $`S`$ | $`n\times n`$ | 行 $`i`$ 查询与列 $`j`$ 键的相似度 |
| $`A`$ | $`n\times n`$ | 每行沿键位置 $`j`$ 做 softmax，行和为 1 |
| $`O`$ | $`n\times d`$ | 对值向量的加权求和 |

缩放 $`1/\sqrt d`$ 可控制点积随维度增长的典型幅度，减少 softmax 过度饱和。


**推导细节与进一步理解**

**每一行、每一列究竟代表什么。** $`Q_i`$ 是位置 $`i`$ 提出的查询；$`K_j`$ 负责说明位置 $`j`$ 有什么可匹配的特征；$`V_j`$ 是匹配后真正传递的信息。$`QK^\top`$ 的 $`(i,j)`$ 项因此一定是 $`q_i^\top k_j`$。若把 softmax 错做成按列归一化，就会变成“所有查询如何分配到同一个键”，不再是每个查询从所有键里汇总信息。

可按矩阵乘法检查输出：$`A_{n\times n}V_{n\times d}`$ 的第 $`i`$ 行是 $`\sum_ja_{ij}V_j`$，仍为 $`d`$ 维。注意力改变了每个位置包含的信息，不要求改变序列长度或隐藏维数。

**缩放的方差解释。** 若初始化时查询与键各坐标近似独立、零均值、方差 1，则每个乘积 $`q_\ell k_\ell`$ 的均值为 0、方差为 1，点积的方差是 $`d`$。除以 $`\sqrt d`$ 后方差回到 1 的量级。这里是帮助理解初始化的近似，不要求训练后的坐标始终独立或精确服从高斯。

### 第2题b 时间与内存复杂度

**原题（PDF第1页）**

(b) Analyze the time complexity in terms of $`n`$ and $`d`$, distinguishing the projections from the attention computation. What is the memory complexity of explicitly storing the attention matrix?

**详细解析**

| 操作 | 时间复杂度 |
|---|---|
| 三次投影 $`XW_Q,XW_K,XW_V`$ | $`O(nd^2)`$ |
| 分数 $`QK^\top`$ | $`O(n^2d)`$ |
| softmax | $`O(n^2)`$ |
| 加权求和 $`AV`$ | $`O(n^2d)`$ |

因此总时间为 **$`O(nd^2+n^2d)`$**。只有在相应项占优时才能简写成 $`O(n^2d)`$，不能一概假设 $`n\gg d`$。显式存储注意力矩阵占 **$`O(n^2)`$** 内存；连同 $`Q,K,V,O`$，激活内存为 $`O(n^2+nd)`$，不计参数和优化器状态。固定层数、固定总维度的多头注意力保持相同主阶；若逐头存注意力，相关存储另有头数因子。


**推导细节与进一步理解**

**把 FLOPs 的来源数出来。** 一个 $`n\times d`$ 矩阵与 $`d\times d`$ 矩阵相乘，有 $`nd`$ 个输出，每个输出累积 $`d`$ 项，所以是 $`O(nd^2)`$。而注意力有 $`n^2`$ 对查询与键，每对做一次长度 $`d`$ 的点积，所以是 $`O(n^2d)`$；之后 $`AV`$ 有 $`nd`$ 个输出，每个累积 $`n`$ 项，也得到 $`O(n^2d)`$。

三次投影和两次注意力矩阵乘法的乘加次数量级可写为 $`3nd^2+2n^2d`$，未计常数倍的乘法与加法定义差异。当 $`n`$ 远小于 $`d`$ 时，投影项可能占优；当 $`n`$ 很大时，二次注意力项逐渐成为主要负担。

**内存的直观数量。** 单头长度 $`n=4096`$ 的注意力矩阵有 $`4096^2`$ 个元素，float32 存储约需 64 MiB，仅这一个矩阵就已经很大。训练还可能保存分数、概率和反向传播中间量，所以实际开销更高。题目只问“显式注意力矩阵”的内存时，回答 $`O(n^2)`$ 即可，不应把模型权重的 $`O(d^2)`$ 混入这一项。

## 第3题 RoPE 的相对位置性质

**原题（PDF第1页）**

Consider the standard, unscaled RoPE implementation with head dimension $`d=2h`$ and a fixed base $`b>1`$ (e.g., $`b=10000`$). Let $`\theta=(b^{-2\ell/d})_{\ell=0}^{h-1}\in\mathbb R^h`$. For one token at position $`t`$, split $`x\in\mathbb R^d`$ into its first and second halves $`x^{(1)},x^{(2)}\in\mathbb R^h`$ and define

```math
x=\begin{pmatrix}x^{(1)}\\x^{(2)}\end{pmatrix},\qquad
\mathrm{rotate\_half}(x)=\begin{pmatrix}-x^{(2)}\\x^{(1)}\end{pmatrix},\qquad
e_t=\begin{pmatrix}t\theta\\t\theta\end{pmatrix}.
```

The implementation applies RoPE to queries and keys using

```math
f(x,t)=x\odot\cos(e_t)+\mathrm{rotate\_half}(x)\odot\sin(e_t),
```

where $`\odot`$ denotes elementwise multiplication and sine and cosine are applied elementwise. For arbitrary query and key vectors $`q_i,k_j\in\mathbb R^d`$ at positions $`i`$ and $`j`$, prove in exact arithmetic that

```math
f(q_i,i)^\top f(k_j,j)=q_i^\top M(i-j)k_j,
```

where $`M(\delta)\in\mathbb R^{d\times d}`$ is independent of $`q_i,k_j`$ and depends on the positions only through $`\delta=i-j`$. Derive the explicit form of $`M(\delta)`$.

**详细解析**

**题意。** 使用题目指定的前半后半配对布局，而非相邻元素配对，证明旋转后的点积仅通过位置差依赖 $`i,j`$，并求 $`M(i-j)`$。

令 $`d=2h`$，$`\theta_\ell=b^{-2\ell/d}`$，并记

```math
C_t=\mathrm{diag}(\cos(t\theta_0),\ldots,\cos(t\theta_{h-1})),\quad
S_t=\mathrm{diag}(\sin(t\theta_0),\ldots,\sin(t\theta_{h-1})).
```

由题目的 $`\mathrm{rotate\_half}`$ 定义，

```math
f(x,t)=R(t)x,\qquad
R(t)=\begin{pmatrix}C_t&-S_t\\S_t&C_t\end{pmatrix}.
```

每一对 $`(x_\ell,x_{\ell+h})`$ 都是二维旋转，因此 $`R(t)^\top=R(-t)`$。直接用三角和差公式可验证

```math
R(s)R(t)=R(s+t).
```

于是

```math
f(q_i,i)^\top f(k_j,j)=q_i^\top R(i)^\top R(j)k_j
=q_i^\top R(j-i)k_j.
```

题目把位置差定义成 $`\delta=i-j`$，所以答案是

```math
\boxed{M(\delta)=R(-\delta)=
\begin{pmatrix}
C_\delta&S_\delta\\
-S_\delta&C_\delta
\end{pmatrix}.}
```

显式展开为

```math
\sum_{\ell=0}^{h-1}\left[
(q_{i,\ell}k_{j,\ell}+q_{i,\ell+h}k_{j,\ell+h})\cos(\delta\theta_\ell)
+(q_{i,\ell}k_{j,\ell+h}-q_{i,\ell+h}k_{j,\ell})\sin(\delta\theta_\ell)
\right].
```

**易错点。** $`M(i-j)`$ 是 $`R(j-i)`$；直接把 $`M(\delta)`$ 写成 $`R(\delta)`$ 会把正弦项符号弄反。这里只证明位置运算的显式依赖是相对位置；$`q_i,k_j`$ 本身仍可包含内容和上下文信息。精确算术下恒等式成立，浮点实现允许舍入差异。


**推导细节与进一步理解**

**把矩阵乘法中的符号也展开。** 由块矩阵相乘，

```math
R(i)^\top R(j)=
\begin{pmatrix}
C_iC_j+S_iS_j&-C_iS_j+S_iC_j\\
-S_iC_j+C_iS_j&S_iS_j+C_iC_j
\end{pmatrix}.
```

这些矩阵都是同一组坐标上的对角矩阵，因此彼此可交换。令 $`\delta=i-j`$，余弦差角公式给出左上、右下块 $`C_\delta`$，正弦差角公式给出右上块 $`S_\delta`$、左下块 $`-S_\delta`$。这就直接得到前面的 $`M(\delta)`$，无需仅凭记忆判断旋转方向。

**最小例子检查正负号。** 取 $`d=2`$，频率 $`\theta_0=1`$，$`q=(1,0)^\top`$、$`k=(0,1)^\top`$，位置 $`i=1,j=0`$。旋转后 $`f(q,1)=(\cos1,\sin1)^\top`$、$`f(k,0)=(0,1)^\top`$，点积为 $`\sin1`$。因此 $`M(1)`$ 的右上角必须是 $`+\sin1`$，若写成负号就与直接计算矛盾。

**为什么题目强调前后两半。** 这里配对的是第 $`\ell`$ 与第 $`\ell+h`$ 个坐标。另一类实现配对相邻坐标，在经过一个固定坐标置换后可以表达相同的二维旋转结构，但直接写出的 $`d\times d`$ 矩阵布局不同。本题必须按指定的 rotate half 实现给出答案。

## 第4题 FlashAttention 的原理

**原题（PDF第1页）**

Paper: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.

Explain the main idea of FlashAttention and why it can compute exact attention faster and with less memory than a standard implementation.

**详细解析**

标准实现把 $`n\times n`$ 分数和概率矩阵写入 GPU 显存，再读回计算输出。FlashAttention 将 $`Q,K,V`$ 分块放入片上 SRAM，融合局部点积、softmax 与加权求和，并在线合并归一化统计量，避免在显存中存完整注意力矩阵。反向传播按块重算所需概率，以计算换取更少的内存访问。[原论文第 3 节](https://arxiv.org/html/2205.14135v2#S3)

在线 softmax 为什么仍精确，可以用一行分数解释。设当前已处理的最大值为 $`m`$，保存

```math
\ell=\sum_{j\in\text{已处理}}e^{s_j-m},\qquad
u=\sum_{j\in\text{已处理}}e^{s_j-m}v_j.
```

新分块的分数为 $`s_j`$，令

```math
m'=\max(m,\max_{j\in\text{新块}}s_j).
```

合并规则是

```math
\ell'=e^{m-m'}\ell+\sum_{j\in\text{新块}}e^{s_j-m'},
```

```math
u'=e^{m-m'}u+\sum_{j\in\text{新块}}e^{s_j-m'}v_j.
```

所有块处理完，输出 $`u/\ell`$，正好是完整 softmax 的加权和。这是对求和与归一化的重排。

因此，稠密注意力的算术主阶仍为 $`O(n^2d)`$；额外注意力存储随序列长度线性增长，且显存与 SRAM 之间的数据传输显著减少。实际提速来自访存与融合计算。这里的“精确”指计算同一个稠密注意力函数，浮点运算顺序改变不保证逐比特一致；稀疏注意力是另外的扩展。

**推导细节与进一步理解**

**在线归一化的一个算例。** 一行注意力分数取 $`(0,\log2,\log3)`$，对应标量值为 $`(10,20,30)`$。直接 softmax 得权重 $`(1,2,3)/6`$，输出是 $`140/6`$。先处理前两项，保存 $`m=\log2`$、$`\ell=1/2+1=3/2`$、$`u=10/2+20=25`$。第三项到来后最大值变为 $`m'=\log3`$，旧累积量乘 $`e^{m-m'}=2/3`$，于是

```math
\ell'=\frac23\cdot\frac32+1=2,\qquad
u'=\frac23\cdot25+30=\frac{140}3,\qquad
\frac{u'}{\ell'}=\frac{140}6.
```

可见不需要保留此前每一个概率，只要保存最大值、归一化和、加权和，就能合并任意数量的块。最大值平移还防止直接指数运算溢出。

**“多算一点反而更快”的原因。** 标准实现通常先把大矩阵写到高带宽显存，再从显存读出。FlashAttention 把适合当前块的数据留在更快的片上存储，并在反向时按需要重算部分注意力。这可能增加部分算术操作，却节省大量昂贵的数据搬运。在访存是瓶颈时，总运行时间因此下降。

**完整答案应同时交代三点。** 它保持所有允许的查询与键交互；通过分块和在线 softmax 避免完整中间矩阵；通过融合与重计算减少显存读写。不能只写“用了更少内存”，也不能写成“把稠密注意力的 $`n^2`$ 次配对变成了 $`n`$ 次”。

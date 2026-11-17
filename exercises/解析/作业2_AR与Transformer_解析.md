# 第二次作业 AR 与 Transformer 逐题解析

对应原题：[Assignment 2.pdf](../Assignment%202.pdf)。题目约定位置 $`i`$ 的输入包含 $`x_i`$，用来预测 $`x_{i+1}`$；下文的掩码据此允许看当前位。

## 第1题 自回归 Transformer 的因果掩码

### 第1题a 无掩码注意力为什么不符合自回归建模

自回归分解要求

```math
p(x_{1:n})=\prod_{i=1}^n p(x_i\mid x_{1:i-1}).
```

训练时整段序列都在内存中。若第 $`i`$ 行注意力可以读取 $`j>i`$ 的 $`K_j,V_j`$，其输出就可能包含 $`x_{i+1}`$，甚至更远处的信息。它在预测 $`x_{i+1}`$ 时看到了答案，造成未来信息泄漏；推理时这些未来 token 尚未生成，训练和推理的条件不一致。

### 第1题b 怎样应用掩码以及为什么有效

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

## 第2题 自注意力计算与复杂度

### 第2题a 公式与维度

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

### 第2题b 时间与内存复杂度

| 操作 | 时间复杂度 |
|---|---|
| 三次投影 $`XW_Q,XW_K,XW_V`$ | $`O(nd^2)`$ |
| 分数 $`QK^\top`$ | $`O(n^2d)`$ |
| softmax | $`O(n^2)`$ |
| 加权求和 $`AV`$ | $`O(n^2d)`$ |

因此总时间为 **$`O(nd^2+n^2d)`$**。只有在相应项占优时才能简写成 $`O(n^2d)`$，不能一概假设 $`n\gg d`$。显式存储注意力矩阵占 **$`O(n^2)`$** 内存；连同 $`Q,K,V,O`$，激活内存为 $`O(n^2+nd)`$，不计参数和优化器状态。固定层数、固定总维度的多头注意力保持相同主阶；若逐头存注意力，相关存储另有头数因子。

## 第3题 RoPE 的相对位置性质

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

## 第4题 FlashAttention 的原理

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

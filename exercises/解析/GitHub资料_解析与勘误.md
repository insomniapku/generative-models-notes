# GitHub 同课程资料解析与勘误

检索与下载日期：2026 年 11 月 17 日。检索结果包括一份 2024 期中民间解析和一份 2025 秋课程笔记；**本次没有找到可核实、且相对本地资料新增的独立作业题单或期中试卷**。这仅描述本次公开检索结果，不代表 GitHub 上绝对不存在其他资料。

## 已下载的同课程资料

| 来源 | 课程归属依据 | 本地内容 | 与现有试题的关系 |
|---|---|---|---|
| [joerunrun/generative_model](https://github.com/joerunrun/generative_model) | README 明写北大生成模型基础、2025 课程和 2024 期中试题 | [8页解析 PDF](../GitHub资料/joerunrun__generative_model/midtest.pdf)、[LaTeX 源码](../GitHub资料/joerunrun__generative_model/midtest.tex)、README | 民间答案，题目对应本地 2024 卷，不是一份新试卷 |
| [TidalHarley/PKU-Genai-notes](https://github.com/TidalHarley/PKU-Genai-notes) | README 标明 Peking University、2025 Fall、生成模型基础 | [20页课程笔记 PDF](../GitHub资料/TidalHarley__PKU-Genai-notes/生成模型基础%20Basic%20of%20Generative%20Model%20(25%20Fall).pdf)、README | 复习补充，未发现独立的作业或试题题单 |

下载的原件保持原样，纠错集中写在本文与逐题解析中。文件的下载 URL、大小和 SHA256 保存在 [downloads.json](../GitHub资料/downloads.json)。GitHub 接口不支持二进制下载，因此 PDF 使用其公开 raw 链接下载，并检查 PDF 文件头及能否正常打开。

## 2024 期中民间解析的逐题使用指南

这份解答覆盖的正是本地 2024 卷的 15 个小问。每问的完整推导已经放在 [2024期中逐题解析](期中2024_逐题解析.md)，可按相同题号阅读，无需重复刷一套同题。

| 民间解答题号 | 正确结论或阅读重点 | 本地完整解析位置 |
|---|---|---|
| 1a | 区分总运算量与训练并行性；Transformer 的总生成 $`O(n^2)`$ 需说明 KV cache | 2024 第1题a |
| 1b | 无位置机制的全连接注意力是排列等变 | 2024 第1题b |
| 1c | 行 softmax 后加权求和值向量 | 2024 第1题c |
| 1d | 四个输出为 $`(1,-1),(-1,1),(1,-1),(1,-1)`$ | 2024 第1题d |
| 2a | ELBO 与 log likelihood 的差为到真实后验的 KL | 2024 第2题a |
| 2b | 参数无关噪声加可微变换，得到路径梯度 | 2024 第2题b |
| 3a | 判别器最大化、生成器最小化，推理仅使用生成器 | 2024 第3题a |
| 3b | $`f(u)=u\log u`$，凸且 $`f(1)=0`$；它不是处处非负函数 | 2024 第3题b |
| 4a | 前向 Jacobian 的绝对行列式在分母 | 2024 第4题a |
| 4b | 可逆线性映射加适当值域上的双射激活 | 2024 第4题b |
| 4c | 精确负对数似然训练，基础分布采样后前向生成 | 2024 第4题c |
| 5a | $`x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t`$ | 2024 第5题a |
| 5b | 条件方差为 $`1-\bar\alpha_t`$ | 2024 第5题b |
| 5c | 后验精度中必须保留 $`\alpha_t/\beta_t`$ | 2024 第5题c |
| 5d | 噪声 MSE 训练与反向高斯采样 | 2024 第5题d |

### 关键错误 后验精度漏乘单步信号系数

民间 PDF 第 7 页及源码第5题c把二次项精度写成

```math
\frac1{\widetilde\beta_t}=\frac1{\beta_t}+\frac1{1-\bar\alpha_{t-1}}.
```

正确式应当是

```math
\boxed{\frac1{\widetilde\beta_t}=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}}.}
```

原因是 $`\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2`$ 展开时，$`x_{t-1}^2`$ 的系数为 $`\alpha_t`$。原文随后写出的最终方差公式是正确的，但它不能由其错误的中间式推出。

例如 $`\beta_t=0.2,\alpha_t=0.8,\bar\alpha_{t-1}=0.6`$ 时，正确精度为 $`0.8/0.2+1/0.4=6.5`$，后验方差为 $`1/6.5\approx0.153846`$；漏乘后变成 $`1/0.2+1/0.4=7.5`$，对应方差约 $`0.133333`$。

### 其他表述需要收紧

民间答案把严格单调直接等同于 $`\mathbb R`$ 上的双射；严格单调只保证对其像集可逆，仍需检查值域。LeakyReLU 的非零负斜率可保证双射，但在 0 不光滑；如果严格按光滑题干作答，可以采用本地解析给出的 $`\sinh`$ 激活。

其 f 散度证明列表中的“non negative”若指 $`f(u)`$ 则不成立，因为 $`u\log u\lt 0`$ 对 $`0\lt u\lt 1`$ 成立。生成函数无需处处非负。

## 2025 秋课程笔记的复习对应关系

| PDF 页码 | 主题 | 逐题解析中可配套阅读的位置 |
|---|---|---|
| 1至4 | AE、VAE、VQ VAE | 作业1；两年期中第2题 |
| 4至6 | Transformer、注意力、位置编码 | 作业2；两年期中第1题 |
| 7至9 | 自回归、采样、复杂度 | 2024第1题；2025第1题；下文采样勘误 |
| 9至10 | 调试 | 本次题单无独立调试题，可配合现有 ppt 笔记复习 |
| 11至13 | GAN、Wasserstein、WGAN | 作业3；两年期中第3题 |
| 14至15 | 归一化流 | 两年期中第4题 |
| 16至18 | 扩散模型 | 作业4；两年期中第5题 |
| 18至20 | 推测解码、剪枝、量化、蒸馏 | 拓展内容，本次试题未单独考这些题 |

这份笔记的正文还存在几处会影响答题的错误；以下是对涉及公式和定义的补充解释。

### Top k 与 Top p 按截断后概率采样

第 9 页把 top k 说成在前 $`k`$ 个词中均匀采样，把 top p 说成挑单个概率大于阈值的词再均匀采样；两者都不准确。

对概率 $`p_1,\ldots,p_V`$，top k 保留概率最大的 $`k`$ 个候选组成集合 $`A`$；top p 按概率降序取累计概率达到阈值 $`p`$ 的最小前缀集合 $`A`$。两者随后通常均按

```math
\boxed{p'_i=\frac{p_i\mathbf1_{i\in A}}{\sum_{j\in A}p_j}}
```

采样，保留相对概率，而非均匀采样。例如 $`(0.50,0.30,0.15,0.05)`$ 在 top k 的 $`k=2`$ 下变成 $`(0.625,0.375,0,0)`$；top p 的阈值 0.8 也会保留前两个。[原论文第 3 节](https://arxiv.org/html/1904.09751v2#S3)

### KV cache 不把完整生成降为线性

第 9 页声称 KV cache 可把注意力推理时间降到 $`O(n)`$，这需要改成：**在已有长度为 $`n`$ 的缓存时，处理一个新 token 的注意力交互是 $`O(nd)`$**；生成 $`n`$ 个 token 时累计交互仍为 $`O(n^2d)`$。它避免重算旧 token 的键值及相关前缀计算，并不让每个新查询免于读取历史键值。同页将 $`AV`$ 的乘法记为 $`O(n)`$ 也不正确，稠密实现应为 $`O(n^2d)`$。

### VQ VAE 选最近码字并区分两类损失

第 4 页正文出现 argmax，但最近邻应为

```math
k=\arg\min_j\|z_e-e_j\|_2^2.
```

直通表示 $`z_{\rm ST}=z_e+\mathrm{stopgrad}(e_k-z_e)`$ 在前向取码字，反向将重建梯度传给编码器。经典损失写为

```math
\boxed{L=L_{\rm recon}
+\|\mathrm{stopgrad}(z_e)-e_k\|_2^2
+\beta\|z_e-\mathrm{stopgrad}(e_k)\|_2^2.}
```

第二项更新码本，第三项为带权重 $`\beta`$ 的 commitment 项，更新编码器。笔记文字把这两项的标准系数位置交换，且有处遗漏平方；配套截图反而展示了标准方向。应以梯度流向识别，而不只靠损失的名字。[VQ VAE 原论文第 3.2 节](https://arxiv.org/html/1711.00937v2#S3.SS2)

### VAE 先验与重参数化的准确含义

第 2 至 3 页将问题解释成“高斯分布不能求导”“$`p(z)`$ 是复杂向量”或固定 $`z`$ 只有少量 $`x`$ 的似然非零，均不能作为 VAE 的一般前提。通常先验 $`p(z)`$ 是容易采样和计算的标准高斯；困难来自 $`p_\theta(x)=\int p(z)p_\theta(x\mid z)dz`$ 中高维、非线性的积分和后验推断。高斯密度可对参数求导，重参数化提供的是对期望的路径梯度；高斯解码似然通常有全空间支持。

正确的基本关系见作业1第1题和第3题。无需上述支持稀疏假设就能推导 ELBO。

### 正交矩阵的行列式及逆运算成本

第 15 页应写成 $`A^{-1}=A^\top`$、$`|\det A|=1`$，而非总有 $`\det A=1`$。例如 $`A=\mathrm{diag}(-1,1)`$ 为正交矩阵，但行列式是 $`-1`$。因此它的 log absolute determinant 为 0。

普通稠密三角矩阵的**求解** $`Az=b`$ 可在 $`O(d^2)`$ 完成；不应把它误写成完整求逆矩阵只需 $`O(d^2)`$。正交矩阵转置易得，但对稠密矩阵执行 $`A^\top x`$ 通常仍需 $`O(d^2)`$ 运算。

### 扩散边缘与似然上界

第 16 页的“最终都会达到标准高斯”在有限步中应理解为日程使累计信号量很小时的近似；除非满足特殊条件，有限步不恰好等于标准高斯。第 17 页的目标 $`\mathbb E[-\log p_\theta(x_0)]`$ 是负对数似然的期望，不能解释为直接平均概率。第 18 页常用无权重噪声 MSE 与完整变分目标的权重、端点项有区别。[DDPM 原论文第 3 节](https://arxiv.org/html/2006.11239v2#S3)

## 检索范围与排除项

本次使用 GitHub 仓库、代码搜索及网页检索，包含“生成模型基础”“PKU generative”“Generative Modeling Peking”“Generative_Model_Midterm”“GenModelExercise”等关键词，并读取候选仓库 README 和目录核对归属。

| 候选仓库 | 核对结果 | 本次处理 |
|---|---|---|
| [RinShiina/PKU-25Fall-Generative_Modeling](https://github.com/RinShiina/PKU-25Fall-Generative_Modeling) | 目录为 ShallowDiffuse 项目与报告，无本次目标中的独立作业题单或期中卷 | 留作项目线索，未下载整仓库 |
| [41xu/Deep-Generative-Models-Practice](https://github.com/41xu/Deep-Generative-Models-Practice) | README 指向北大 2021 春深度生成模型课程与另一课程网站 | 年份和课程归属不同，不当作贺笛本课程的往年题 |
| [WeqZhang/MLCoursework](https://github.com/WeqZhang/MLCoursework) | README 明写东北大学机器学习作业 | 排除 |
| [insomniapku/generative-models-notes](https://github.com/insomniapku/generative-models-notes) | 目录与本地 ppt 的九章笔记对应，无独立题单 | 避免重复下载 |
| [PoplarCrystal/generative_model](https://github.com/PoplarCrystal/generative_model) | 树中有学习材料与模型代码，未核实为同一课程的往年题 | 不纳入同课程题库 |

因此本地原题数量仍是五份作业与两份期中试卷；新增的是两份公开参考资料及其解释、勘误，而不是额外一套独立真题。

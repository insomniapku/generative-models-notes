"""Independent numerical checks for the study solutions (Python + NumPy only)."""
from pathlib import Path
from itertools import product, permutations
from urllib.parse import unquote
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261117)
RESULTS = []


def record(name, error, tolerance, description):
    error = float(error)
    passed = bool(np.isfinite(error) and error <= tolerance)
    RESULTS.append(dict(name=name, passed=passed, error=error,
                        tolerance=tolerance, method=description))
    print(('PASS ' if passed else 'FAIL ') + name + f': {error:.3g}', flush=True)


def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def gaussian_logpdf(z, mean, cov):
    delta = z - mean
    return -.5 * (len(mean) * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]
                  + np.sum(delta * np.linalg.solve(cov, delta.T).T, axis=-1))


def gaussian_kl():
    mq = np.array([.2, -.3, .5]); mp = np.array([-.1, .4, .2])
    cq = np.array([[1.2, .1, .2], [.1, .8, -.1], [.2, -.1, 1.]])
    cp = np.array([[.9, -.2, .1], [-.2, 1.4, .1], [.1, .1, 1.1]])
    delta = mq - mp
    closed = .5 * (np.trace(np.linalg.solve(cp, cq)) + delta @ np.linalg.solve(cp, delta)
                   - 3 + np.linalg.slogdet(cp)[1] - np.linalg.slogdet(cq)[1])
    nodes, weights = np.polynomial.hermite.hermgauss(8)
    total = 0.
    chol = np.linalg.cholesky(cq)
    for indices in product(range(8), repeat=3):
        z = mq + chol @ (np.sqrt(2) * nodes[list(indices)])
        weight = np.prod(weights[list(indices)]) / np.pi**1.5
        total += weight * (gaussian_logpdf(z, mq, cq) - gaussian_logpdf(z, mp, cp))
    record('一般高斯 KL', abs(total - closed), 1e-12,
           '三维 Gauss Hermite 求积直接积分 log q minus log p，与闭式公式比较')


def reparameterization():
    mu = np.array([.2, -.7, 1.]); r = np.array([-.4, .2, -.1])
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    noise = np.sqrt(2) * nodes
    z = mu[:, None] + np.exp(r[:, None] / 2) * noise
    grad_mu = (4 * z**3) @ weights / np.sqrt(np.pi)
    grad_r = (2 * z**3 * np.exp(r[:, None] / 2) * noise) @ weights / np.sqrt(np.pi)
    def exact(m, logv):
        return np.sum(m**4 + 6 * m**2 * np.exp(logv) + 3 * np.exp(2 * logv))
    fd_mu = []; fd_r = []
    for i in range(3):
        direction = np.eye(3)[i]
        fd_mu.append(derivative(lambda t: exact(mu + t * direction, r), 0.))
        fd_r.append(derivative(lambda t: exact(mu, r + t * direction), 0.))
    record('log variance 重参数化梯度', max(np.max(np.abs(grad_mu - fd_mu)),
                                      np.max(np.abs(grad_r - fd_r))), 1e-7,
           '用四次函数精确高斯矩的有限差分，对照路径梯度的 Gauss Hermite 积分')


def gumbel_max():
    weights = np.array([.1, .2, .3, .4])
    g = -np.log(-np.log(RNG.random((300000, 4))))
    sampled = np.argmax(np.log(weights) + g, axis=1)
    empirical = np.bincount(sampled, minlength=4) / len(sampled)
    record('Gumbel Max 类别频率', np.max(np.abs(empirical - weights)), .004,
           '固定随机种子采样三十万次，与目标类别概率比较；频率检查不代替证明')


def rope():
    error = 0.
    for d in [2, 8, 32]:
        theta = 10000.**(-2 * np.arange(d // 2) / d)
        def rot(t):
            c = np.diag(np.cos(t * theta)); s = np.diag(np.sin(t * theta))
            return np.block([[c, -s], [s, c]])
        def implementation(x, t):
            angles = np.tile(t * theta, 2)
            half = np.concatenate([-x[d // 2:], x[:d // 2]])
            return x * np.cos(angles) + half * np.sin(angles)
        for i, j in [(0, 0), (2, 7), (19, -3), (-8, 15)]:
            q = RNG.normal(size=d); k = RNG.normal(size=d)
            direct = implementation(q, i) @ implementation(k, j)
            formula = q @ rot(-(i - j)) @ k
            error = max(error, abs(direct - formula))
    record('RoPE 相对旋转及符号', error, 1e-12,
           '题目 rotate half 的逐元素实现，对比显式 M(i minus j) 块矩阵')


def online_attention():
    scores = RNG.normal(size=(17, 17)) * 20
    values = RNG.normal(size=(17, 7))
    error = 0.
    for row in range(17):
        legal = scores[row, :row + 1]
        p = np.exp(legal - np.max(legal)); p /= p.sum()
        reference = p @ values[:row + 1]
        m = -np.inf; ell = 0.; u = np.zeros(7)
        for start in range(0, row + 1, 4):
            stop = min(start + 4, row + 1)
            chunk = scores[row, start:stop]
            new_m = max(m, np.max(chunk))
            rescale = np.exp(m - new_m)
            exp_chunk = np.exp(chunk - new_m)
            ell = ell * rescale + exp_chunk.sum()
            u = u * rescale + exp_chunk @ values[start:stop]
            m = new_m
        error = max(error, np.max(np.abs(reference - u / ell)))
    record('FlashAttention 在线归一化', error, 1e-12,
           '四元素分块累计与完整因果注意力直接 softmax 比较，含大幅度分数')


def gan():
    p = RNG.dirichlet(np.ones(13)); q = RNG.dirichlet(np.ones(13)); mix = (p + q) / 2
    d = p / (p + q)
    objective = np.sum(p * np.log(d) + q * np.log1p(-d))
    js = .5 * (np.sum(p * np.log(p / mix)) + np.sum(q * np.log(q / mix)))
    record('最优 GAN 与 JS', abs(objective - (2 * js - 2 * np.log(2))), 1e-12,
           '直接代入最优判别器与独立计算 JS 的结果比较')
    error = 0.
    for a in [-8., -2., 0., 3., 8.]:
        sigmoid = 1 / (1 + np.exp(-a))
        sat = derivative(lambda t: -np.logaddexp(0, t), a)
        ns = derivative(lambda t: np.logaddexp(0, -t), a)
        error = max(error, abs(sat + sigmoid), abs(ns - (sigmoid - 1)))
    record('GAN 两种生成器损失导数', error, 1e-8,
           '对稳定 logit 损失有限差分，对照负 D 与 D minus 1')


def diffusion():
    betas = np.linspace(.001, .2, 30)
    alphas = 1 - betas
    cumulative = np.cumprod(alphas)
    variance = 0.; forward_error = 0.; posterior_error = 0.
    mean_product = 1.
    for i, (beta, alpha, abar) in enumerate(zip(betas, alphas, cumulative)):
        variance = alpha * variance + beta
        mean_product *= np.sqrt(alpha)
        forward_error = max(forward_error, abs(variance - (1 - abar)),
                            abs(mean_product - np.sqrt(abar)))
        if i == 0:
            continue
        old = cumulative[i - 1]
        # Conditional Gaussian from the independently constructed 2x2 joint covariance.
        cov = np.array([[1 - old, np.sqrt(alpha) * (1 - old)],
                        [np.sqrt(alpha) * (1 - old), 1 - abar]])
        joint_mean = np.array([np.sqrt(old) * 1.3, np.sqrt(abar) * 1.3])
        xt = -.7
        conditional_mean = joint_mean[0] + cov[0, 1] / cov[1, 1] * (xt - joint_mean[1])
        conditional_var = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]
        claimed_var = beta * (1 - old) / (1 - abar)
        claimed_mean = (np.sqrt(old) * beta * 1.3 + np.sqrt(alpha) * (1 - old) * xt) / (1 - abar)
        posterior_error = max(posterior_error, abs(claimed_mean - conditional_mean),
                              abs(claimed_var - conditional_var))
    record('DDPM 前向边缘', forward_error, 1e-12,
           '逐步均值方差递推与累计信号公式比较，三十步日程')
    record('扩散高斯后验', posterior_error, 1e-12,
           '独立构建二元高斯协方差，用通用条件高斯公式对照配方结果')
    correct = 1 / (.8 / .2 + 1 / .4)
    wrong = 1 / (1 / .2 + 1 / .4)
    record('网上后验精度勘误反例', abs(correct - .2 * .4 / .52), 1e-12,
           f'正确方差 {correct:.9f}；漏乘系数后的错误方差 {wrong:.9f}，两者不同')


def flow():
    w = np.array([[1.2, 0., 0.], [.3, .8, 0.], [-.2, .5, 1.1]])
    bias = np.array([.1, -.2, .3]); z = np.array([.4, -.3, .1])
    def g(x):
        return np.sinh(w @ x + bias)
    x = g(z)
    inverse = np.linalg.solve(w, np.arcsinh(x) - bias)
    fd_jac = np.column_stack([derivative(lambda t: g(z + t * np.eye(3)[i]), 0.) for i in range(3)])
    closed = np.linalg.slogdet(w)[1] + np.log(np.cosh(w @ z + bias)).sum()
    record('可逆 MLP 与 Jacobian', max(np.max(np.abs(inverse - z)),
                                    abs(np.linalg.slogdet(fd_jac)[1] - closed)), 1e-8,
           '正逆复合与数值 Jacobian 行列式检查，激活为 sinh')
    def params(x):
        mus = np.array([.1, .2 * x[0], .3 * x[0] - .4 * x[1]])
        scales = np.exp([.2, .1 * x[0], -.15 * x[1]])
        return mus, scales
    def maf(z):
        x = np.zeros(3)
        for i in range(3):
            mus, scales = params(x)
            x[i] = mus[i] + scales[i] * z[i]
        return x
    x = maf(z); mus, scales = params(x)
    inverse = (x - mus) / scales
    jac = np.column_stack([derivative(lambda t: maf(z + t * np.eye(3)[i]), 0.) for i in range(3)])
    record('MAF 逆映射与 log determinant', max(np.max(np.abs(inverse - z)),
                                             abs(np.linalg.slogdet(jac)[1] - np.log(scales).sum())),
           1e-8, '真实顺序采样后取数值 Jacobian，与逆映射及对角积比较')


def normalization():
    x = np.array([[2., 1.], [0., 2.], [2., 0.], [4., 1.]])
    out = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    expected = np.array([[1., -1.], [-1., 1.], [1., -1.], [1., -1.]])
    record('两年期中 LayerNorm', np.max(np.abs(out - expected)), 1e-12,
           '四个 token 沿各自特征维使用总体方差计算')


def policy_gradient():
    rewards = np.array([-.3, 1.2, 2.1]); theta = np.array([.2, -.4, .6])
    def probabilities(t):
        e = np.exp(t - max(t)); return e / e.sum()
    p = probabilities(theta)
    score_grad = p * (rewards - p @ rewards)
    fd_grad = np.array([derivative(lambda t: probabilities(theta + t * np.eye(3)[i]) @ rewards, 0.)
                        for i in range(3)])
    record('对数导数奖励梯度', np.max(np.abs(fd_grad - score_grad)), 1e-8,
           '三类别 softmax 期望奖励有限差分，与 score function 梯度比较')
    ref = np.array([.2, .5, .3]); beta = .7
    z = np.sum(ref * np.exp(rewards / beta)); star = ref * np.exp(rewards / beta) / z
    error = 0.; largest = -np.inf
    for i in range(41):
        for j in range(41 - i):
            candidate = np.array([i, j, 40 - i - j]) / 40
            nz = candidate > 0
            kl_ref = np.sum(candidate[nz] * np.log(candidate[nz] / ref[nz]))
            kl_star = np.sum(candidate[nz] * np.log(candidate[nz] / star[nz]))
            value = candidate @ rewards - beta * kl_ref
            error = max(error, abs(value - (beta * np.log(z) - beta * kl_star)))
            largest = max(largest, value)
    record('KL 正则策略恒等式', error, 1e-12,
           '遍历三类别单纯形网格，含边界零概率；每点比较两种目标写法')
    score = np.array([-.5, .5]); prob = np.array([.5, .5])
    baseline_updates = (np.ones(2) - 1) * score
    record('常数奖励基线', max(abs(prob @ score), np.max(np.abs(baseline_updates))), 1e-12,
           'Bernoulli theta 等于零时的两个样本，验证期望零与基线逐样本零')


def transport():
    a = np.array([[-2., 0.], [0., 0.], [2., 0.]])
    b = np.array([[0., 1.], [-.866, -.5], [.866, -.5]])
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    costs = []
    for perm in permutations(range(3)):
        gamma = np.eye(3)[list(perm)].T / 3
        # All permutation matrices are feasible; the best gives an assignment solution.
        costs.append(np.sum(cost * gamma))
    direct = min(sum(cost[i, perm[i]] for i in range(3)) / 3 for perm in permutations(range(3)))
    record('最优运输与指派', abs(min(costs) - direct), 1e-12,
           '三人情形穷举排列，对比运输矩阵代价与平均欧氏距离')


def files_and_coverage():
    specs = [
        ('作业1_VAE_解析.md', ['第1题', '第2题', '第3题', '第4题']),
        ('作业2_AR与Transformer_解析.md', ['第1题a', '第1题b', '第2题a', '第2题b', '第3题', '第4题']),
        ('作业3_GAN_解析.md', ['第1题a', '第1题b', '第1题c', '第2题', '第3题']),
        ('作业4_扩散模型_解析.md', ['第1题a', '第1题b', '第2题', '第3题a', '第3题b', '第3题c']),
        ('作业5_语言模型微调_解析.md', ['第1题', '第2题', '第3题a', '第3题b', '第3题c', '第4题a', '第4题b']),
        ('期中2024_逐题解析.md', ['第1题a', '第1题b', '第1题c', '第1题d', '第2题a', '第2题b',
                                 '第3题a', '第3题b', '第4题a', '第4题b', '第4题c',
                                 '第5题a', '第5题b', '第5题c', '第5题d']),
        ('期中2025_逐题解析.md', ['第0题', '第1题a', '第1题b', '第1题c', '第2题a', '第2题b',
                                 '第3题a', '第3题b', '第4题a', '第4题bi', '第4题bii', '第4题biii',
                                 '第5题a', '第5题b', '第5题c']),
    ]
    coverage = []
    for name, units in specs:
        text = (ROOT / name).read_text(encoding='utf-8')
        headings = re.findall(r'^#{2,3} (.+)$', text, re.M)
        found = [bool(any(h.startswith(unit + ' ') for h in headings)) for unit in units]
        coverage.append(dict(file=name, units=[dict(id=u, present=p) for u, p in zip(units, found)],
                             count=len(units), all_present=all(found)))
    (ROOT / 'coverage.json').write_text(json.dumps(dict(total=sum(x['count'] for x in coverage),
        files=coverage), ensure_ascii=False, indent=2), encoding='utf-8')
    record('58个原题作答单元覆盖', sum(not u['present'] for x in coverage for u in x['units']), 0,
           '按原 PDF 人工列出题号，再逐一检查相应标题是否存在；正文另经人工核对')
    broken = []
    unbalanced = []
    for path in ROOT.glob('*.md'):
        text = path.read_text(encoding='utf-8')
        if text.count('$$') % 2:
            unbalanced.append(path.name)
        fence = None
        for line in text.splitlines():
            if line.startswith('```'):
                fence = None if fence else line.strip()
        if fence:
            unbalanced.append(path.name + ': unclosed fence')
        # Handles ordinary relative paths and paths containing balanced parentheses.
        for target in re.findall(r'\]\(((?:[^()]|\([^()]*\))+)\)', text):
            if re.match(r'^[a-z]+://', target) or target.startswith('#'):
                continue
            target_path = unquote(target.split('#', 1)[0])
            if not (path.parent / target_path).exists():
                broken.append((path.name, target))
    record('本地链接完整性', len(broken), 0, '检查全部 Markdown 文档的相对文件链接：' + str(broken))
    record('块公式分隔符配对', len(unbalanced), 0, '检查双美元分隔符和 Markdown 公式块均成对：' + str(unbalanced))
    manifest = json.loads((ROOT.parent / 'GitHub资料/downloads.json').read_text(encoding='utf-8'))
    hash_errors = []
    for item in manifest:
        target = ROOT.parent / 'GitHub资料' / item['local']
        if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            hash_errors.append(item['local'])
    record('GitHub 下载 SHA256', len(hash_errors), 0, '逐一复算五份下载文件的内容哈希')


def write_report():
    failed = [x for x in RESULTS if not x['passed']]
    (ROOT / 'verification.json').write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ['# 公式与文件核验记录', '', '固定随机种子为 20261117，运行命令为 `python verify_math.py`。'
            '使用 Python 与 NumPy；这是数值及文件检查，数学证明见逐题解析。', '',
            '| 检查 | 结果 | 最大误差或错误数 | 容差 | 方法 |', '|---|---|---:|---:|---|']
    for x in RESULTS:
        rows.append(f"| {x['name']} | {'通过' if x['passed'] else '失败'} | {x['error']:.3g} | {x['tolerance']:.3g} | {x['method']} |")
    rows.extend(['', f'共 {len(RESULTS)} 项检查，{len(RESULTS)-len(failed)} 项通过，{len(failed)} 项失败。', '',
                 '覆盖检查核对题号存在性，不能自动证明正文完整或正确；已结合原页阅读与独立推导核对。'
                 'Gumbel 采样使用有限频率容差，其余代数公式使用确定性计算或有限差分。'])
    (ROOT / '核验记录.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def main():
    gaussian_kl(); reparameterization(); gumbel_max(); rope(); online_attention()
    gan(); diffusion(); flow(); normalization(); policy_gradient(); transport()
    # Write generated outputs before validating links to those outputs.
    write_report()
    files_and_coverage()
    write_report()
    failed = [x for x in RESULTS if not x['passed']]
    if failed:
        raise SystemExit('Failed checks: ' + ', '.join(x['name'] for x in failed))


if __name__ == '__main__':
    main()

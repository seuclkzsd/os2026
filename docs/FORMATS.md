# 文件格式（生成器 / 评测 / CLI 之间的契约）

参考实现：`tools/case.py`（Python 侧唯一读写入口，其它脚本不得自行解析这些文件）。
C++ 侧：`src/io`（读矩阵）、`src/driver` + `apps/`（写结果与报告）。

## 1. 测例与参考解

```
data/cases/<name>/A.npz          scipy.sparse.save_npz，CSR，完整厄密存储（两个三角都在）
data/cases/<name>/B.npz          可选（广义问题）
data/cases/<name>/meta.json      来源、参数、期望（不含任何参考值）
data/cases/<name>/config.json    默认求解配置（§2），即交付物 d 的「默认配置样例」
data/refs/<name>/eigenvalues.npy 参考特征值，升序（复数时按实部）
data/refs/<name>/eigenvectors.npy 可选，n×m，Fortran 序，列与 eigenvalues 对应
data/refs/<name>/ref.json        {"kind","note","count","complete","window"}
```

- 参考解与输入**物理隔离**：CLI 只接收 `--A --B --config --out`（及 §2 的覆盖参数），从不获得
  `data/refs` 路径（赛题红线「禁止读取参考结果」）。
- `meta.json` 字段：`name, family, problem("HEP"|"GHEP"|"GHIEP"), n, nnz_A, nnz_B?, dtype,
  description, params(可复现生成的全部参数), inertia?({"A":[neg,zero,pos],"B":[...],"pencil":[...]}),
  targets?([{"sigma","k","why"}]), scenarios?(["zero-cluster","multiplicity","near-singular",
  "indefinite-B","complex-hermitian","memory","clustered","large"]), license?, citation?,
  schema_version`。
- `ref.json`：`kind ∈ {analytic, discrete-exact, dense, solver, partial-analytic}`；
  `complete=true` 表示 `window`（为 null 时指全谱）内的特征值全部列出，评测据此判定漏根。
  评测在内存中另有 `kind = "pooled"`（§4），从不写入 `data/refs`。

## 2. 求解配置 `config.json`

一个 JSON 对象，键名与 `SolverConfig` / `Target` 字段名**逐字一致**，枚举用 snake_case 字符串；
未知键报错。省略的键取 `SolverConfig` 默认值（下例除 `target` 外即全部默认值，键序同
`config_to_json`）。

```json
{
  "target": {"mode": "nearest", "sigma": -2.5e5, "k": 10},
  "problem": "auto",
  "method": "auto",
  "tol": 1e-10,
  "inner_tol_factor": 0.1,
  "hermitian_tol": 1e-10,
  "ncv": 0,
  "block_size": 0,
  "slice_size": 0,
  "max_restarts": 1000,
  "seed": 2026,
  "threads": 0,
  "memory_budget_gb": "auto",
  "factor_precision": "double",
  "ordering": -1,
  "deterministic": false,
  "blr": "auto",
  "out_of_core": "auto",
  "inertia_check": "auto",
  "inertia_counter": "auto",
  "zero_filter_tol": -1.0,
  "complex_tol": 1e-8,
  "max_shift_retries": 5
}
```

枚举取值：`target.mode ∈ {nearest, interval, smallest, largest}`（interval 用 `lower`/`upper`）；
`problem ∈ {auto, standard, generalized_definite, generalized_indefinite}`；
`method ∈ {auto, shift_invert_lanczos, shift_invert_krylov_schur, lanczos}`；
`factor_precision ∈ {double, single, auto}`；`blr / out_of_core / inertia_check / inertia_counter ∈ {auto, on, off}`。

取值与语义（校验见 `parse_config`；算法见 `docs/DESIGN.md` §3.6）：
- `tol ∈ (0, 1)`；`inner_tol_factor ∈ (0, 1]`；`ncv, block_size, slice_size, threads, max_shift_retries ≥ 0`；
  `max_restarts ≥ 1`；`memory_budget_gb > 0`（GiB）或 `"auto"` / `0`；interval 需 `lower < upper`，其它模式
  `k ≥ 1`。
- `memory_budget_gb`（P7）：`"auto"`（默认；`0` 同义，`config_to_json` 写作 `"auto"`）= 由进程的最小
  内存上限 L 推出：max(0.5, min(0.85·L, L − 1)) GiB；L 取内存 cgroup（v2 `memory.max`、v1
  `memory.limit_in_bytes` / `hierarchical_memory_limit`）、环境变量 `HSOLVE_MEMORY_LIMIT_GB`（GiB，调用方
  声明；`tools/run/guard.py` 导出其 RSS 上限）、有限的 RLIMIT_AS / RLIMIT_DATA、MemTotal 中最小者，都读不到
  时 14。数值（GiB = 2^30 字节）= 就用它（P7 前的配置照旧有效）；超过 L 时告警。预算决定因子内存内 /
  外存 / 拒绝（OutOfMemoryBudget，未分配），以及 Krylov 块宽与切片大小（DESIGN §3.6-3）；报告
  `memory` 记录结果与来源（§3）。
- `factor_precision`（P7）：`double`（默认）；`single` = 单精度因子（因子内存 0.51–0.67 倍）、每次算子
  apply 双精度修正，更慢；运行受检查（修正须收敛、单精度因子关检测后不得奇异、结果须被接受），否则以双精度
  同一主元顺序重解，不会返回错误结果；`auto` = 双精度因子在预算内放不进内存而单精度放得进时才用单精度
  （同样受检查与回退）。报告 `precision` 记录请求、实际与回退（§3）；BLR（`blr = on`）同样受检查与回退。
- `block_size`（P4）：整数 ≥ 0，位移反演 Krylov 运行的块宽 p；0（默认）= 自动，1 = 单向量。
  HEP/GHEP：自动时需求 ≥ 8 对的运行（含切片、单边运行）p = 4，否则 1；配置 > 0 时直接使用；证书收缩
  重跑缺 m 对时用 max(p, min(4, m 上取偶))。GHIEP：自动时 n ≥ 2e4 且 nev ≥ 4 的运行 p = 2，否则 1；
  配置值被限到 min(p, nev, 64, n/16)，探测运行（nev ≤ 2）恒为 1。两条路径都会在块基放不下内存预算时
  把 p 减半（告警）。规则与依据见 DESIGN §3.6-5。
- `inertia_counter`（P4）：证书的 Sylvester 计数是否在 InertiaCounter（只计惯性的分解，复用主分析的
  主元顺序）上做。auto（默认）= nearest 证书在代价模型判定划算时、以及主分解为单精度/BLR 时一律；
  on = 总是（含窗口/切片的计数）；off = 在主分解对象上计数（主分解为单精度/BLR 时仍用计数器——其自身
  惯性不可信）。谱隙点上的计数相同（测试与实测一致），区别在速度与峰值内存（DESIGN §3.6-8）。
- `slice_size`：整数 ≥ 0，默认 0 = 自动（代价模型）。interval / smallest / largest 目标（B 正定或
  B = I）以及零带干扰时的窗口化 nearest：每个位移至多求这么多个特征值（谱切片）；内存预算可能把它
  再压低（报告中有告警）。
- `ordering`：−1（默认）= 自适应：AMD 或嵌套剖分（由确定性代价模型选择；嵌套剖分为 SCOTCH，
  `deterministic = true` 时为 METIS）；0..7 = MUMPS ICNTL(7)（7 自动、5 METIS、3 SCOTCH、0 AMD、
  4 PORD、6 QAMD、2 AMF）。hsolve 自研的并行嵌套剖分（factor 层的 10）目前**不**是合法配置值。
- `deterministic`：false（默认）= MUMPS L0-OpenMP，回代更快，结果在容差内可复现；true = 同线程数下
  分解与回代逐位可复现，自适应排序随之用 METIS（SCOTCH 的排序逐次运行都会变，显式 `ordering = 3`
  亦然，逐位复现时不要用）。
- `zero_filter_tol`：< 0（默认）关闭；否则 |λ| ≤ zero_filter_tol·scale 为非物理零模，靠位移放置排除
  （HEP/GHEP 由惯性在两个计数点间切掉零带，计入报告 `zero_modes_filtered`；GHIEP 零簇本来就不被
  需要，零带取 max(zero_filter_tol, 1e-8)·scale）。用于零附近密集谱 / 低频击穿场景。
- `inertia_check`：off 关闭 nearest 的证书；interval / smallest / largest 本身由惯性计数构成，off 被
  忽略（告警）。
- 当前限制：B 不定（GHIEP）只支持 `mode = nearest`（其余抛 Unsupported）；`method = lanczos`
  （免分解）抛 Unsupported。

**CLI 覆盖参数**（`hsolve --A A.npz [--B B.npz] [--config config.json] --out DIR [...]`，覆盖配置文件
中的对应键，合并后整体再校验）：`--sigma S`（nearest）、`--k K`、`--interval LO HI`、
`--smallest K`、`--largest K`、`--tol`、`--threads`（0 = 全部）、`--memory-budget-gb GiB|auto`、
`--precision double|single|auto`、`--slice-size N`（→ `slice_size`）、`--block-size N`（→ `block_size`，
0 = 自动，1 = 单向量）、`--zero-filter-tol TOL`
（→ `zero_filter_tol`，< 0 关闭）；另有 `--log FILE`（JSON-lines 事件日志）、`-v/--verbose`、
`-q/--quiet`、`--version`。退出码：0 = 全部达标（且证书通过），1 = 完成但未达标，2 = 错误。

## 3. 求解结果目录（`hsolve --out DIR`，scipy 基线同格式）

| 文件 | 内容 |
|---|---|
| `eigenvalues.npy` | 形状 (m,)，float64，实特征值（HEP/GHEP 为全部），按目标排序（nearest：离 σ 由近到远；interval/smallest：升序；largest：降序） |
| `eigenvectors.npy` | 形状 (n, m)，Fortran 序；矩阵为复数时 complex128，否则 float64；HEP/GHEP 为 B-正交归一，GHIEP 为 2-范数归一 |
| `complex_eigenvalues.npy` | 仅 GHIEP 出现真复特征值时存在，形状 (c,)，complex128 |
| `complex_eigenvectors.npy` | 同上，形状 (n, c) |
| `report.json` | `SolveReport` 的全部字段（键名同 C++ 字段名），外加 `schema_version: 1` |

`report.json` 主要键（键序同 `SolveReport::to_json`）：`schema_version, solver, version, n, nnz_a,
nnz_b, complex_scalars, problem, norm2_a, norm2_b, scale, hermitian_defect_a, hermitian_defect_b, method,
sigma_requested, sigma_used, shift_retries, block_size, factor_options{...}, factor_stats{...}, restarts,
op_applies, time_total_s, time_setup_s, time_factor_s, time_iterate_s, time_verify_s,
time_certificate_s, peak_rss_bytes, memory{...}, precision{...}, pairs[{lambda_re, lambda_im, abs_residual, rel_residual,
backward_error, accepted}], certificate{performed, lower, upper, count_inertia, count_found,
endpoint_moves, passed, note}, slices[...], zero_cluster_dim, zero_modes_filtered, success, warnings[]`。
`pairs` 先列实特征对（与 `eigenvalues.npy` 同序），再列复特征对。

- `solver`（P4）：恒为 `"hsolve"`（评测在调用方未指名时用它作求解器名，§4）；`version`：`hs::version()`。
  Python 接口的 `report` 是同一份 JSON。
- `block_size`（P4）：HEP/GHEP 第一次位移反演运行实际使用的块宽（配置解析 + 内存适配之后，DESIGN
  §3.6-5）。GHIEP 的报告目前恒为 1（driver 未写入），各次运行的实际块宽见事件日志的 `krylov_schur`
  事件。
- `factor_stats` 只统计主分解对象（分析、分解、回代次数与时间、估计）；InertiaCounter 的分析与计数
  不在其中，只计入 `time_certificate_s`（事件日志：`inertia_counter`、`main_factors_released`）。
- **计时**（五个互斥阶段，`docs/DESIGN.md` §3.6-0）：
  `time_total_s == time_setup_s + time_factor_s + time_iterate_s + time_verify_s + time_certificate_s`，
  各项 ≥ 0。`time_setup_s`：输入检查、范数估计、问题类型判定、规划；`time_factor_s`：**每一次**符号
  分析（含自适应排序的 AMD 探测）与位移、B（正定性检测）、A（GHIEP 零主元计数）的数值分解；
  `time_iterate_s`：Krylov 求解（回代、正交化、重启）；`time_verify_s`：Rayleigh–Ritz、残差、补救、
  选择与证书簿记；`time_certificate_s`（P4）：HEP/GHEP 的全部 Sylvester 惯性计数（证书端点、窗口端点、
  切点、簇检查；主对象上的重分解或 InertiaCounter 的计数，含计数器的分析），GHIEP 与关闭证书的
  nearest 为 0。P3 的报告没有此键，惯性计数在 `time_factor_s` 中。时间只覆盖 `hs::solve`（CLI 读矩阵
  不计入）。
- `certificate.lower / upper`：±∞ 写作 `null`（smallest：lower = null；largest：upper = null）。
  `count_inertia` 是按 Sylvester 惯性**要求**的个数（smallest/largest 结束于数值相等的大簇内时为簇前
  个数加仍需的副本数）；`count_found` 是窗口内不同 B-正交向量数。GHIEP：`performed = false`，
  note 说明不可用。
- `slices`：切片 / 单边求解中每次位移反演 Krylov 运行一条（单位移 nearest 为空数组）：
  `{lower, upper (±∞ 为 null), sigma, which ∈ {largest_magnitude, smallest_algebraic,
  largest_algebraic}, count_inertia（GHIEP 为 −1）, count_found, nev, reruns, restarts, op_applies,
  passed}`。which 的含义：largest_magnitude = σ 两侧最近，smallest_algebraic = σ 以下，
  largest_algebraic = σ 以上。GHIEP 的两侧单边运行、复特征值恢复运行，以及判定某侧为空的探测运行
  （nev = 1）也列在这里。
- `zero_cluster_dim`：GHIEP 需要时计算的零簇维数（A 的零主元数），未计算为 −1。
- `zero_modes_filtered`：被 `zero_filter_tol` 排除的特征值个数（HEP/GHEP 窗口为惯性计数所得）。
- `memory`（P7）：`{budget_gb, source, limit_gb, limit_source, limits[{name, gib, detail}]}`——本次运行
  规划用的预算（GiB）；`source` = `config`（配置的数值）或自动预算所据上限的来源（`cgroup`、`env`、
  `rlimit_as`、`rlimit_data`、`mem_total`），都读不到时 `default`；`limit_gb` / `limit_source` 为探测到的
  最小上限（无则 −1 / ""，配置数值时也探测）；`limits` 按上述顺序列出每个来源（`gib` 为 null 表示该来源
  不设限，`detail` 说明读自何处或为何不设限）。因子是否走外存见 `factor_options.out_of_core`；内存内分解
  因内存失败后改走外存的次数见 `factor_stats.ooc_switches`（`factor_options.ooc_on_memory_error` 为真时）。
- `precision`（P7）：`{requested, used, blr, decision, fallback, fallback_reason, attempt_s}`——请求的
  `factor_precision`、返回结果所用因子的精度（`double` / `single`）与是否 BLR；`decision` 为 auto 的依据
  （各精度内存内估计对 cap），否则空；`fallback` 为真时非精确（单精度 / BLR）尝试被放弃并以双精度重解，
  `fallback_reason` 说明原因，`attempt_s` 为被放弃尝试的墙钟时间（已计入各阶段时间）。

## 4. 评测输出（`tools/eval`）

`python -m tools.eval.evaluate CASE RESULT [--guard guard.json]` 评测一个结果目录；
`python -m tools.eval.sweep RUN` 评测一次原始运行的全部结果并写汇总表（两者都须在
`tools/run/guard.py` 下运行）。每个（测例, 求解器）一份 JSON（默认 `RESULT/evaluation.json`；sweep 加 `--eval-dir EVAL` 时写到
`EVAL/<case>/<solver>/evaluation.json`），一切指标都由测例矩阵与返回向量重算，`report.json` 只用于
簿记。

评测文档键（`schema_version: 1`）：
- 标识：`evaluated`（false = 无法评测，见下）、`case`、`solver`（参数 > `report.json` 的 `solver` >
  结果目录名）、`inputs{case, result, reference}`、`problem`、`n`、`nnz_a`、`nnz_b`、
  `complex_scalars`、`config`、`target`、`tol`、`tolerances{match_rtol, cluster_rtol, tie_rtol,
  dup_tol}`（默认 1e-6、1e-6、10·tol、1e-4）。
- 簿记：`time_total_s`；`solve_time_s` 与 `solve_time_scope`（同口径的求解时间：报告同时有
  `time_setup_s` 与 `time_solve_s` 时取二者之和——SLEPc 基线：EPSSetUp（MUMPS 分析与分解）+ EPSSolve +
  EPSSolve 推迟到首次 EPSGetEigenpair 的特征向量计算（广义问题的纯化，P4 起计入 `time_solve_s`），其
  `time_total_s` 还含读测例、PETSc 组装与输出——否则取 `time_total_s`——hsolve、scipy 基线）；`peak_rss_bytes`（求解器自报，
  缺时取 SLEPc 基线的 `peak_rss_bytes_sum`）；`measured{wall_s, peak_rss_bytes, returncode, killed,
  cores, threads}`（guard 摘要，没有时为 null）；`solver_success`；`solver_warnings`。
- `norms{norm2_a, norm2_b, scale, norm_method{A, B}}`（n ≤ 4000 稠密精确，否则 ARPACK 下界，
  tol 1e-6、ncv 40）。
- `reference{kind, comparable, complete, window, count, has_eigenvectors}`：`kind` 取自 `ref.json`，
  或 `"pooled"`——没有可比参考解的测例由同一次运行中多个求解器已接受且互异的特征对合并而成的
  未认证参考（只在内存中，从不保存；不报对它的特征值误差）。连续模型参考（partial-analytic）
  `comparable = false`，不参与匹配与完备性。
- `pairs[]`：每对 `index, kind(real|complex), lambda_re, lambda_im, abs_residual, rel_residual,
  backward_error, accepted, ref_index, ref_lambda_re, ref_lambda_im, eig_abs_error, eig_rel_error,
  eig_scaled_error, duplicate, duplicate_of`；`duplicates[]`（`duplicate_of = null` 表示零或非有限
  向量）。
- `completeness{status, certified, expected, found, missing, boundary_deficit, boundary_candidates,
  extra, beyond_target, unmatched, spurious, distinct_returned}`：status ∈ complete / incomplete
  （对已认证参考）、consistent / inconsistent（对未认证参考：solver、pooled）、unknown（无参考）。
- `clusters[]`：参考特征值簇（`ref_indices, multiplicity, lambda_min, lambda_max, computed_indices,
  angles, max_angle`），有参考特征向量时给出子空间夹角。
- `format{order_ok, max_normalization_error, max_orthogonality_error, issues}`：§3 的格式检查
  （目标排序、B-/2-范数归一、HEP/GHEP 向量的 B-正交性）；只报告，不影响判定。
- `summary{n_pairs, n_real, n_complex, n_accepted, n_duplicates, max_abs_residual, max_rel_residual,
  max_backward_error, max_eig_abs_error, max_eig_rel_error, max_eig_scaled_error, max_subspace_angle,
  b_orthogonality, solver_claim}`；`solver_claim` ∈ agrees / overclaim（自称成功而判定失败）/
  underclaim / null。
- `verdict` ∈ pass / fail 与 `reasons[]`：pass ⇔ 每对 η ≤ tol、无重复或退化向量、完备性不是
  incomplete/inconsistent、无 spurious、返回了特征对（区间目标在参考确认区间为空时可为空）且运行
  未被 guard 杀死或被信号终止。无法评测的运行（无结果目录、文件畸形、测例无法读入）也生成
  `evaluated = false`、`verdict = fail` 的文档，计入判定统计。

**原始运行目录**（`tools/bench/compare.py` 与实验脚本写出，`tools/eval/sweep.py` 读入）：

```
RUN/<case>/<solver>/              结果目录（§3）
RUN/<case>/guard-<solver>.json    该求解器那次运行的 tools/run/guard.py --summary
RUN/<case>/guard.json             只有一个求解器目录的测例用这个名字
```

每个（测例, 求解器）都是汇总表的一行：只有 guard 摘要而无结果目录的运行（崩溃、被杀、超时）记为
fail。每个测例的矩阵与范数只读 / 估计一次；无可比参考且有两个以上求解器结果的测例用 pooled 参考
（`--no-pool` 关闭）。汇总表默认写到 `results/eval_<run>.md`（`tools/eval/report.py` 排版），前言给出
重新生成该表与评测 JSON 的命令（对同一原始运行、数据根与线程数是确定的；评测 JSON 不入库）。

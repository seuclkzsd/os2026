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

- 参考解与输入**物理隔离**：CLI 只接收 `--A --B --config --out`，从不获得 `data/refs` 路径
  （赛题红线「禁止读取参考结果」）。
- `meta.json` 字段：`name, family, problem("HEP"|"GHEP"|"GHIEP"), n, nnz_A, nnz_B?, dtype,
  description, params(可复现生成的全部参数), inertia?({"A":[neg,zero,pos],"B":[...],"pencil":[...]}),
  targets?([{"sigma","k","why"}]), scenarios?(["zero-cluster","multiplicity","near-singular",
  "indefinite-B","complex-hermitian","memory","clustered","large"]), license?, citation?,
  schema_version`。
- `ref.json`：`kind ∈ {analytic, discrete-exact, dense, solver, partial-analytic}`；
  `complete=true` 表示 `window`（为 null 时指全谱）内的特征值全部列出，评测据此判定漏根。

## 2. 求解配置 `config.json`

一个 JSON 对象，键名与 `SolverConfig` / `Target` 字段名**逐字一致**，枚举用 snake_case 字符串；
未知键报错。省略的键取 `SolverConfig` 默认值。

```json
{
  "target": {"mode": "nearest", "sigma": -2.5e5, "k": 10},
  "tol": 1e-10,
  "inner_tol_factor": 0.1,
  "problem": "auto",
  "method": "auto",
  "ncv": 0,
  "max_restarts": 1000,
  "seed": 2026,
  "threads": 0,
  "memory_budget_gb": 14.0,
  "factor_precision": "double",
  "ordering": 7,
  "blr": "auto",
  "out_of_core": "auto",
  "inertia_check": "auto",
  "zero_filter_tol": -1.0,
  "complex_tol": 1e-8,
  "hermitian_tol": 1e-10,
  "max_shift_retries": 5
}
```

枚举取值：`target.mode ∈ {nearest, interval, smallest, largest}`（interval 用 `lower`/`upper`）；
`problem ∈ {auto, standard, generalized_definite, generalized_indefinite}`；
`method ∈ {auto, shift_invert_lanczos, shift_invert_krylov_schur, lanczos}`；
`factor_precision ∈ {double, single}`；`blr / out_of_core / inertia_check ∈ {auto, on, off}`。

## 3. 求解结果目录（`hsolve --out DIR`，scipy 基线同格式）

| 文件 | 内容 |
|---|---|
| `eigenvalues.npy` | 形状 (m,)，float64，实特征值（HEP/GHEP 为全部），按目标排序（nearest：离 σ 由近到远；interval/smallest：升序；largest：降序） |
| `eigenvectors.npy` | 形状 (n, m)，Fortran 序；矩阵为复数时 complex128，否则 float64；HEP/GHEP 为 B-正交归一，GHIEP 为 2-范数归一 |
| `complex_eigenvalues.npy` | 仅 GHIEP 出现真复特征值时存在，形状 (c,)，complex128 |
| `complex_eigenvectors.npy` | 同上，形状 (n, c) |
| `report.json` | `SolveReport` 的全部字段（键名同 C++ 字段名），外加 `schema_version: 1` |

`report.json` 主要键：`n, nnz_a, nnz_b, complex_scalars, problem, norm2_a, norm2_b, scale,
hermitian_defect_a, hermitian_defect_b, method, sigma_requested, sigma_used, shift_retries,
factor_options{...}, factor_stats{...}, restarts, op_applies, time_total_s, time_factor_s,
time_iterate_s, time_verify_s, peak_rss_bytes, pairs[{lambda_re, lambda_im, abs_residual,
rel_residual, backward_error, accepted}], certificate{performed, lower, upper, count_inertia,
count_found, endpoint_moves, passed, note}, success, warnings[]`。`pairs` 先列实特征对（与
`eigenvalues.npy` 同序），再列复特征对。

## 4. 评测输出（`tools/eval`）

每个（测例, 求解器）一份 JSON：输入路径、配置、时间、峰值内存、逐特征对指标（残差三件套、
与参考值的误差、匹配到的参考序号）、完备性（应得/实得/漏/多）、重特征值簇的子空间夹角；
`tools/eval/report.py` 汇总为 `results/*.md` 表格。

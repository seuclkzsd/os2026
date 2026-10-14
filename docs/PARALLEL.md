# hsolve 的并行方案：多线程与多进程（MPI）

赛题要求（f）与「并行方案」一节：支持多线程、MPI 与 GPU 加速，说明分工并给出**实测**加速比，并提供无 GPU 时可复现的
降级方案（评测机只有 CPU）。本页说明 hsolve 的线程层与多进程（MPI）模式：做什么、怎么构建和运行、评测机需要什么、
实测多少。设计契约见 `docs/DESIGN.md` §3.9，全部数据与命令见 `results/p10_mpi.md`。GPU 加速是另一条线（P10 gpu），
不在本页。

## 1. 分工

| 层 | 负责的计算 | 怎么并行 | 何时用 |
|---|---|---|---|
| 线程（OpenMP，进程内） | 一切数值计算：MUMPS 的分解与回代（L0 子树并行 + 多线程 BLAS，`docs/BLAS.md`）、稀疏核 SpMV / SpMM 与融合残差、正交化（固定分块的确定性归约）、自研并行嵌套剖分、BLAS hook 的并行切分 | 一个进程的全部核共享内存 | 总是（默认）：`hsolve --threads T`，0 = `OMP_NUM_THREADS` / 进程 cpuset 内的全部核 |
| 进程（MPI，P10） | 彼此独立的大块分解：nearest 证书的两次 Sylvester 计数（同时），窗口目标（区间 / 最小 / 最大 k 个）的谱切片（每片一次分解 + Krylov） | 每个 socket（NUMA 节点）一个 helper 进程，各用本 socket 的核与内存带宽；leader 跨全部核做顺序部分，二者从不同时计算 | 多 socket 机器上可选：`hsolve-mpirun`；单 socket 机器上不用（实测更慢） |

**降级（评测机只有 CPU、或没有 MPI）**：普通的单进程 `hsolve` 就是默认路径，不需要 MPI 也不需要 GPU；多进程模式的
结果与它相同（下文 §4），只是在多 socket 机器上更快。

## 2. 为什么是「每个 socket 一个进程」

数值分解与计数受内存带宽限制（P9：同一 socket 上 4 → 16 线程只快 1.84 倍）。所以同一 socket 上的多个进程只会互相
抢带宽——P9 的 MPI MUMPS（2×8 … 16×1）在一个 socket 上只有单进程的 0.85–0.59 倍。两个 socket 则不同（2×Xeon Silver
4314，每次运行 16 核，`results/p10_mpi.md` §2）：

- 每个 socket 上一个进程、各 8 线程**同时**做一次计数，彼此不干扰：每个只用单独做时的时间（1e6：27.1 s 对 27.0 s）；
  而同一 socket 上的两个并发进程各慢 1.4–1.8 倍。证书的两次计数因此从顺序的 35.2 / 29.7 s（一个进程 16 线程跨两个
  socket）降到 27.1 / 22.5 s（1e6 两例），1e5 降到 0.49–0.54 s（顺序 1.1–1.2 s）。
- 一个进程的 16 线程跨两个 socket 在 1e6 上比单 socket 快 1.2–1.3 倍（用到两份带宽），所以顺序部分（主分解、Krylov）
  留在一个占全部核的 leader 上；一个 socket 的 8 核做 1e6 的主分解要慢 1.6 倍。
- MPI 版 MUMPS 2×8 跨 socket 做一次分解：1e5 快 1.16–1.39 倍，1e6 只有 0.95 / 1.09 倍，且需要分布式的主分解对象——
  未采用（`results/p10_mpi.md` §2.2）。

## 3. 构建

```bash
# CMake 选项 HSOLVE_WITH_MPI：AUTO（默认，找到 MPI 就构建）、ON（找不到即报错）、OFF
tools/dev/build.sh release          # 集群节点：自动用 $HS_REF_ENV（os2026-ref 环境）的 Open MPI 5.0.11
cmake --preset release -DHSOLVE_WITH_MPI=ON -DMPI_HOME=/path/to/openmpi    # 其它 MPI 安装
```

产物：`build/<preset>/hsolve-mpi`（与 `hsolve` 同一份命令行代码 `apps/cli.cpp`）、`build/<preset>/hsolve-mpirun`
（启动脚本）与测试 `test_mpi_session`（ctest 以 1、2、3 个 rank 运行）。每个进程仍链接**顺序版** MUMPS 与 OpenMP，MPI
只传会话消息。注意：顺序 MUMPS 自带的 libmpiseq 以桩函数定义了 `MPI_Init` 等，CMake 把真正的 MPI 库排在 MUMPS 之前；
`hsolve-mpi` 启动时检查 `MPI_Init` 解析到的库，若是桩则报错退出。

依赖：Open MPI 5（conda-forge `openmpi` 5.0.11，BSD-3-Clause）及其运行库（libevent、hwloc、PMIx、PRRTE、UCX、
libfabric 1.22）。集群节点上它们在 os2026-ref 环境里，由 `tools/dev/deploy_cluster.sh` 从本机 conda 包缓存**离线**
重建（`docs/DEPENDENCIES.md`）；任何提供 `mpicc`（供 CMake FindMPI）与 `mpiexec` 的 MPI 也可以。

## 4. 运行

```bash
build/release/hsolve-mpirun [--ranks R] [--dry-run] -- --A A.npz [--B B.npz] --config config.json --out DIR [hsolve 选项]
```

- `hsolve-mpirun` 读本进程的 cpuset 与 NUMA 拓扑：rank 0（leader）绑定全部允许的核、`OMP_NUM_THREADS` = 其个数；
  每个有允许核的 NUMA 节点一个 helper，绑定该节点的核。默认 R = 1 + socket 数；只有一个 socket 时 R = 1（普通
  hsolve）。`--ranks R` 强制 R − 1 个 helper（NUMA 节点不够时把核等分）。`--dry-run` 只打印布局与 mpiexec 命令。
  mpiexec 取 `--mpiexec`、`$MPIEXEC`、PATH 或 `$HS_REF_ENV/bin/mpiexec`。
- 也可以直接 `mpiexec -n R … hsolve-mpi <同样的参数>`（每个 rank 的参数相同；绑定与线程数由调用者负责）。只有一个
  helper 时它什么也不做（并发计数与切片都要两个 helper，否则与 leader 抢核），所以 R = 2 等于普通 hsolve。
- 选项（只对多进程有效）：`--helper-counts auto|on|off`、`--helper-slices auto|on|off`（`hs::ParallelOptions`）。auto
  按代价规则：计数——本次实测的主分解时间下一次计数长于 helper 的准备（约两次给定顺序的分析 + 0.05 s）；切片——一片的
  预测工作长于 helper 的准备。
- 输出与 `hsolve` 相同（`docs/FORMATS.md` §3），报告另有 `parallel`：processes、threads、helper_counts、helper_slices、
  fallbacks、share_s、wait_s、helper_busy_s、notes（为何没交给 helper、失败与回退）。`--log FILE` 时 helper 的事件写到
  `FILE.rank<r>`。退出码同 hsolve（rank 0 的）。
- 容错：helper 的任务失败（内存不足、MUMPS 错误、任何异常）时 leader 自己做，结果照常认证；helper 进程崩溃则整个
  MPI 作业结束（MPI 的默认错误处理）。

## 5. 实测加速

测量节点 c005（2×Xeon Silver 4314，每 socket 16 核，无 GPU），每次运行 16 核、32 GiB，5 次交错、每次结果经
tools/eval 判定有效；时间为报告的 `time_total_s`（求解本身，不含读矩阵与 MPI 启动），括号内为对同一两 socket 核上
单进程 hsolve 的加速比。「单 socket」= 一个 socket 的 16 核；「两 socket」= 每个 socket 8 核（共 16 核）。

| 布局（16 核） | 位置 | 1e6 nearest：syn-mem27-100 / yee-pec-box-1e6 | 1e5 三维 nearest：syn-mem27-46 / yee-pec-box-1e5 / ss-qa8fk-qa8fm | 多片窗口：syn-mem27-46 [0.1, 0.3] / yee-pec-cube-32 [40, 400] |
|---|---|---|---|---|
| hsolve 1×16 | 单 socket | 87.4 / 72.2 s | 2.02 / 2.45 / 1.50 s | 31.2 / 13.9 s |
| hsolve-mpi 1 + 2×8 | 单 socket | 93.9 / 77.7 s（0.93×） | 2.27 / 2.64 / 1.68 s（0.89–0.93×） | 26.7 / 12.5 s（1.17 / 1.11×） |
| hsolve 1×16 | 两 socket | 75.4 / 64.1 s | 2.26 / 2.78 / 1.74 s | 32.9 / 15.3 s |
| **hsolve-mpi 1 + 2×8** | **两 socket** | **68.3 / 58.5 s（1.10 / 1.10×）** | **1.99 / 2.54 / 1.64 s（1.14 / 1.10 / 1.06×）** | **18.8 / 10.2 s（1.75 / 1.51×）** |

各阶段（两 socket，hsolve → hsolve-mpi）：证书 35.9 → 29.2 s、29.7 → 24.2 s（1e6），0.97–1.01 → 0.72–0.76 s（1e5）；
窗口的切片 25.8 → 15.7 s、11.8 → 8.3 s；分析、主分解、Krylov 不变。单独的分解阶段（`results/p10_mpi.md` §2、§5）：
证书的两次计数每 socket 一个同时做 27.1 / 22.5 s，对一个进程跨两个 socket 依次做 35.2 / 29.7 s、单 socket 43.2 /
35.7 s；MPI 版 MUMPS 2×8 跨 socket 做一次分解 19.0 / 14.6 s，对顺序 MUMPS 跨 socket 18.1 / 16.0 s（未采用）。

## 6. 限制

- **要两个 socket。** 单 socket 上 helper 与 leader 共用一份内存带宽，多进程模式更慢（0.86–0.95 倍，多片窗口除外），
  `hsolve-mpirun` 在单 NUMA 节点上因此直接运行普通 hsolve。评测机的 16 核若是一个 socket，多进程模式不改变结果
  也不加速。
- **nearest 只并行证书的两次计数。** 主分解与 Krylov 是一条顺序计算，仍由占全部核的 leader 做；1e6 上证书约占运行的
  40–45%，端到端 1.10 倍。GHIEP（没有惯性计数）、randomized 证书、只有一片的窗口与太小的问题（代价规则）没有可分的工作。
- **MPI 启动约 1.8–2.0 s 墙钟**（prterun 与三个进程的 MPI_Init），不计入 `time_total_s`；它超过 1e5 nearest 与
  108 个特征值窗口的收益，1e6 与多片窗口仍净赚。
- **内存。** 每个 helper 持有问题的副本与自己的计数器或因子：1e6 上进程树峰值 18.0 / 14.9 GiB，对单进程 13.1 /
  11.3 GiB。预算覆盖整棵进程树，内存紧时计数留在 leader。
- **拓扑是模拟的。** 两 socket 的布局用每个 socket 16 核中的 8 核：每个 socket 以整份带宽与 L3 服务 8 核，与
  2×8 核的机器相当，但更小的 socket 每 socket 的带宽可能更低。
- helper 进程崩溃会结束整个 MPI 作业（MPI 默认的错误处理）；helper 任务内的失败由 leader 接手。

# PARDISO Phase 0 运行时可行性记录

## 范围

- 记录时间：2026-08-10T02:44:33+08:00
- 仓库提交：`dc4c3a24908921d471713b6004eef9553dd2f44b`
- Phase 0 只涉及依赖声明、锁文件、characterization tests 和本记录。
- 未修改 `src/fem/solvers/static_linear.py` 或其他生产求解代码。
- 未运行 `data/BiomimeticFourFingerHand_C3D20.inp`，未删除 `splu`，未引入替代求解库。

## 环境快照

| 项目 | 记录值 |
| --- | --- |
| 操作系统 | Windows 11 25H2，build 26200.8875，AMD64 |
| Python | CPython 3.13.11，MSC v.1944，64 bit |
| CPU | Intel Core Ultra 5 125H，18 个逻辑处理器 |
| 物理内存 | 33,945,935,872 bytes（约 31.61 GiB） |
| 记录时可用物理内存 | 16,740,954,112 bytes（约 15.59 GiB） |
| Commit limit | 42,779,799,552 bytes（约 39.84 GiB） |
| 记录时 committed bytes | 35,612,418,048 bytes（约 33.17 GiB） |
| NumPy | 2.5.2 |
| SciPy | 1.18.0 |
| py-mkl-pardiso | 0.0.5 |

Windows 注册表仍把产品名报告为 Windows 10 Home China；Python 平台信息、25H2 显示版本和 build 26200 对应当前 Windows 11 环境。NumPy 与 SciPy 的实际锁定版本高于规划时快照，同时仍满足项目声明的 `numpy>=2.3,<3` 和 `scipy>=1.16,<2`。

记录时 committed bytes 约占 commit limit 的 83.25%。这不影响小矩阵 smoke；Phase 4 代表模型验收前仍需按计划关闭无关高内存程序。

## 依赖与 wheel

- 核心依赖约束：`py-mkl-pardiso>=0.0.5,<0.1`。
- `uv.lock` 解析版本：0.0.5。
- 已安装模块：`pymklpardiso`。
- 已验证导出：`PardisoSolver` 和 `MTYPE_REAL_SYM_POSDEF`。
- 已验证 SPD 矩阵类型常量值：2。
- wheel：`py_mkl_pardiso-0.0.5-cp313-cp313-win_amd64.whl`。
- wheel tag：`cp313-cp313-win_amd64`。
- wheel SHA-256：`db123636c5f42882ef0ad6792917f6b22af12ad8e6d35a4b3c8c453655532623`。
- 安装使用预编译 Windows x86-64 wheel，没有本地 C++ 编译步骤。

该 wheel 将 oneMKL native runtime 打包在扩展模块中。0.0.5 的 Python API、wheel 元数据和 PARDISO 诊断输出均未暴露精确的 oneMKL build 版本，因此本阶段可核实的 native 后端版本标识为 py-mkl-pardiso 0.0.5。此限制应保留到最终 handoff；若 Phase 4 需要精确 oneMKL build，需由上游 wheel 增加版本 API 或发布构建清单。

## 默认线程信息

记录时以下进程环境变量均未设置：

- `MKL_NUM_THREADS`
- `MKL_DYNAMIC`
- `OMP_NUM_THREADS`
- `OMP_DYNAMIC`
- `KMP_AFFINITY`
- `KMP_HW_SUBSET`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`

真实 3×3 smoke 的 PARDISO 诊断报告 `Parallel Direct Factorization is running on 1 OpenMP`，`get_iparm_value(2)` 同样返回 1。该数值是小矩阵实际运行信息，不代表 Phase 4 大模型会使用相同线程数；大模型验收必须重新记录线程设置与 PARDISO 实际诊断。

## 真实 3×3 PARDISO smoke

输入为实对称正定矩阵：

| | 第 1 列 | 第 2 列 | 第 3 列 |
| --- | ---: | ---: | ---: |
| 第 1 行 | 4 | 1 | 0 |
| 第 2 行 | 1 | 3 | 1 |
| 第 3 行 | 0 | 1 | 2 |

右端项为 1、2、3。传给 `PardisoSolver` 的对象是只含上三角、排序后的 `float64` CSR，共 5 个非零项，矩阵类型为 `MTYPE_REAL_SYM_POSDEF`。

| 验证项 | 结果 | 门槛 |
| --- | ---: | ---: |
| 解向量 | 0.2222222222222222、0.1111111111111111、1.4444444444444444 | 记录值 |
| 相对残差 | 1.1868783374443499e-16 | 不超过 1e-12 |
| 与 `numpy.linalg.solve` 的最大绝对差 | 4.4408920985006262e-16 | `rtol=1e-12, atol=1e-12` |
| NumPy oracle `allclose` | 通过 | 必须通过 |
| native `release()` | 完成 | 必须完成 |

## Characterization tests

改动生产后端前已冻结或已有明确覆盖的合同如下：

| 合同 | 测试覆盖 |
| --- | --- |
| `static_linear.__all__` | `test_static_linear_solver_public_surface` |
| `prepare`、`solve`、`PreparedSystem` 公共调用形状 | `test_static_linear_solver_public_call_shapes` |
| 小型解析模型的位移、反力与结果名称 | `test_static_linear_solver_builds_step_boundary_and_solves_case`、`test_static_linear_solver_returns_scalar_result_with_name_and_reactions` |
| 多步返回形状与选择语义 | `test_plural_*`、`test_scalar_*` 系列 |
| 稠密消元 oracle | `test_reduced_solver_matches_full_dirichlet_oracle_for_multiple_cases` |
| 同约束模式的因子复用，载荷和规定值不进入缓存键 | `test_factor_key_ignores_load_and_prescribed_values` |
| 完全约束不创建因子 | `test_factor_cache_handles_fully_constrained_and_unconstrained_systems` |
| 并发请求只创建一次因子 | `test_factor_cache_serializes_concurrent_factor_creation` |
| `PreparedSystem.clone()` 共享基刚度与因子缓存，同时隔离公开模型 | `tests/performance/test_solver_efficiency_contracts.py::test_prepared_system_reuses_work_without_sharing_public_models` |
| 欠约束预检与直接求解的错误摘要、异常链 | `test_static_linear_stiffness_preflight_detects_free_rigid_dofs`、`test_static_linear_solve_preserves_singular_error_summary_and_cause` |
| 顶层计时键 | `test_prevalidated_solve_skips_duplicate_validation_and_records_stages` |

聚焦测试结果：

- `tests/test_solvers.py --collect-only -q`：收集 62 项。
- `tests/test_solvers.py -q`：62 passed in 0.66s。
- `tests/performance/test_solver_efficiency_contracts.py::test_prepared_system_reuses_work_without_sharing_public_models -q`：1 passed in 0.99s。

Phase 0 未运行全量测试。

## 工作区重叠审计

计划指定的后续重叠关注文件如下：

- `src/fem_gui/main_window.py`
- `src/fem_gui/analysis_dialogs.py`
- `tests/gui/test_analysis_jobs.py`

Phase 0 检查时这三个路径相对当前提交均无未提交差异。后续 Phase 在编辑 GUI 测试监测点前仍需重新执行 path-scoped diff，避免覆盖用户随后产生的改动。

Phase 0 检查时可见的未提交路径仅为 `pyproject.toml`、`uv.lock` 和 `tests/test_solvers.py`；本记录位于被 `.gitignore` 忽略的 `docs/`，需由主 session 明确决定是否只跟踪本运行时记录，且不得为提交本记录而强制跟踪计划文档。

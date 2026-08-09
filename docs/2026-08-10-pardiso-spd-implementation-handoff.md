# PARDISO SPD 求解器实现交接记录

## 状态

- 日期：2026-08-10
- 分支：`develop/pardiso-spd-cholesky`
- 验收代码基线：`7e1ec42bde2c5784914c169597588264eab17d19`
- Phase 0–3：实现提交存在，聚焦回归通过
- Phase 4：代表模型、资源、复用与 Abaqus 数值验收通过
- 最终 Definition of Done：按用户指定的聚焦验证范围完成

Phase 4 代表模型为用户指定的 `data/complicate_cae_model.inp`，同名 ODB 是本次 Abaqus 基准。本记录只使用与本次求解器替换有关的聚焦测试，不运行全仓或 CAD/Gmsh 测试。

## 实现提交

| Phase | 提交 | 主题 |
| --- | --- | --- |
| Phase 0 | `52de64fd92327e98fdd4cb2aaf11b220f92e4453` | `chore(solver): complete PARDISO phase 0 feasibility` |
| Phase 1 | `ead91c4a260a94c88b4616f1eaaa7331cd7d12d8` | `feat(solver): add PARDISO SPD adapter` |
| Phase 2 | `bdcd6d7dbc2dad719222c0553a80d4a1e891ec19` | `feat(solver): replace SuperLU with PARDISO SPD` |
| Phase 3 | `7e1ec42bde2c5784914c169597588264eab17d19` | `fix(solver): bound PARDISO factor lifecycle` |
| Phase 4 | 本次独立提交 | `test(solver): complete PARDISO phase 4 acceptance` |

Phase 4 只提交本交接记录；代表模型、ODB、临时验收脚本和导出数据均不进入提交。本文档位于被忽略的 `docs/`，提交时仅对本文件执行强制暂存，不包含同目录中的实施计划或其他文件。

## 环境与运行时

| 项目 | 已核实值 |
| --- | --- |
| 操作系统 | Windows 11，build 26200，AMD64 |
| CPU | Intel Core Ultra 5 125H |
| 逻辑处理器 | 18 |
| 物理内存 | 33,945,935,872 bytes，31.615 GiB |
| Python | CPython 3.13.11，MSC v.1944，64 bit |
| NumPy | 2.5.2 |
| SciPy | 1.18.0 |
| pytest | 9.1.1 |
| py-mkl-pardiso | 0.0.5 |
| wheel | `py_mkl_pardiso-0.0.5-cp313-cp313-win_amd64.whl` |
| wheel SHA-256 | `db123636c5f42882ef0ad6792917f6b22af12ad8e6d35a4b3c8c453655532623` |
| PARDISO 矩阵类型 | `MTYPE_REAL_SYM_POSDEF = 2` |

当前进程环境中的 `MKL_NUM_THREADS`、`MKL_DYNAMIC`、`OMP_NUM_THREADS`、`OMP_DYNAMIC`、`KMP_AFFINITY`、`KMP_HW_SUBSET`、`OPENBLAS_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS` 均未设置。代表模型真实因子化后的 PARDISO `iparm[2]` 记录为 1 个 OpenMP 线程。

py-mkl-pardiso 0.0.5 的 Python API、wheel 元数据和诊断输出没有暴露精确 oneMKL build 版本。当前能够核实的 native 后端标识仅为 py-mkl-pardiso 0.0.5。若交付要求 oneMKL 精确 build，需要上游 wheel 提供版本 API 或构建清单。

## 已验证项

### 聚焦回归

按 adapter、solver、performance、application 和 GUI 作业边界逐组运行，未运行全仓 pytest。

| 范围 | 结果 |
| --- | --- |
| `tests/solvers/test_pardiso_spd.py` | 33 passed in 0.65s |
| `tests/test_solvers.py` | 65 passed in 0.82s |
| `tests/performance/test_solver_efficiency_contracts.py` | 4 passed in 1.06s |
| `tests/application/test_analysis_runs.py` | 14 passed in 1.11s |
| `tests/gui/test_analysis_jobs.py` | 22 passed in 4.75s |
| `tests/test_abaqus.py::test_abaqus_read_builds_and_solves_full_c3d20_model` | 1 passed in 1.59s |
| 合计 | 139 passed，0 failed，0 skipped |

adapter 专项包含真实 PARDISO 的向量/多 RHS 稠密 oracle 和不定矩阵因子化失败测试。其余专项覆盖公共调用形状、解析模型、稠密消元 oracle、内联完整 C3D20 公共 INP 读取与真实求解、单项缓存、复用、淘汰释放、并发串行化、完全约束、Session stale/failed/cancelled/succeeded 事务语义及 GUI 作业线程和计时合同。C3D20 用例由测试内联文本写入 pytest 临时目录，不读取、复制或派生 `data/` 代表模型。

相关文件的全局 Ruff 聚焦检查结果为 `All checks passed!`。`git diff --check main..HEAD` 与工作区 `git diff --check` 均通过。

### 源码、diff 与数据依赖审计

- `rg -n "splu|SuperLU" src tests` 无匹配；生产和当前测试没有 SuperLU 依赖或 monkeypatch。
- PARDISO 第三方类型只出现在私有适配器及其专项测试；`static_linear.__all__` 仍为 `PreparedSystem`、`prepare`、`solve`。
- `pyproject.toml` 声明 `py-mkl-pardiso>=0.0.5,<0.1`。
- `_FACTOR_CACHE_MAX_ENTRIES` 为 1。
- `main..HEAD` 没有修改 `README.md`、`README_Zh.md`、`src/fem_gui/main_window.py` 或 `src/fem_gui/analysis_dialogs.py`。
- 验收时 tracked 工作区无未提交差异；用户 GUI 生产文件没有被 Phase 0–3 提交覆盖。
- 测试中没有 `complicate_cae_model` 或仓库 `data/` 代表模型路径依赖。泛化 `data` 搜索命中的内容是无关文本和 `examples/examples_data/` 布局规则。
- 未发现进入版本控制的代表模型性能日志、大结果文件或临时 Abaqus 导出。

## 代表模型与输入身份：verified

| 项目 | 值 |
| --- | --- |
| INP | `data/complicate_cae_model.inp` |
| INP 大小 | 6,418,655 bytes |
| INP SHA-256 | `ED34394D0EAA3B5EBB577E5CCDBEAF6606A3BCDC88DB31AFF9361B691DEC58FE` |
| ODB | `data/complicate_cae_model.odb` |
| ODB 大小 | 20,464,376 bytes |
| ODB SHA-256 | `C50A27A2374F9D380557C7D8A622805A823CF0A89BA0B44D9F825A4B37643A7C` |
| 节点 | 31,167 |
| 单元 | 133,323 个 C3D4，对应项目 `Tet4` |
| 总自由度 | 93,501 |
| 约束后自由自由度 | 91,822 |
| 约束自由度 | 1,679 |
| 分析步 | `Step-1`，linear static，`nlgeom=NO` |
| 材料 | `Generic_Aluminum_Engine_Part`，E=70,000，ν=0.33 |

约束和载荷为：

- `Set_Fixed_BC_1_Faces`：403 个节点，U1/U2/U3 固定；
- `Set_Displacement_BC_2_Face`：470 个节点，U1=10；
- `Set_ConcentratedForce_Load_3_Point`：1 个节点，F2=-1000；
- `Surface_Pressure_Load_1_Face`：789 个三角面，压力 1000；
- `Surface_Pressure_Load_2_Face`：322 个三角面，压力 1000。

全新进程的 quick preflight 和逐单元 full structural preflight 均通过。quick preflight 用时 0.572 s，full structural preflight 用时 7.606 s；二者均无 error。输出请求的 Abaqus `PRESELECT` field/history 不能由项目直接执行，因此产生预期 warning；大型模型 quick 模式另有 sampled capability 和 stiffness-skipped warning。数值稳定性随后由真实 PARDISO 因子化、生产残差检查和 Abaqus 对比验证。

## 代表模型性能与资源 gate：verified

验收在全新 Python 进程中按正常路径执行：INP `read_with_report`、`ModelSession` 导入/验证/作业生命周期、`static_linear.prepare/solve`、`build_solve_result_bundle`。第二工况在内存中把原压力和集中力缩放为 50%，保持相同的 1,679 个约束自由度；未修改 INP 或 ODB。

| 阶段 | 冷启动实测 |
| --- | ---: |
| INP 解析 | 17.440 s |
| Session 导入所有权转移 | 3.802 s |
| Session quick preflight | 0.585 s |
| 分析准备 | 7.479 s |
| 刚度矩阵装配 | 14.858 s |
| 第一工况线性方程求解 | 3.407 s |
| 第一工况完整 solve | 4.015 s |
| 第一工况输出请求与初始结果 bundle | 9.978 s |
| 第二工况完整 solve | 0.678 s |
| 第二工况线性方程求解 | 0.05648 s |
| 第二工况输出请求与初始结果 bundle | 10.837 s |
| 子进程内部总耗时 | 89.331 s |
| 外部监控 wall time | 91.280 s |

| 资源/复用指标 | 实测 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| 峰值 private memory | 2.931 GiB | ≤16 GiB | 通过 |
| 峰值 working set | 1.795 GiB | 记录项 | 通过 |
| 峰值 system committed | 30.015/37.773 GiB，79.46% | <90% | 通过 |
| 最低 available physical | 15.628 GiB | 无持续换页/失去响应 | 通过 |
| 冷启动总耗时 | 89.331 s | ≤15 min | 通过 |
| 首次线性阶段 | 3.407 s | ≤10 min | 通过 |
| 第二/首次线性阶段 | 1.658% | ≤20% | 通过 |
| 因子化调用数 | 第一工况后 1，第二工况后仍为 1 | 不重新因子化 | 通过 |
| PARDISO 实际线程 | `iparm[2]=1` | 记录项 | 已记录 |

所有位移与反力均有限；多次完整代表求解均通过生产 `_validate_free_dof_equilibrium`，即自由自由度残差满足 `residual <= 1e-8 × scale`，没有绕过或放宽项目检查。监控设置了 system commit 88%、available physical 1 GiB 和 15 分钟三个提前停止点，验收未触发任何安全停止，也没有关闭用户程序。

## Abaqus 数值 gate：verified

本机 `D:\Program Files (x86)\Abaqus2023\Commands\abaqus.bat` 可用。Abaqus Python 2.7.15 的 `odbAccess.openOdb(..., readOnly=True)` 成功读取同名 ODB；项目 `.venv` 本身不含 `odbAccess`。ODB 身份证据为 `Step-1`、instance `STEP_FIRSTPART-1`、31,167 个节点、133,323 个单元，与 INP 完全对应。

ODB 作业记录为：

- Abaqus/Standard 2023；
- `JOB_STATUS_COMPLETED_SUCCESSFULLY`；
- 0 analysis errors，0 analysis warnings；
- `numDomains=1`；
- 2 个 frame，比较最终 frame value 1.0。

比较前锁定近零特征尺度为：位移 1e-9、反力 1e-6、应力 1e-6，单位沿用 INP 的一致单位制。误差指标保持 max(|a-b|) / max(s, max(|b|))。

| 结果 | 归一化误差 | 门槛 | 结论 |
| --- | ---: | ---: | --- |
| 31,167 节点 U1/U2/U3 | 3.3682e-8 | ≤1e-6 | 通过 |
| 873 个受约束节点 RF1/RF2/RF3 | 3.0141e-8 | ≤1e-5 | 通过 |
| RF 合力 X | 6.7787e-9 | ≤1e-6 | 通过 |
| RF 合力 Y | 7.6097e-10 | ≤1e-6 | 通过 |
| RF 合力 Z | 8.5110e-9 | ≤1e-6 | 通过 |
| 30 个中心积分点 S11 | 2.6179e-8 | ≤1e-5 | 通过 |
| 30 个中心积分点 S22 | 3.7387e-8 | ≤1e-5 | 通过 |
| 30 个中心积分点 S33 | 2.5993e-8 | ≤1e-5 | 通过 |
| 30 个中心积分点 S12 | 4.0428e-8 | ≤1e-5 | 通过 |
| 30 个中心积分点 S23 | 2.8959e-8 | ≤1e-5 | 通过 |
| 30 个中心积分点 S13 | 1.6692e-8 | ≤1e-5 | 通过 |

FEM-Python 与 Abaqus 的 RF 合力分别为：

- FEM-Python：(-5,968,505.5024, 16,399,332.7137, 318,396.6327)；
- Abaqus：(-5,968,505.4619, 16,399,332.7012, 318,396.6354)。

C3D4 只有一个体积分点，项目核函数确认全部样本的自然坐标为 (0.25, 0.25, 0.25)，即四面体中心。30 个预先确定的样本覆盖固定边界邻接单元、规定位移邻接单元、两个压力面和全局元素标签分位。Abaqus 分量顺序 `S11,S22,S33,S12,S13,S23` 在比较前显式映射为项目 canonical 顺序 `S11,S22,S33,S12,S23,S13`，没有调整样本、尺度或容差。

执行命令为：

```powershell
& 'D:\Program Files (x86)\Abaqus2023\Commands\abaqus.bat' python 'docs\_phase4_odb_extract.py' 'data\complicate_cae_model.odb' 'docs\_phase4_abaqus_reference.json'
.\.venv\Scripts\python.exe docs\_phase4_complicate_acceptance_runner.py --abaqus-reference docs\_phase4_abaqus_reference.json
```

以上两个临时验收脚本和中间 JSON 在记录结果后删除，没有进入版本控制。ODB 始终按只读方式打开，INP、ODB 和其他 `data/` 文件均未修改。

## 最终测试范围：聚焦验证

用户明确要求不要运行全仓测试，CAD/Gmsh 也不属于本次 PARDISO 求解器替换的依赖范围。因此最终自动化证据以本记录前述 139 项 adapter、solver、C3D20、performance、application 与 GUI 作业聚焦回归为准，不安装 `cad` extra，也不把 CAD 或其他无关 characterization 测试作为 Phase 4 gate。

主 session 曾误启动一次全仓 `pytest -q` 和后续目录诊断；相关命令均已停止，遗留 pytest 子进程已按精确 PID 清理。诊断中出现的 Gmsh 与旧 characterization 问题没有进入本次实现范围，不据此修改依赖、生产代码或无关测试。

## 已知限制

- 首版验收平台限定为 Windows x86-64、CPython 3.13 和项目 `uv` 环境。
- py-mkl-pardiso 0.0.5 无法报告精确 oneMKL build。
- native factorization 的取消仍是返回后的协作式取消，不能承诺中途打断。
- 当前代表模型实测使用 1 个 PARDISO OpenMP 线程；本 Phase 只验收当前默认线程配置，没有开展线程扩展性调优。
- Abaqus `PRESELECT` field/history 输出请求在项目中仍显示为不支持的输出 warning；本次 Abaqus 数值 gate 使用只读 ODB API 和项目公共应力恢复核独立完成。

## 后续建议

1. 若后续设置 `OMP_NUM_THREADS` 或 MKL 线程变量，应在导入 NumPy、SciPy 或 pymklpardiso 前固定配置，并重新记录同一代表模型的时间、内存和数值误差。
2. 若需要 native factorization 中途取消，应单独规划 PARDISO 可中断边界；当前协作式取消语义保持不变。
3. 若上游 wheel 增加 oneMKL build API，在运行时记录中补充精确 native build；无需扩散第三方调用到私有适配器之外。
4. `complicate_cae_model.inp/.odb` 继续只作为人工验收资产，不得加入 pytest fixture、golden data 或 CI 依赖。

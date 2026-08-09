# FEM-Python 模型与结果持久化实施交付记录

- 日期：2026-08-09
- 分支：`develop/fempy-femres-persistence`
- 起点：`4e3b756`

Phase 6 工作区状态：本提交由主 session 最终提交。

## Phase 提交范围

| Phase | 提交 |
| --- | --- |
| 0 | `7f9718b` |
| 1 | `8caa928` |
| 2 | `3bebfe3` |
| 3 | `296949c` |
| 4 | `8f04828` |
| 5 | `9f4c2f6` |
| 6 | 主 session 最终提交 |

## Phase 6 变更

- `src/fem/io/result_archive_v1.py` 在 ZIP entry 被读取前增加 untrusted-input 边界：单 entry 512 MiB、总解压 1 GiB、压缩比 10,000；manifest 数组声明的 dtype/shape/nbytes 必须精确一致且不超过 512 MiB。数组仍固定使用 `allow_pickle=False`，不接受 object/complex dtype。
- 同一 reader/writer 合同增加 1 GiB container、4096 entry、8 MiB manifest 限制及压缩比限制；路径 load、atomic readback 和 bytes/bytearray decode 都在 `ZipFile`/JSON parse 前执行 bounded read/length 检查。writer 在生成 `.npy`、manifest、ZIP 和返回 bytes 前执行同样的 size/count/total/compression-ratio 校验。
- `tests/io/test_result_archive_phase6_hardening.py` 增加确定性的 7 字段、98,304 记录 fixture；验证 archive save/load、result-only install 和 save prepare 共享只读矩阵身份，不进行第二次全矩阵复制。
- 同一测试文件覆盖 declared size/shape、compression ratio 和 create/open/write/flush/fsync/readback/verify/replace/cleanup 原子写失败注入。每次失败都保留旧目标；清理阶段故障的临时文件由测试显式回收，随后真实 helper 可重试成功。

## 性能实测

使用 `.venv`、单进程、`pytest -q -s`，fixture 固定为 7 fields / 98,304 records，archive 1,631,090 bytes：

```text
write: 14.386 s, tracemalloc peak 70,016,249 B
read:  10.653 s, tracemalloc peak 68,433,219 B
```

这是一次同机环境测量，没有与独立 baseline 做倍率比较，也不设置硬件无关阈值。计时包含 schema 编码、ZIP、readback 验证和严格解码；主线程只接收 immutable worker payload。GUI save/open 的 worker、事件循环和 cooperative cancel 由 Phase 5 垂直测试覆盖。

## 安全矩阵

- exact manifest/entry allow-list、canonical order、duplicate/unknown/missing entry、危险路径、checksum、truncated/corrupt ZIP：已有 v1 codec tests。
- object/complex/pickle payload、dtype/shape/nbytes/finite 值、mask、region/index 边界：已有 v1 codec tests；Phase 6 增加超大声明和 ZIP 压缩比边界。
- Phase 6 增加 container length、entry count、manifest size、single raw entry、total expanded bytes、array nbytes、compression-ratio 和 writer/reader 对称限制；限制在 ZIP 解析、manifest JSON、NumPy 解码和 atomic readback 前生效。
- 原子事务覆盖临时文件创建、打开、写入、flush、fsync、readback/semantic verify、replace 和 cleanup 注入；失败保留旧目标并清理（cleanup 故障本身除外，测试验证诊断后回收）。

## 入口与迁移矩阵

- legacy `.femproj` → `.fempy` 的源字节保持和首次 Save As 由 `tests/io/test_project_migration.py`、`tests/gui/test_project_io.py` 覆盖。
- `.fempy` 打开、求解、保存 `.femres`、关闭、standalone `.femres` 打开由 Phase 5 GUI vertical test 覆盖。
- INP/headless result output smoke：`tests/integration/test_result_output_workflow.py` 2 passed；`tests/architecture/test_public_gui_workflows.py` 2 passed。
- native/inp GUI integration 的组合命令在本机 180 秒后仍停留在 gmsh/meshing 输出，未作为 Phase 6 硬阈值；未发现与 Phase 6 codec 改动相关的 traceback。

## 聚焦验证

以下均未运行全量 pytest（按用户要求使用 focused tests）：

```text
pytest -q tests/io/test_result_archive_phase6_hardening.py                         22 passed
pytest -q tests/io/test_result_archive_v1.py tests/io/test_result_archive_atomic.py
          tests/application/test_result_archive_session.py
          tests/application/test_result_archive_session_install.py                  57 passed
pytest -q tests/io/test_result_archive_v1.py tests/io/test_result_archive_atomic.py
          tests/io/test_result_archive_phase6_hardening.py
          tests/application/test_result_archive_session.py
          tests/application/test_result_archive_session_install.py
          tests/gui/test_result_document_workflow_phase5.py                         96 passed
pytest -q tests/gui/test_action_states.py tests/gui/test_action_projection.py
          tests/gui/test_main_window_layout.py tests/gui/test_result_export_commands.py
          tests/gui/test_result_document_workflow_phase5.py                         98 passed
pytest -q tests/architecture/test_result_provider_boundaries.py
          -k 'not agent_imports_are_confined_to_phase5_gui_runtime_adapter'          7 passed, 1 deselected
pytest -q tests/architecture/test_public_gui_workflows.py                            2 passed
pytest -q tests/integration/test_result_output_workflow.py                           2 passed
ruff check src/fem/io/result_archive.py src/fem/io/result_archive_v1.py
          tests/io/test_result_archive_phase6_hardening.py                            passed
git diff --check                                                                        passed
```

`.venv` 没有 Ruff，按仓库 fallback 使用全局 `ruff`；未新增 skip/xfail。

## 静态审计

- 项目文件菜单和 Ribbon 的 visible tuple 均为 `new_native, open_project, save_project, open, save_result, open_result`；未发现 visible `reload`/`close`。
- `src/fem/io/result_archive_v1.py` 仅出现 `allow_pickle=False` 两处；没有 `allow_pickle=True` 或 `dtype=object`。
- `src/fem` 没有 `fem_agent` 反向 import。GUI adapter 中仍有 agent authoring imports；既有 architecture allowlist 测试因此失败，见残余风险。
- `.femproj` 命中均位于 legacy codec/router、迁移兼容、fixture 或相应测试；未将其重新作为新模型输出。

## 已知非本计划失败和限制

以下在本计划基线或 Phase 5 审阅中已存在，本 Phase 没有扩展修复：

1. BeamFrameField 非 JSON signature：`tests/application/test_session_invalidation.py::test_beam_orientation_edit_and_clear_recompile_and_invalidate`、`tests/application/test_task_tokens.py::test_orientation_edit_rejects_old_validation_and_solve_callbacks`。
2. import deepcopy：`tests/application/test_task_tokens.py::test_prepared_import_transfers_worker_owned_model_without_second_copy`。
3. GUI agent import allowlist drift：`tests/architecture/test_result_provider_boundaries.py::test_agent_imports_are_confined_to_phase5_gui_runtime_adapter`。
4. GUI compact CSV 与 strict reader header：`tests/gui/test_workflow.py::test_gui_exports_the_current_result_field_as_csv_and_vtk`。
5. typed live projection fixture 首节点：基线 `4e3b756` 已同样失败。
6. 本次 focused project migration 命令另观察到以下 4 个既有 `MeshSettings` v1 field-contract failures（`auto_level`/`strict_cell_shape`），与 result archive 改动无关：
   - `tests/io/test_project_migration.py::test_current_writer_reverse_matrix_round_trips_both_profiles`
   - `tests/io/test_project_migration.py::test_current_writer_rejects_unsupported_falloff_and_unproven_radius_target`
   - `tests/io/test_project_migration.py::test_current_writer_rejects_multiple_target_radius_controls`
   - `tests/io/test_project_migration.py::test_current_writer_dedupes_same_control_and_rejects_conflicting_size`

全量 pytest 未运行；native/inp GUI integration 的长时间 gmsh 运行也没有转为硬阈值。生产仍保留底层 `reload_model`/`close_model` public workflow，Phase 5 只退休了可见控件位置。新图标、自动保存、多 run 容器、history/frame 和结果 attach 均不在本次范围。

## 使用与迁移

新模型保存使用 `.fempy`；旧 `.femproj` 继续可读，首次显式保存通过 Save As 生成 `.fempy` 且不覆盖旧字节。成功作业在“保存结果”中生成 `.femres`；结果文件可在无原模型/INP 的进程中独立打开，进入只读 result-only Session，继续查询、检查、云图、CSV、VTK 和截图。归档不包含 pickle、可执行对象或绝对源路径。

# FEM Agent Profile Transform Reliability：Phase 0 基线

日期：2026-08-06
范围：仅冻结 Profile 拉伸/扫掠的现有几何与 Provider 请求证据。

本步骤只新增测试、测试 fixture、只读 Provider request capture 和本记录；没有修改
system prompt、工具 schema、动态工具可见性、geometry recipe/compiler、Session 或
GUI 行为，也没有修改 README。工作树中预先存在的 viewport 导出相关修改保持不变。

## 可重复 fixture

`tests/fixtures/profile_transform_baseline.py` 构造严格 XY 同心圆环：外半径 50 mm、
内半径 25 mm。两个圆形成一个 material Profile 和一个嵌套 hole，canonical source face
为 `face:profile/28c27031fe7162d7`。

冻结的 feature catalog 事实：

- dimension：2；exact：true；
- material Profile 数：1（`face:profile/28c27031fe7162d7`）；
- hole lineage：`edge:hole-loop` → `edge:C2`；
- feature summary：`点=2，曲线=2，Profile=1，孔=1`。

以高度 10 mm 拉伸该 Profile 后，detached topology 证明一个 Body、两个端面、外侧面
和孔侧面：

| logical ID | semantic role |
| --- | --- |
| `face:bottom` | `copy.bottom.sketch.profile` |
| `face:top` | `copy.top.sketch.profile` |
| `face:side/C1` | `sweep.boundary.outer` |
| `face:side/C2` | `sweep.boundary.hole` |
| `body:domain` | `sweep.domain` |

在 `MeshSettings(20.0, cell_shape="tetrahedron")` 下，真实 Gmsh 生成 88 个节点、192
个 `Tet4` 单元。

## MESH_READY 工具发布基线

使用现有 GUI authoring bridge 接受二维环 Profile 后，workflow stage 为 `mesh_ready`，
实际发布顺序为：

1. `read_authoring_context`
2. `read_geometry_feature_catalog`
3. `set_authoring_requirements`
4. `read_mesh_refinement_context`
5. `read_geometry_edit_context`
6. `prepare_geometry_edit`
7. `request_project_save`
8. `read_deletable_objects`
9. `prepare_delete_proposal`

`prepare_geometry_edit` 的现有 schema 暴露以下 Profile transform seam：

- `extrude_profiles`
- `revolve_profile`
- `path_sweep_profile`

Phase 0 测试冻结本轮每个已发布 ToolDefinition 的完整（名称、描述、参数）SHA-256，
用于后续区分工具未发布和 Provider 未调用：

| tool | schema SHA-256 |
| --- | --- |
| `read_authoring_context` | `423e6e7a45db09bd401d9a3936016b985c1e934d1c283b8bbb8b8ccbcb2cda42` |
| `read_geometry_feature_catalog` | `e1a6abffa6f9d6e0920062850071c8e572f891e8333972f3b5b937a736f254ea` |
| `set_authoring_requirements` | `7b186040e13f73dceca7ff7eb2b3dd55367bf1e22cb5f3b0171484052114369d` |
| `read_mesh_refinement_context` | `1fe75ea3901584a465dc55ee4584306f4e02026cf7aad1442b1e12f0e5506369` |
| `read_geometry_edit_context` | `606a43aa3e97b68834fc0d2dcfb05e615a4e69a6b38224b17e718b42a03e01e3` |
| `prepare_geometry_edit` | `0d56b9760766e8125c05e84df38f45d569818df5d57afda93f7ef12e01d5bc95` |
| `request_project_save` | `d44c4954dade2d24c2c4c8561f459e2e2478a7e13deb1a3e0c357b3de29da9ab` |
| `read_deletable_objects` | `0dbb5d89c21c9dad6fb2ff2b3c0f9961292f9d044d9cf636f12e3a10d45fa0d3` |
| `prepare_delete_proposal` | `19e90f5ea80eaca6f9827fc090ff59729d8e7d86397f38456f60b1374a4ed6a2` |

## 最小故障证据

`tests/helpers/profile_transform_capture.py` 的 `RequestCaptureProvider` 每轮只保留：

- redacted system context；
- 已发布的 tool names；
- 每个完整 ToolDefinition 的 schema hash。

它不保存用户消息、tool arguments、完整 tool result、session ID、绝对路径、业务名称或
credentials。当前回合最小化为：

| fact | captured value |
| --- | --- |
| user request | `拉伸成3d` |
| authoring capability | `mesh_ready`，`prepare_geometry_edit` 已发布 |
| Provider tool calls | none |
| final capability statement | `拉伸不受支持；必须先生成网格。` |

捕获到的 engine `Current local state (structured metadata only)` 仅包含通用分析 session
元数据，不包含 `active_part_id`、`recipe_kind` 或 Profile transform 摘要；同时同一轮的
tool catalog 已包含 `prepare_geometry_edit`。这固化了“底层 seam 已可见但 Provider 在
关键回合没有调用读取/准备工具”的当前对照，不把错误声明当作几何能力事实。

## 验证记录

环境（仓库 `.venv`）：Python 3.13.11、pytest 9.1.1、NumPy 2.3.5、SciPy 1.16.3、
PySide6 6.11.1、Gmsh 4.15.2。

| command | result | elapsed |
| --- | --- | ---: |
| `.venv\Scripts\pytest.exe -q tests/gui/test_agent_profile_transform_baseline_phase0.py` | 6 passed | 1.09 s |
| `.venv\Scripts\pytest.exe -q tests/gui/test_agent_profile_extrusion_phase2.py tests/gui/test_agent_profile_sweep_phase3.py` | 25 passed | 2.25 s |
| `.venv\Scripts\pytest.exe -q tests/geometry/test_profile_extrusion.py` | 7 passed | 0.61 s |
| `.venv\Scripts\pytest.exe -q tests/integration/test_profile_extrusion_headless.py` | 5 passed | 1.25 s |
| combined 57-test Profile extrusion/sweep focus command | 57 passed | 5.25 s |
| `git diff --check` | passed | — |

Ruff 不在当前 `.venv` 或全局 PATH 中，因此本步骤未执行 Ruff；没有因此替换为其他
静态检查。真实 Provider/云凭据冒烟测试未执行，离线基线只使用 deterministic capture
Provider。

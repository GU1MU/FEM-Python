# FEM Agent Profile Transform Reliability：Implementation Handoff

日期：2026-08-07

分支：`develop/fem-agent-profile-transform-reliability`

范围：计划 `docs/2026-08-06-fem-agent-profile-transform-reliability-plan.md` 的
Phase 0--7。本文是独立完工审计记录；只记录已经落地和已经验证的事实，不把
Fake Provider 控制流测试写成远程模型能力证明。

## 1. 结论与边界

Phase 0--6 已按串行流程落地为七个独立提交，Phase 7 交接文档由主 session 审阅后
提交。接受二维 native Profile 后，Agent 现在可以通过 revision 绑定的专用 read/prepare
工具提案拉伸、旋转或受限路径扫掠；空白项目可以在一张 proposal 中表达 Profile、hole
与三维变换。GUI 控件仍是唯一确认授权，确认前 proposal 与 Session 保持 detached，
失败、拒绝、取消和 stale 路径保持模型不变。

步骤 6 的聚焦、相关回归和静态门已通过；完整套件按计划在首个既有基线失败处停止，
两个额外的 1D/2D/3D 回归失败也已独立复现并证明与本分支修改文件无交集（见第 6 节）。
真实 Provider 云端冒烟本次未执行；默认情况下测试跳过，本文不声称任何真实模型能力。

## 2. 串行提交与实际改动

以下短哈希均在当前分支解析到不可变提交；文件列表是实际提交内容，而不是计划中的
目标文件猜测。

| Phase | 提交 | 实际文件 | 主要公共契约/行为 |
| --- | --- | --- | --- |
| 0 | `9485f48` | `docs/2026-08-06-fem-agent-profile-transform-reliability-phase-0-baseline.md`；`tests/fixtures/profile_transform_baseline.py`；`tests/gui/test_agent_profile_transform_baseline_phase0.py`；`tests/helpers/profile_transform_capture.py` | 冻结 R50/R25/H10 同心圆环、严格 Profile/hole lineage、MESH_READY 工具与 schema hash，以及不调用工具的最小故障证据。 |
| 1 | `c491793` | `src/fem_agent/__init__.py`；`src/fem_agent/authoring_runtime.py`；`src/fem_agent/engine.py`；`src/fem_agent/tools/registry.py`；`src/fem_gui/agent_runtime.py`；`tests/gui/test_agent_profile_transform_snapshot_phase1.py` | 每轮 immutable `AuthoringTurnSnapshot`、owner-thread cache、freshness/revision 绑定、bounded provider projection 和 round/tool schema audit。 |
| 2 | `c6c9621` | `src/fem_agent/engine.py`；`src/fem_agent/routing.py`；`tests/gui/test_agent_profile_transform_baseline_phase0.py`；`tests/gui/test_agent_profile_transform_routing_phase2.py` | `GeometryRouteHint` 的中英文窄意图分类；mesh intent 与 path sweep 分离；缺 probe 的误拒绝最多一次本地纠正，第二次返回有界恢复。 |
| 3 | `5c3b2d0` | `src/fem_agent/__init__.py`；`src/fem_agent/authoring_runtime.py`；`src/fem_agent/engine.py`；`src/fem_agent/geometry_authoring.py`；`src/fem_agent/routing.py`；`src/fem_gui/agent_authoring.py`；`tests/gui/test_agent_profile_extrusion_phase2.py`；`tests/gui/test_agent_profile_sweep_phase3.py`；`tests/gui/test_agent_profile_transform_baseline_phase0.py`；`tests/gui/test_agent_profile_transform_routing_phase2.py` | `read_profile_transform_context`、`prepare_profile_extrusion`、`prepare_profile_revolution`、`prepare_profile_path_sweep` 一等工具；本地唯一 material Profile 解析；显式 ID 同 revision 校验；旧 dispatch 兼容。 |
| 4 | `2224d4b` | `src/fem/application/recipe_compiler.py`；`src/fem_agent/authoring.py`；`src/fem_agent/authoring_runtime.py`；`src/fem_gui/agent_authoring.py`；`tests/gui/test_agent_composite_geometry_phase4.py` | `prepare_geometry_proposal` 新增 `extruded_profiles` 与 `path_swept_profile`；严格 XY Profile/hole containment、exact 单 Body/正体积 detached preflight、原子 blank-to-3D proposal、保存重开和路径 lineage。 |
| 5 | `c1755c9` | `src/fem_agent/__init__.py`；`src/fem_agent/diagnostics.py`；`src/fem_gui/agent_authoring.py`；`tests/gui/test_agent_profile_transform_diagnostics_phase5.py`；`tests/test_agent_authoring_phase_a8.py` | `ProfileTransformDiagnostic` 及 13 个稳定 code；bounded UTF-8 recovery、候选/首个失败 member、异常路径清洗；preflight 失败和 GUI 记录刷新保持原子。 |
| 6 | `0e183b7` | `tests/gui/test_agent_profile_transform_phase6.py` | 跨 engine/runtime/controller/GUI/Gmsh 的 Fake Provider 控制流、ring/center-hole plate/path 真实几何网格、保存重开、continuation 和负例验收矩阵；云端测试显式 gated。 |

## 3. 工具发布矩阵与兼容策略

### 3.1 发布矩阵

Phase 0 冻结的二维环在 `MESH_READY` 的旧发布目录为：

1. `read_authoring_context`
2. `read_geometry_feature_catalog`
3. `set_authoring_requirements`
4. `read_mesh_refinement_context`
5. `read_geometry_edit_context`
6. `prepare_geometry_edit`
7. `request_project_save`
8. `read_deletable_objects`
9. `prepare_delete_proposal`

旧的 `prepare_geometry_edit` schema 在其 `oneOf` 中包含
`extrude_profiles`、`revolve_profile`、`path_sweep_profile` 三个 transform 分支。

当前在可写 native-ready stage（包括 `MESH_READY`）另外发布四个专用工具，插在
`read_geometry_feature_catalog` 之后：

- `read_profile_transform_context`
- `prepare_profile_extrusion`
- `prepare_profile_revolution`
- `prepare_profile_path_sweep`

因此二维环的 MESH_READY 目录在没有当前网格时为旧的 9 项加上述 4 项（共 13 项；
若已有其他满足条件的 handler，目录还会按原有动态门控增加其工具）。pending、stale、
cancelled 等不可写 stage 仍只发布 `read_authoring_context`。通用
`prepare_geometry_edit` 仍可调用，但 Provider 可见 schema 已移除上述三个 transform
分支，仅保留草图、刚体和精确 Boolean 编辑；本地兼容 dispatch 与已保存项目没有被拆分。

### 3.2 专用输入契约

- `read_profile_transform_context`：`part_id`。
- `prepare_profile_extrusion`：`part_id`、`profile_selection`、正 `height`；
  `unique_material_profile` 只在本地证明恰有一个 material Profile 时成立，显式
  canonical face ID 数组作为 `profile_selection` 时要求来自同一次 read 的
  `context_revision`。
- `prepare_profile_revolution`：上述 selection，加 `axis ∈ {x,y,z}` 和
  `0 < angle_degrees ≤ 360`。
- `prepare_profile_path_sweep`：上述 selection，加有序开放 path（bounded points
  与 members）和 `frame_strategy ∈ {fixed,transport}`。

所有新 schema 为 exact object（`additionalProperties: false`），canonical
`face:*`、Part/Body 和 revision 是唯一 provider 几何身份；OCC/Gmsh tag、网格编号、
Qt/VTK 对象和完整 B-rep 不进入 provider payload。

### 3.3 Blank composite 契约

`prepare_geometry_proposal` 的 `geometry.oneOf` 现包括：

- `extruded_profiles`：bounded rectangle/circle/polygon contour 数组（material 或
  hole 语义由本地 containment 证明）、正 `height`，可选 `provisional`；
- `path_swept_profile`：一个闭合 material Profile（可含 hole）、bounded 有序开放
  path 和 `fixed`/`transport` frame。

两种 variant 都先构造严格 XY sketch，再 detached preflight exact topology、正体积、
预期一个 Part/一个 Body，并由一张 GUI card 原子提交。确认前空白项目仍为空，接受后
直接进入三维 geometry stage；不会提前建立 mesh、材料、截面或分析定义。

### 3.4 Schema/version 与兼容

- `AuthoringTurnSnapshot`：`schema_version="1"`，`AUTHORING_TURN_SNAPSHOT_MAX_BYTES=8192`，
  最多 96 个名称；`available=false` 的 snapshot 不得从历史补值。
- `profile_transform_context`：`schema_version=1`；geometry recipe provider payload
  schema 为 1；现有 `ExtrudedGeometry`、`RevolvedGeometry`、`PathSweptGeometry`
  类型和项目 codec 继续复用。
- AuthoringContext 既有 `AUTHORING_SCHEMA_VERSION="1.0"` 不变；engine conversation
  schema 仍为 1，tool-audit 写入 schema 2，同时兼容读取旧 schema 1。audit 只保留
  session/round、stage/revision、published tool names、schema hashes、route hint 和
  read/prepare 调用标记，不保存 arguments、完整结果、绝对路径或 credentials。
- `.femproj` 当前项目 schema 仍为 13；本计划没有项目 schema migration，也没有改变
  已保存 recipe 的解码版本。新增工具只是现有 recipe/proposal/bridge 的发现层。

## 4. 失败与授权语义

`GeometryRouteHint` 公开 `requested_operation`、`target_part_dimension`、
`required_probe_tool`、`required_prepare_tool`、`mesh_prerequisite`、
`missing_fields`、`allow_arbitrary_size`、`intent_kind`。明确 transform 请求先要求
`read_profile_transform_context`；普通“扫掠”只返回 `sweep_type` 歧义；“扫掠六面体
网格”被标记为 meshing，不映射为 geometry sweep；缺高度/路径时只追问决定性字段。

当专用 probe 和 prepare 已发布而 Provider 首轮只返回“工具不支持/先网格”文本时，
engine 拦截该文本并追加一次 bounded、non-authorizing correction。第二次仍无 probe
调用时返回本地中英文恢复消息，不声称底层能力不存在；有 typed unsupported 诊断、
取消或正常缺参问题时 guard 不拦截。正确工具调用仍必须经过 local detached preflight
与唯一 GUI confirmation，接受后旧 mesh/下游依赖按现有 revision 规则失效。

Phase 5 的稳定 code 为：

`profile-transform.part-not-found`、`source-not-planar`、`source-not-strict`、
`no-material-profile`、`ambiguous-material-profiles`、`invalid-source-id`、
`nonpositive-height`、`invalid-path`、`unsupported-frame`、`topology-unproven`、
`unexpected-body-count`、`stale-context`、`preflight-failed`。

每项返回 bounded `code/message/operation/retryable/required_fields/preserve_draft`，
歧义可带 `candidates`，路径错误可带 `first_failed_member`；UTF-8 预算会保留完整
recovery 指引，路径/traceback/credential 不会泄露。未知 code fail-fast，编程错误仍
走通用 `INVALID_TOOL_ARGUMENTS`，不被伪装成几何诊断。

## 5. 真实几何与网格验收结果

- **冻结圆环**：外半径 50、内半径 25、高度 10 mm 的严格 XY ring 生成一个
  exact Body、`face:bottom`、`face:top`、outer side 和 hole side；在
  `MeshSettings(20.0, cell_shape="tetrahedron")` 下真实 Gmsh 生成 88 个节点和
  192 个 `Tet4`。
- **专用 ring E2E**：Phase 6 用外半径 5、内半径 2、高度 4 的 ring 走
  `read_profile_transform_context → prepare_profile_extrusion → GUI accept`，保存/重开
  后 recipe 相等，topology exact、一个 selectable Body、hole side lineage 保留；
  `order=1` 生成非空纯 `Tet4`，`order=2` 生成非空纯 `Tet10`。
- **中心孔平板**：blank composite 以 `10×6` 矩形、中心半径 1 hole、高度 2 生成
  一张最终三维 proposal；接受前 snapshot 原子，接受后 exact 一个 Body、bottom/top
  和 canonical hole side `face:side/C1`（可选择），真实 Gmsh 生成非空纯 `Tet4`，
  保存/重开保留该 hole selection；geometry 接受瞬间 mesh/material/section/assignment/
  step/artifact 仍为空，证明没有用网格删除单元伪造孔洞。
- **路径扫掠**：`1×1` 矩形 Profile 沿 `A(0,0,0)→B(0,0,3)→C(2,0,4)` 的两段开放
  path，`fixed` 与 `transport` 均保持 member 顺序 `AB,BC`、exact 一个 Body、端面/侧面
  lineage；保存/重开 recipe 相等，并进入当前 Tet 链路生成非空纯 `Tet4`。带中心
  `r=0.15` hole 的 fixed/transport 变体同样保留 hole side 且生成纯 `Tet4`。

以上结果来自 committed Phase 0/3/4/6 tests 的真实 recipe compiler/Gmsh 调用；Fake
Provider 场景只证明控制流和授权边界。

## 6. 验证记录

所有命令使用仓库 `.venv\Scripts\python.exe`（Python 3.13.11，pytest 9.1.1，Gmsh
4.15.2）。

### 6.1 聚焦与回归

| 命令（省略重复前缀时仍以 `.venv\Scripts\python.exe -m pytest` 执行） | 结果 | 耗时 |
| --- | ---: | ---: |
| `-q tests/gui/test_agent_profile_transform_baseline_phase0.py tests/gui/test_agent_profile_transform_snapshot_phase1.py tests/gui/test_agent_profile_transform_routing_phase2.py tests/gui/test_agent_profile_extrusion_phase2.py tests/gui/test_agent_profile_sweep_phase3.py tests/gui/test_agent_composite_geometry_phase4.py tests/gui/test_agent_profile_transform_diagnostics_phase5.py` | 111 passed | 15.19 s |
| `-vv --maxfail=1 tests/test_agent_engine.py tests/gui/test_agent_event_contract.py tests/test_agent_authoring_contracts.py tests/test_agent_provider_contract.py` | 107 passed | 72.32 s |
| `-vv --maxfail=1 tests/gui/test_agent_runtime.py tests/gui/test_agent_continuation_e2e.py` | 21 passed | 153.62 s |
| `-vv --maxfail=1 tests/gui/test_agent_authoring_e2e_phase_a8.py tests/gui/test_agent_authoring_bridge.py tests/gui/test_agent_chat_overlay.py tests/gui/test_agent_authoring_recovery_phase_a8.py tests/gui/test_agent_event_contract.py` | 66 passed | 59.35 s |
| `-vv --maxfail=1 tests/io/test_profile_extrusion_project.py tests/gui/test_agent_project_save_gui.py tests/application/test_native_mesh_contract.py` | 23 passed | 2.99 s |
| `-vv --maxfail=1 tests/gui/test_agent_profile_extrusion_phase2.py tests/gui/test_agent_profile_sweep_phase3.py` | 36 passed | 2.91 s |
| `-vv --maxfail=1 tests/gui/test_profile_extrusion_workflow.py tests/gui/test_profile_sweep_workflow.py tests/integration/test_profile_extrusion_headless.py tests/integration/test_profile_sweep_headless.py` | 26 passed | 4.13 s |
| `-q tests/test_agent_authoring_phase_a2.py tests/test_agent_authoring_phase_a3.py tests/test_agent_authoring_phase_a4.py tests/test_agent_authoring_phase_a5.py tests/test_agent_authoring_phase_a6.py tests/test_agent_authoring_phase_a7.py tests/test_agent_authoring_phase_a8.py` | 64 passed | 1.41 s |
| `-q tests/gui/test_agent_profile_transform_phase6.py` | 10 passed, 1 skipped (cloud) | 3.05 s |
| `-q --collect-only` | 5556 collected | 4.69 s |

Phase 0 的基线记录另外冻结了：baseline 6 passed/1.09 s；Profile extrusion/sweep
25 passed/2.25 s；geometry extrusion 7 passed/0.61 s；headless integration 5 passed/
1.25 s；合并 57 passed/5.25 s。Phase 6 的最终专用测试已扩展为上表的 10 passed/1
skipped；其中新增中心孔平板测试 1 passed/1.26 s，composite/profile 相关三文件
批次 54 passed/12.50 s。

### 6.2 全量与既有失败

全量命令：

`.venv\Scripts\python.exe -m pytest -q --maxfail=1`

在 **405 passed，6.25 s** 后首败：

`tests/application/test_authoring_candidates.py::test_automatic_rectangle_is_enabled_but_local_load_is_limited`

断言期望 `line_load.status is AuthoringStatus.LIMITED`，实际为
`AuthoringStatus.ENABLED`。

显式 1D/2D/3D 维度回归命令（`pytest -vv --maxfail=1`，以下 10 个文件，收集 88）：

`tests/test_agent_native_1d_mesh_phase2.py tests/gui/test_agent_native_1d_mesh_bridge_phase2.py tests/integration/test_agent_native_1d_mesh_gmsh_phase2.py tests/integration/test_native_1d_authoring.py tests/integration/test_native_1d_gui_workflow.py tests/test_agent_authoring_phase_a2.py tests/gui/test_profile_extrusion_workflow.py tests/gui/test_profile_sweep_workflow.py tests/test_agent_native_3d_feature_contract_phase1.py tests/integration/test_agent_native_3d_analysis_phase5.py`

结果为 **25 passed/1 failed，2.95 s**。首个失败 nodeid：

`tests/integration/test_native_1d_gui_workflow.py::test_native_1d_public_gui_workflow_persists_checks_solves_and_displays[Beam2]`

保存重开后 `window.document.steps[0].line_loads == (local_load,)` 失败；实际
`LineLoad.name == "载荷-兼容-Load-线-1"`，输入/期望名称为 `None`。

2D/3D 补测（命令取上述 10 文件的最后 5 个：
`tests/test_agent_authoring_phase_a2.py tests/gui/test_profile_extrusion_workflow.py tests/gui/test_profile_sweep_workflow.py tests/test_agent_native_3d_feature_contract_phase1.py tests/integration/test_agent_native_3d_analysis_phase5.py`）结果为
**54 passed/1 failed，5.38 s**。首个失败 nodeid：

`tests/integration/test_agent_native_3d_analysis_phase5.py::test_phase5_logical_face_region_survives_remesh_and_reopen`

`src/fem/application/native_scope_materialization.py:149` 访问
`LogicalEntityRef.part_id`，但该对象没有此属性，触发 `AttributeError`。

上述三个失败涉及的测试/生产文件均不在本分支七阶段修改集合中：
`tests/application/test_authoring_candidates.py`、`tests/integration/test_native_1d_gui_workflow.py`、
`tests/integration/test_agent_native_3d_analysis_phase5.py`、
`src/fem/application/native_scope_materialization.py` 以及 LineLoad/LogicalEntityRef 定义
均未被修改。因此它们记录为既有基线回归，不是本计划的修复项。

### 6.3 静态门

```powershell
$phaseFiles = git diff --name-only main...HEAD -- '*.py'
& 'C:\Users\25485\.local\bin\ruff.exe' check $phaseFiles
# All checks passed!（22 个本分支 Python 文件）

$sourceFiles = $phaseFiles | Where-Object { $_ -like 'src/*' }
& .venv\Scripts\python.exe -m py_compile $sourceFiles
# exit 0（11 个 src 文件；测试目录存在已知 ACL/cache 限制，故只编译源码）

git diff --check main...HEAD
# exit 0
```

真实 Provider 冒烟测试的 gating 为 `FEM_AGENT_CLOUD_SMOKE=1`、
`FEM_AGENT_CLOUD_SMOKE_CONFIG` 指向绝对路径的外部配置、provider=`deepseek` 且有 API
key；配置将 timeout≤30 s、retries=0、output≤256。六个受控 prompt 是：
`拉伸成3d`、`把这个截面加厚到 20 mm`、`extrude this profile by 10 mm`、
`沿 A-B-C 这条路径扫掠`、`做扫掠六面体网格`、`这个功能支持吗`。每个 prompt 最多
三轮 `complete`；判定依据是工具调用：transform 先 read 再 dedicated prepare（缺高
度的第一条可只读）、swept mesh 只调用 `prepare_mesh_proposal`、support question
不创建 proposal。当前环境未提供显式配置，因此 Phase 6 cloud/integration test
**skipped**，没有网络调用、密钥输出或真实工具调用结果。

## 7. 遗留事项与下一计划边界

- 处理第 6 节三个既有失败应另开兼容性修复：LineLoad 自动命名/Beam2 保存重开契约，
  以及 `native_scope_materialization` 对 `LogicalEntityRef` 的字段假设；不得把它们回填
  到本计划提交。
- 有凭据时可单独运行六提示 Provider matrix，按工具调用与最终状态判定，而不是按回复
  文本逐字匹配；本次没有执行，故不能推出真实模型可靠性。
- 本计划仍不包含 loft、shell、fillet/chamfer、装配约束、分支/自交/闭合/曲线路径、
  负向/双向/对称/中面/拔模拉伸、自动合并多个 material Profile、六面体 swept mesh
  保证、imported INP 逆向 CAD、任意代码执行或 GUI 自动点击。
- 路径首版仍限 straight ordered open polyline 与 `fixed`/`transport`；失败返回稳定
  诊断，不自动修复或重排路径。

## 8. 预存用户工作树变更

主 session 审计确认下列四个未提交文件从 Phase 0 前即存在，七阶段均未修改、未 stage、
未纳入任何提交：

| 文件 | `git diff -- <path> | git hash-object --stdin` patch-stream fingerprint |
| --- | --- |
| `src/fem_gui/viewport_image_export_dialog.py` | `b9614ba336f888665116c545df17a61a0652fc9e` |
| `src/fem_gui/widgets/viewport.py` | `2dc307df7c0f8345775cdbc60989a4d02424e77b` |
| `tests/gui/test_typed_result_viewport.py` | `2fadc0d08973ce165b9d1b540524228c746ea791` |
| `tests/gui/test_viewport_image_export.py` | `8437ac5c1b74c52039ed616b847edb0afc240e08` |

上述 fingerprint 是当前未暂存 patch 字节流的 Git blob object ID，并已在 Phase 0 前审计
到 Phase 6 提交后多次复算一致。本文没有修改 README、计划源文档或上述用户文件。

## 9. 交接结论

可交付范围已经满足计划的 geometry/Agent reliability 目标：typed context → dedicated
transform tool → detached preflight → GUI-only confirmation → revision-aware
continuation → Tet mesh 的链路有本地证据，且所有失败路径保持原子。继续工作时应先
保留本 handoff 与七个不可变提交，再单独处理既有 1D/3D 回归或显式云端 Provider 验证。

# FEM Agent V1 Authoring Phase A3 实施计划

## 状态

- 日期：2026-07-30
- 基线：A2 提交 `3668fef`
- 阶段：A3（网格意图、局部加密和确认生成）
- 状态：已实现并通过主 Agent 审查
- 绑定边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

1. 建立统一、严格、可 JSON 持久化的 `MeshIntent`；显式全局尺寸与
   `AutoMesh` level 必须且只能选择一种。
2. 复用 `MeshSettings`、`AutoMeshSpec`、稳定 `LogicalEntityRef` 和类型化
   `MeshSizeFalloff`，覆盖二维三角形/四边形、一阶/二阶及孔边局部加密。
3. 建立全进程单一所有者 Gmsh 协调器，使 GUI 和 Agent 网格生成共用同一串行边界。
4. 只有 A1 proposal 经 GUI“开始划分”控件授权后才构造 detached 后台任务。
5. 后台运行期间不修改当前 GUI 模型；完成后重新核对
   document/session/revision，并在一个 session 事务中提交网格意图和生成模型，
   成功后只刷新一次。
6. 失败、取消、陈旧和严格单元形状不满足时丢弃候选结果并保留旧网格。

## 允许项

- 扩展纯 `fem_agent` authoring DTO，增加 `MeshIntent`、局部加密和 mesh proposal。
- 扩展应用层 detached mesh task、原子接受事务及 Gmsh 进程协调器。
- 让现有 GUI/native 网格入口和 Agent 网格端口复用公共生成函数及协调器。
- 扩展 A1 bridge 的异步 proposal 生命周期，但继续由 GUI 控件颁发一次性授权。
- 增加 A3 专用纯契约、session/bridge 和可选真实 Gmsh 聚焦测试。
- 在本计划中记录实现和测试；顶层设计文档状态由主 Agent 更新。

## 禁止项

- 不在用户点击“开始划分”前导入、初始化或调用 Gmsh。
- 不让 Provider 获得确认、接受、拒绝、取消或 session 写入工具。
- 不持久化 Gmsh tag、OCC handle、`EntityRef`、完整 Gmsh model 或候选 FEM 数组。
- 不在严格形状失败后降级，不自动更换网格算法。
- 不提前实现 A4 作用域、材料、截面或分配能力。
- 不修改 `README.md` 或 `README_Zh.md`。
- 不使用 computer use，不运行全量测试，不提交 commit。

## 目标文件

- `src/fem_agent/mesh_authoring.py`：`MeshIntent`、局部加密和 mesh proposal。
- `src/fem_agent/__init__.py`：公开 A3 纯 DTO。
- `src/fem/geometry/gmsh_coordinator.py`：全进程单一 Gmsh 所有者。
- `src/fem/application/preprocessing.py`：显式/自动生成接入及统一协调。
- `src/fem/io/project_v9.py`：严格版本化的自动网格意图持久化；v1–v8 契约不变。
- `src/fem/application/revisions.py`、`src/fem/application/session.py`：
  detached Agent mesh task 和原子接受事务。
- `src/fem_gui/agent_authoring.py`：异步 mesh port 与 A1 bridge 生命周期接入。
- `src/fem_gui/widgets/agent_chat.py`：沿用 proposal 卡片和 GUI 授权入口。
- `tests/test_agent_authoring_phase_a3.py`：纯契约验收。
- `tests/application/test_agent_mesh_phase_a3.py`：session 原子事务和协调器。
- `tests/gui/test_agent_mesh_commit_phase_a3.py`：GUI bridge 成功、拒绝、陈旧、
  取消和失败控制流。
- `tests/integration/test_agent_mesh_gmsh_phase_a3.py`：独立真实 Gmsh 聚焦测试。
- 本计划：实施记录、验证结果和剩余风险。

具体文件可依职责做最小调整；不修改顶层边界文档。

## 实施批次

1. `MeshIntent` 严格 DTO、JSON 往返和 `MeshSettings`/`AutoMeshSpec` 转换。
2. Gmsh 进程协调器及现有 native 生成入口接入。
3. revision-bound detached Agent mesh task与 session 原子接受事务。
4. mesh proposal、异步 port、GUI 控件授权和单次刷新。
5. A3 八条验收及成功、拒绝、陈旧、取消、失败、全局串行聚焦测试。
6. 更新本计划的实施和验证记录，复核未越过 A4 边界。

## 聚焦测试

- 纯契约：显式/自动互斥、稳定孔边引用、falloff、严格形状映射、proposal 摘要及
  工具目录无确认能力。
- 应用层：确认前零 Gmsh 调用、detached 快照、运行期间状态不变、成功原子提交、
  失败/取消/陈旧保留旧网格、协调器跨线程串行并在异常后释放。
- GUI bridge：只有 GUI 控件授权可开始，拒绝和重复点击终态确定，成功只刷新一次。
- 真实 Gmsh：单独验证显式/自动三角形和四边形严格生成；缺少可选依赖时只跳过该文件。

只运行新增 A3 测试以及直接受影响的 A1/A2 bridge、native preprocessing、
session token 和 Gmsh runtime 聚焦回归。

## 完成条件

1. A3 八条验收均有自动化覆盖并通过。
2. 成功、拒绝、陈旧、取消、失败和全局串行路径终态确定。
3. Provider 工具目录不存在任何可调用确认入口。
4. 不包含 A4 能力、README 修改、全量测试或 commit。

## 实施记录

- 已增加纯 `fem_agent` `MeshIntent`。它严格要求显式 `global_size` 或
  `auto_level` 二选一，只接受 A3 的二维三角形/四边形与一阶/二阶；局部控制继续使用
  `LogicalEntityRef`、`LocalMeshControl` 和 `MeshSizeFalloff`，JSON schema 1.0
  严格往返并生成稳定 SHA-256。
- `MeshIntent` 转换为现有 `MeshSettings`；自动模式同时生成现有
  `AutoMeshSpec`。`MeshSettings` 增加 `auto_level` 和 `strict_cell_shape` 执行来源，
  旧调用默认保持显式、非严格行为。
- 当前项目格式升级为严格 schema v9。v9 在 v8 canonical authoring 上为每个非空
  Part mesh settings 增加 `intent_mode`、`auto_level` 和
  `strict_cell_shape`；v7/v8 解码器未放宽且继续拒绝这些未知字段，v1–v8 读取时得到
  明确的显式、非严格默认。
- 已增加独立于 GUI busy 状态的 `GmshExecutionCoordinator`。它是进程级、
  可重入、可取消等待的单一所有者锁；所有 `GeometryModel` 生命周期和
  `generate_fem_model` 均进入同一协调器，并在成功、异常和清理失败路径释放。
- 显式 strict 意图通过 typed point size（有局部场时通过 background field）加
  `AutoMeshSpec(level=3)`，使用内核现有严格单元类型审计，失败不转回 `MeshSpec`
  或另一算法。自动模式无局部场时保留请求 level；存在 background field 时，
  effective far-field size 已包含 level 因子，生成 spec 固定 level 3，避免局部绝对
  尺寸和 far-field size 被 `MeshSizeFactor` 二次缩放。
- `ModelSession.prepare_agent_mesh_generation` 只签发绑定
  session/mesh/model revision 的 detached 候选快照，权威 mesh intent 和旧 artifact
  不变；`accept_agent_generated_model` 完成后再次 CAS 校验，在一个 session
  revision 内同时安装 Part mesh settings 和新 artifact。候选复制/定义编译全部在
  mutation 前完成。
- 失败、取消和陈旧路径使用 `terminate_agent_mesh_task` 消费精确 token，不修改
  mesh settings、artifact 或 session revision；迟到结果无法覆盖当前模型。
- A1 bridge 继续只接受一次性 GUI 控件授权。mesh port 在授权后才构造 task，并由
  主窗口 `BackgroundTaskController` 后台运行；成功回调经 port 调用
  `accept_agent_generated_model`，随后只做一次 `_apply_session_delta` 投影。
  Provider 工具目录没有 `accept_proposal`、`confirm_mesh` 或 `confirm_solve`。
- 未增加 A4 作用域、材料、截面或分配工具，未修改 README，未使用 computer use，
  未提交 commit。

## 聚焦验证记录

- `python -m pytest tests/test_agent_authoring_phase_a3.py
  tests/application/test_agent_mesh_phase_a3.py
  tests/gui/test_agent_mesh_commit_phase_a3.py tests/io/test_project_v9.py -q`
  ：20 项通过。覆盖 A3 八条验收、成功、拒绝、陈旧、取消、失败、启动异常、
  token 消费和
  全局串行。
- `python -m pytest tests/integration/test_agent_mesh_gmsh_phase_a3.py -q`
  ：3 项通过。该文件独立使用真实 Gmsh 4.15.2，验证显式 strict quad 尺寸密度、
  Auto+孔边局部加密的 level-3 background 防二次缩放，以及 strict quad 不混入
  triangle 降级。
- `python -m pytest tests/gui/test_agent_authoring_bridge.py
  tests/gui/test_agent_geometry_commit_phase_a2.py -q`：13 项通过。
- `python -m pytest tests/io/test_project_router.py tests/io/test_project_v7.py
  tests/io/test_project_v8.py tests/io/test_project_v9.py -q`：30 项通过。
- `ruff check` 对全部 A3 新增/修改 Python 文件通过。
- 未运行全量测试。一次较宽的 Gmsh/preprocessing 组合在 120 秒限制内失败并超时；
  单独定位的首项是既有 fake `_FakeGmsh.initialize()` 不接受产品现有
  `interruptible=False` 参数；`tests/test_gmsh_session.py` 单独 35 项通过。未在
  A3 扩大范围修改该 fake。
- 主 Agent 复验纯契约、session、v8/v9 严格格式和 GUI 生命周期共 22 项通过，
  真实 Gmsh 3 项通过；主审补齐后台控制器忙或启动异常时不推进 session revision
  的边界测试。`git diff --check` 通过。

## 剩余风险

- Gmsh 原生 `mesh.generate` 仍是不可抢占调用；取消可中断协调器排队及调用前后
  checkpoint，进入原生调用后需等待它返回，再丢弃结果。
- schema v9 首轮拒绝 Boolean 结果 Part 的 AutoMesh，避免 v7 Part Boolean undo
  子记录无法无损表达新增执行来源；A3 的新普通 Part 和首个里程碑不受影响。
- 自动 level 转换为 local-control far-field effective size 使用现有
  `recipe_characteristic_size` 规则；真实 Gmsh 聚焦测试证明不发生二次缩放，但
  不把节点数展示成精确估计。

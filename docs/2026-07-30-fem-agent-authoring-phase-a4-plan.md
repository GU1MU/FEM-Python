# FEM Agent Authoring Phase A4 实施计划

## 状态

- 阶段：A4
- 状态：待主 Agent 审查
- 基线：`be541b1`
- 边界来源：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 允许项

- 在已接受且已生成网格的 native 偏心孔板上，从内置区域、
  `LogicalEntityRef` 和 `MeshEntityRef` 建立稳定作用域。
- 为固定端、加载端、孔边和板体分配符合 `{type}-{function}` 的用户可见别名。
- 使用已经确认的材料常数、二维截面假设和厚度，建立材料、截面和分配候选。
- 生成绑定精确 document、session 和 revision 的自动 `ModelPatch`，并生成精确前状态
  对应的一次性 inverse patch。
- 经 `AgentAuthoringBridge` 和公开 `ModelSession` 命令原子应用作用域及定义后状态。
- 在当前 revision 仍等于自动补丁后 revision 时，通过 GUI 撤销入口应用 inverse patch。
- 检测有效结果失效影响，并将该编辑转为 GUI 显式确认的 destructive-edit proposal。
- 复用当前 schema 保存和重开已接受作用域、材料、截面及分配。

## 禁止项

- 不实现 A5 的分析步、边界条件、载荷或结果请求。
- 不从材料名称、二维假设或厚度中推断工程数值。
- 不接受临时 Gmsh tag、ModelSession、Qt、VTK、Gmsh 或完整网格数组进入 Provider
  契约。
- 不在拓扑 lineage 不精确、作用域为空、实体类型混合或选择数量异常时继续。
- 不覆盖用户已有同名作用域或定义；破坏性修改只形成确认提案。
- 不让 Provider 或自然语言调用授权确认或撤销 GUI 操作。
- 不修改顶层 boundary 文档、README，不提交 commit，不运行全量测试，不使用
  computer use。

## 目标文件

- `src/fem/application/commands.py`
- `src/fem/application/session.py`
- `src/fem/application/__init__.py`
- `src/fem_agent/definition_authoring.py`
- `src/fem_agent/__init__.py`
- `src/fem_gui/agent_authoring.py`
- `src/fem_gui/main_window.py`
- `src/fem_gui/widgets/agent_chat.py`
- `tests/test_agent_authoring_phase_a4.py`
- `tests/gui/test_agent_definition_patch_phase_a4.py`
- 本计划文档

如当前严格项目 schema 已能无损保存全部 A4 定义，则仅增加聚焦回归测试，不新建
schema 版本。

## 实施批次

1. 建立纯 A4 作用域选择和证据 DTO；用 recipe exactness、语义逻辑引用及当前网格
   catalog 证明偏心孔板四个作用域，数量或几何证据异常时 fail closed。
2. 建立材料、截面、分配候选及安全 JSON 后状态；所有名称经统一
   `NamePolicy`/`NameAllocator`。
3. 增加公开、revision-gated 的 session 原子定义命令，一次验证并提交作用域、材料、
   截面和分配。
4. 生成自动 patch、inverse patch 和结果失效确认 proposal；在 GUI bridge/port
   建立自动应用回执、摘要、一次性撤销以及 revision 陈旧门。
5. 验证 current project schema 保存/重开、原子失败、拒绝、陈旧和结果失效路径。

## 聚焦测试

- A4 纯契约：四个稳定作用域、语义别名、逻辑/网格引用、名称策略、证据和 fail
  closed。
- application：定义批次单 revision 原子成功、异常无后状态、stale compare-and-swap。
- bridge/GUI：自动应用摘要和撤销入口；一次撤销；revision 改变后禁用/拒绝；结果
  存在时只形成确认提案；拒绝保持模型不变。
- project I/O：当前 schema 保存并重开后四个作用域、材料、截面和分配完全保留。
- 聚焦回归：现有 definition edit、native region/materialization、A1–A3 authoring
  契约；真实 Gmsh 仅在确有必要时单独执行。

## 审查门

- 已实现并通过主 Agent 审查。
- 顶层 boundary 文档状态由主 Agent 审阅后更新。

## 实现与聚焦验证记录

- 已完成偏心孔板 `LEFT`、`RIGHT`、`HOLE`、`DOMAIN` 内置语义与
  `LogicalEntityRef`/`MeshEntityRef` 双证据解析；固定端、加载端、孔边、板体均保存
  为带 Part owner 的稳定 mesh 引用，异常 identity/count、非精确拓扑和空选择均
  fail closed。
- 已完成一个 `ScopedDefinitionBatch` session 事务；作用域、材料、截面和分配先在
  detached 状态解析、materialize 和 compile，成功后只增加一个 revision，异常与
  陈旧批次保持原状态。
- 已完成自动 `ModelPatch`、精确前状态 inverse patch、幂等重放、一次性撤销和
  revision 陈旧门；自动端口另行证明补丁只增加对象，不能删除或覆盖已接受状态。
- 已完成结果存在时 destructive-edit proposal 转换；自动端口再次检查实际 accepted
  result 并 fail closed。确认、拒绝和撤销仍只存在于 GUI 边界，Provider 工具目录
  不暴露对应能力。
- 已增加真实聊天卡片摘要和“撤销本次 Agent 修改”按钮。主窗口使用保留的
  `SessionDelta` 更新模型树、inspection/result 状态和 action gate，测试证明没有
  调用 `viewport.set_model` 重建 mesh actors。
- 当前 schema v9 无需扩展，保存/回读无损保留四个作用域、材料、截面和分配。
- A4 新增聚焦测试 14 项通过；定义/作用域/materialization/A1–A3/project v9 聚焦
  回归 58 项通过；GUI bridge/定义/布局聚焦组 28 项通过，其中 A4 GUI 7 项已计入
  新增 14 项，合计 93 个不重复测试通过。
- 单独真实 Gmsh 偏心孔板检查通过，四个作用域分别解析到 6、6、7 和 155 个稳定
  mesh entity；全程未运行全量测试。
- 全局 `ruff` 对 A4 涉及文件检查通过；项目 venv 未安装 ruff，按仓库规则使用了
  全局可用命令。
- 主 Agent 独立复验 A4 新增测试 14 项通过，并将 A1–A4 纯契约及 GUI
  bridge/提交/撤销聚焦组合扩展为 48 项通过；`ruff check` 与
  `git diff --check` 均通过。

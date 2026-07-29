# FEM Agent Authoring Phase A5 实施计划

## 状态

- 阶段：A5
- 基线：`63bdf63`
- 状态：已实现，主 Agent 审查通过
- 上游边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

在 A4 已接受的作用域、材料、截面和可逆自动 patch 基础上，增加一个完整且可持久化、
可重新编译的线性静力分析定义：

- 恰好一个 `NLGEOM` 关闭的线性静力分析步；
- 具有独立、稳定、唯一名称的位移边界条件、载荷和结果请求；
- 当前内核支持的位移、节点载荷、二维边载荷或三维面载荷；
- 当前内核支持的结果请求；
- revision-bound、原子、可逆的自动 patch，以及结果失效或破坏性修改的 GUI 确认门；
- 保存、关闭、重开和重新编译后不丢失对象身份或工程含义。

## 允许项

- 增加应用层分析对象身份和有界摘要 DTO。
- 为当前项目格式增加显式新 schema，并从旧 schema 匿名对象确定性迁移。
- 让旧版本 encoder 明确拒绝无法无损表示的命名对象；旧 decoder 的字段集合保持严格。
- 复用 `ModelDefinitions`、`ScopedDefinitionBatch`、A4 bridge、patch/inverse 和最小
  `SessionDelta` 投影。
- 对已有有效结果、覆盖或删除已有对象生成破坏性编辑确认提案。
- 为 GUI 列表、编辑和删除提供名称驱动的稳定身份，同时兼容匿名旧对象。
- 运行直接 DTO、项目 codec、application/session 和聚焦 GUI 测试。

## 禁止项

- 不实现 A6 的预检提案、求解摘要、求解确认或求解执行。
- 不实现 A7 的结果查询、结果聚合或工程解释。
- 不从“固定”等模糊文字推断未知维度下的全部自由度。
- 不猜测自由度、载荷方向、符号、数值、单位、分布或结果变量。
- 不覆盖、删除已有定义或使有效结果失效后自动应用。
- 不放宽 schema 1–9 decoder，不向旧 schema 静默加入新字段。
- 不修改顶层 boundary 文档或 README。
- 不运行全量测试，不使用 computer use。

## 契约与版本隔离

1. 分析对象名称由统一 `NamePolicy`/`NameAllocator` 分配，Agent 新建对象严格满足
   `{type}-{function}`；同类命名空间内进行 Unicode/大小写规范化后的唯一性检查。
2. 名称是独立于作用域、步骤和列表位置的应用层身份。匿名兼容对象获得按步骤、
   类型和原顺序生成的确定性显示名；新格式保存时写入显式名称。
3. 新 schema 对每个边界、载荷和结果请求要求精确的 `name` 字段。解码时先验证新
   字段，再通过旧 codec 解码旧字段；旧 codec 本身仍拒绝未知字段。
4. A5 patch 只表达精确后状态，包含完整分析定义和安全摘要；inverse 保存精确前状态。
5. 自动应用只允许纯新增且当前无有效结果。覆盖、删除、过程修改或结果失效必须成为
   `DESTRUCTIVE_EDIT` 提案，由 GUI 控件确认。

## 工程字段门

每个分析对象 DTO 必须显式包含并验证：

- 所属分析步；
- 对象名称；
- 目标作用域和实体类型；
- 自由度或载荷/结果类型；
- 分量、方向和符号；
- 数值与单位；
- 分布方式；
- 结果变量及目标位置（结果请求）。

缺失、非有限数值、维度不明确、目标实体不匹配、未经确认或超出当前内核能力时，
构造阶段确定性 fail closed，GUI 和 session 均不变化。

## 目标文件

- `src/fem/core/model.py`
  - 增加向后兼容的独立可选名称字段。
- `src/fem/application/analysis_authoring.py`（新增）
  - 稳定身份、匿名迁移、名称解析和分析对象辅助契约。
- `src/fem/application/definitions.py`
  - 在 detached normalize/compile 中验证分析对象身份和单步线性静力约束。
- `src/fem/io/project_v10.py`（新增）、`src/fem/io/project.py`、`src/fem/io/__init__.py`
  - 严格 schema v10、旧项目匿名迁移和当前 writer 路由。
- `src/fem/io/_project_codec.py`
  - 旧 encoder 对命名对象进行显式无损能力拒绝；decoder 字段集合不放宽。
- `src/fem_agent/analysis_authoring.py`（新增）、`src/fem_agent/__init__.py`
  - confirmed DTO、完整静力定义 patch/proposal、摘要和 inverse 后状态编码。
- `src/fem_agent/definition_authoring.py`
  - 复用/扩展精确定义后状态 operation 解码，不改变 A4 行为。
- `src/fem_gui/agent_authoring.py`
  - 复用 A4 bridge 执行 A5 自动 patch、确认提案和 inverse。
- `src/fem_gui/analysis_definition_dialogs.py`、必要时 `src/fem_gui/main_window.py`
  - 按稳定名称显示和解析 GUI 编辑/删除身份，保持最小投影。
- `tests/test_agent_authoring_phase_a5.py`（新增）
- `tests/io/test_project_v10.py`（新增）
- `tests/gui/test_agent_analysis_patch_phase_a5.py`（新增）

最终实现若不需要其中某个现有文件，不为满足清单而制造空改动。

## 实施批次

### 批次 1：身份与 schema

- 增加可选名称字段，不破坏旧构造器的位置参数。
- 增加统一分析对象身份辅助函数和确定性匿名迁移。
- 实现严格 schema v10；验证 schema 9 拒绝 v10 字段，v10 严格拒绝未知/缺失字段。
- 覆盖匿名旧项目打开以及新格式保存/重开名称保持。

### 批次 2：严格分析 DTO 与完整步骤

- 建立明确的位移、载荷和结果请求 DTO。
- 验证维度、实体类型、自由度/分量、方向、符号、数值、单位和分布方式。
- 只编译恰好一个 `procedure=static`、`NLGEOM` 关闭的步骤。
- 将 DTO 转为当前内核支持的 `AnalysisStep`、约束、载荷和 `OutputRequest`。

### 批次 3：原子 patch、确认门与 GUI 投影

- 生成完整定义精确后状态 operation。
- 自动应用只允许新增、无结果、无覆盖；生成一次性精确 inverse。
- 破坏性修改或结果失效进入 GUI 确认提案。
- 成功只投影最小分析定义树/符号；拒绝、陈旧、异常和撤销保持原子。
- GUI 编辑和删除优先使用稳定名称身份，不依赖对象列表位置。

### 批次 4：聚焦验收

- 唯一命名和模糊/缺项 fail closed。
- 匿名迁移、schema 严格性、保存/重开/重新编译完整性。
- patch 成功、重放、拒绝、陈旧、异常、撤销和结果失效确认。
- 完整偏心孔板静力定义 compile 通过。

## 聚焦测试

计划运行：

- `tests/test_agent_authoring_phase_a5.py`
- `tests/io/test_project_v10.py`
- `tests/gui/test_agent_analysis_patch_phase_a5.py`
- A4 patch/定义回归：
  - `tests/test_agent_authoring_phase_a4.py`
  - `tests/gui/test_agent_definition_patch_phase_a4.py`
- project/session 聚焦回归：
  - `tests/io/test_project_v9.py`
  - `tests/application/test_application_model_definitions.py`
  - `tests/application/test_definition_edit_batches.py`
  - `tests/application/test_session_authoring_projection.py`
- GUI 分析定义身份聚焦回归：
  - `tests/gui/test_analysis_definitions.py` 中相关选择、编辑和删除用例。

不运行全量测试；如果聚焦回归暴露既有失败，记录其基线证据并交由主 Agent 决定是否
扩展范围。

## 主审交付

- 精确文件清单和 schema 迁移说明；
- 所有聚焦测试命令、通过数和失败证据；
- 未覆盖能力和残余风险；
- 本计划状态更新为“实现完成，等待主 Agent 审查”；
- 不创建提交。

## 实施结果

- 分析对象增加独立可选 `name`，覆盖位移、节点/边/面/线/体/重力载荷和结果请求。
- 新增应用层确定性匿名身份迁移；显式名称按边界、载荷、结果请求命名空间做
  Unicode/大小写规范化后的唯一性检查。
- 新增严格 schema v10。v10 的每个分析对象要求精确 `name` 字段；schema 1–9
  decoder 未放宽，旧 encoder 对命名对象明确拒绝无损表示。
- 当前项目路由把匿名旧对象迁移为确定性兼容名称；当前 writer 保存 schema v10。
- 新增严格 A5 DTO 和单步线性静力构造：
  - `procedure` 精确为 `static`；
  - `metadata.nlgeom` 必须显式为布尔 `False`；
  - 位移自由度、节点分量、二维边/三维面作用域和载荷方向/符号均 fail closed；
  - pressure 使用“内向为正、外向为负”的内核符号规范；
  - 位移/节点/边/面载荷和 U/RF/S 单位与项目 `UnitContext` 精确匹配，未实现单位换算。
- A5 自动 patch 只允许在无步骤、无有效结果且不修改 A4 对象时增加一个完整步骤；
  已有步骤、覆盖/删除或结果失效转为 `DESTRUCTIVE_EDIT` GUI 确认提案。
- 复用 A4 精确后状态、一次性 inverse、revision 门和最小定义投影；A4 legacy
  operation 缺少 `steps` 时精确保留当前步骤，未知字段仍拒绝。
- 模型树对命名分析对象使用 `(step_name, object_name)` 稳定编辑/删除 key，并保留
  匿名对象的旧索引兼容；管理器显示独立名称与目标作用域。

## 验证记录

- A5 新增测试：
  - `tests/test_agent_authoring_phase_a5.py`
  - `tests/io/test_project_v10.py`
  - `tests/gui/test_agent_analysis_patch_phase_a5.py`
  - 结果：`15 passed`
- A4/A5、bridge、v9/v10、application definitions/session 聚焦组合：
  - 结果：`81 passed`
- 旧项目 codec 聚焦回归：
  - `tests/io/test_project_codec.py`
  - `tests/io/test_project_v2.py`
  - 结果：`133 passed`
- GUI 分析定义与模型树附加回归：
  - `42 passed, 1 failed`
  - 唯一失败
    `test_main_window_filters_distributed_load_regions_by_model_dimension`
    使用无 Part owner 的旧 `LogicalEntityRef`；`git diff 63bdf63 --`
    已确认对应 session 和测试文件均未由 A5 修改，基线同一严格检查会拒绝该输入。
    未扩展 A5 范围修改这一既有 characterization。
- `git diff --check` 通过。

## 主 Agent 审查

- 审查结论：通过。
- 独立复跑 A5 新增 DTO、schema v10 和 GUI 身份测试：`15 passed`。
- 独立复跑 A1–A5 契约、application、GUI bridge 和 v9/v10 聚焦组合：
  `72 passed, 1 failed`。
- 唯一失败
  `test_definition_serialisation_shape_remains_compatible`
  仍断言旧版 `NativePart` 的 `asdict` 字段形状；对比基线 `63bdf63`
  已确认 A5 开始前同一实现与测试组合即会失败，因此不计为 A5 回归。
- 全局 Ruff 对 A5 新增及核心修改文件检查通过；`git diff --check` 通过。
- 审查确认 schema 1–9 decoder 保持严格、旧 A4 operation 可回放、pressure
  方向与符号可持久化、单位绑定 `UnitContext`，并且 NLGEOM 只接受显式布尔
  `False`。
- 未运行全量测试，未使用 computer use。

## 残余风险

- 结果请求单位没有复制进 `OutputRequest.metadata`；它们由随项目持久化的
  `UnitContext` 和受控变量 U/RF/S 确定性恢复，A5 patch 摘要仍保存显式单位。
- GUI 手工编辑一个覆盖多个自由度的命名位移时，既有对话框会拆成逐分量对象；
  A5 为拆分对象保留原名称并生成确定性短后缀，但通用 GUI 全局 undo/redo 仍不在
  本阶段范围。
- A5 Agent 工具只发布当前里程碑需要的节点、二维边和三维面载荷；线载荷、体力与
  重力已具备名称、迁移和 GUI 身份兼容，未作为 A5 Agent 新建 DTO 发布。

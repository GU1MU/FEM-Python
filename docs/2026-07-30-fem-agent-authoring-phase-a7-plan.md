# FEM Agent Authoring Phase A7 实施计划

## 状态

- 阶段：A7
- 基线：`e47bb95`
- 状态：已实现，主 Agent 审查通过
- 上游边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

在 A6 已由 GUI 接受并安装的 native 求解结果上，建立严格、只读、有界的 Agent
结果查询链：

- 复用 application 层原生 `ResultProvider`、类型化结果目录和精确查询 API；
- 支持位移 `U`、反力 `RF` 和应力 `S` 的受控变量、分量、位置、区域和聚合；
- 支持最大值、最小值、绝对值极值，以及能力允许时的反力合计；
- 首个里程碑能够读取最大位移、应力极值和固定端反力合计；
- 每个标量携带数值、单位、位置或区域身份，以及精确
  run/step/result source/materialization generation；
- Agent 解释只格式化本地查询返回的真实标量，不估算或补充数值。

## 允许项

- 在 `fem_agent` 增加严格、不可变、JSON 安全的查询请求、标量、诊断和响应 DTO。
- 在 `fem_agent` 定义只读查询端口协议、Fake Port、查询 bridge 和确定性简洁解释器。
- 在 `fem_gui` 实现该协议，将 Agent 请求映射到当前已接受的
  `ResultProvider.catalog()`、`validate_query()` 和 `query()`。
- 在本地 GUI 适配器内把 named region 解析为当前结果模型的节点或单元身份，再用
  原生 `ResultQuery` 过滤。
- 在本地 GUI 适配器内聚合原生查询记录，只把有界标量和身份摘要返回给 Agent 层。
- 从随 native 项目持久化的 `UnitContext` 恢复 U/RF/S 的长度、力和应力单位。
- 增加纯 DTO/Fake Port 测试和不启动真实视口的 GUI 适配器聚焦测试。

## 禁止项

- 不把完整位移、反力、应力数组或原生 `ResultQueryResult.records` 传给 Agent
  Provider。
- 不传递完整节点坐标、完整连接、VTK 数据、绝对路径、Qt/VTK/GUI 对象、
  `ModelSession` 或原始 `ResultProvider`。
- 不按需物化 LAZY 字段；A7 只查询当前 generation 已经 READY 的已接受字段。
- 不通过查询改变结果字段、分量、相机、选择、变形比例或当前显示结果。
- 不实现“显示结果”命令、结果导出、任意 Python/Shell 或 A8 端到端流程。
- 不接受省略变量、分量、位置、区域或聚合的模糊请求，也不选择默认工程含义。
- 不修改 README 或顶层 boundary 文档，不运行全量测试，不使用 computer use，
  不创建 commit。

## 查询 schema

### 请求

`AgentResultQuery` 使用严格 schema `1.0`，字段集合固定为：

- `variable`：`U`、`RF`、`S`；
- `component`：必须精确匹配当前结果目录发布的分量；
- `position`：必须精确匹配原生 `FieldPosition`；
- `region`：`all_nodes`、`all_elements` 或明确 named region；
- `aggregation`：`maximum`、`minimum`、`absolute_extreme`、`sum`；
- `expected_source`：完整 result/session/artifact/model/step/run 身份；
- `expected_materialization_generation`：非负整数。

`sum` 首轮只允许 `RF`，避免对位移或应力给出没有受控工程意义的合计。单次 bridge
调用只返回一个标量；多个工程量由多个明确请求表达。

`read_accepted_result_catalog` 使用无参数严格工具 schema，返回当前 source、
materialization generation、可用 variable/position/component/unit，以及有界的 nodal
和 element named-region 名称。Agent 必须先读取该目录，不能猜测 source、generation
或分量。

### 响应

`AgentResultScalar` 只包含：

- 查询变量、分量、位置、区域和聚合；
- 有限标量及单位；
- 有界位置身份：可选 node、element、integration point、local node 和 region
  signature，不包含坐标；
- 精确 source 六元身份与 materialization generation。

失败响应不包含数值，只包含稳定 code、阶段、有界消息、是否可重试和是否需要澄清。

## 身份与陈旧门

1. GUI 适配器从 `ModelSession.current_result_provider()` 取得当前已接受 provider，
   同时读取 `current_result_identity()`。
2. 调用开始时，当前 source、provider snapshot generation、请求中的 expected source
   和 expected generation 必须完全一致。
3. 查询只使用当前目录中唯一、精确匹配 variable/position 的 READY field；LAZY、
   UNAVAILABLE、重复或缺失均 fail closed。
4. named region 从同一 provider 所有的 accepted result model 中解析为节点或单元
   identity；类型不适用于变量时拒绝。
5. 原生 `provider.query()` 的结果必须匹配精确 query、source 和 generation。
6. 聚合完成后再次读取 Session 当前结果身份；source 或 generation 任一变化即丢弃
   标量并返回 stale。

结果切换、结果物化 generation 推进、模型/作业切换都会使旧请求确定性陈旧。

## 单位策略

- `U` 使用当前 native `UnitContext.length`；
- `RF` 使用 `UnitContext.force`；
- `S` 使用 `UnitContext.stress`。

A7 不转换数值，不信任结果目录的可选展示单位，也不从命名或数值猜测单位。native
项目缺少单位上下文时拒绝数值响应。

## 目标文件

- `src/fem_agent/result_authoring.py`（新增）
  - 严格请求/响应/诊断/身份 DTO、只读端口协议、Fake Port、bridge 和解释器。
- `src/fem_agent/__init__.py`
  - 导出 A7 纯 Python 契约。
- `src/fem_gui/agent_authoring.py`
  - Session 当前结果到原生 provider 的只读适配器、named region 解析和本地聚合。
- `src/fem/application/session.py`
  - 让既有 GUI shallow projection 保留已持久化单位上下文，供只读查询恢复单位。
- `src/fem/application/results/provider.py`
  - 增加仅供本地消费者使用的精确 named region 节点/单元身份解析 API。
- `src/fem_gui/main_window.py`
  - 持有 A7 bridge；不增加显示命令或视口副作用。
- `tests/test_agent_authoring_phase_a7.py`（新增）
  - 严格 DTO、Fake Port、无结果、歧义、陈旧、payload 和解释边界。
- `tests/gui/test_agent_result_query_phase_a7.py`（新增）
  - 原生 provider 三类成功查询、named region、切换/推进陈旧和视口零变更。

最终实现按现有职责做最小调整，不为满足清单制造空改动。

## 实施批次

1. 建立严格 A7 DTO、端口协议、Fake Port、bridge、工具 schema 和解释器。
2. 实现 Session `ResultProvider` 适配器、目录解析、区域解析、单位绑定和本地聚合。
3. 将 bridge 接入主窗口对象，不接入任何视口显示路径。
4. 覆盖成功、无结果、歧义、不支持、越界、陈旧、结果切换、generation 推进、
   payload 有界和查询后视口零变更。

## 聚焦测试

计划运行：

- `tests/test_agent_authoring_phase_a7.py`
- `tests/gui/test_agent_result_query_phase_a7.py`
- 直接依赖的原生结果查询回归：
  - `tests/application/results/test_query_contracts.py`
  - `tests/application/results/test_query_evaluator.py`
  - `tests/application/results/test_provider_primary.py`
- A6 bridge 聚焦回归：
  - `tests/test_agent_authoring_phase_a6.py`
  - `tests/gui/test_agent_solve_phase_a6.py`

另运行 A7 目标 Python 文件的 Ruff、编译检查和 `git diff --check`。不运行全量测试。

## 主审交付

- 精确实现文件、协议和身份门说明；
- 聚焦测试命令、通过数和失败证据；
- viewport 零变更与 payload 不含路径/数组的测试证据；
- 未覆盖能力和残余风险；
- 本计划状态更新为“实现完成，等待主 Agent 审查”；
- 不创建提交。

## 实施结果

- 新增严格 A7 目录和查询契约：
  - `read_accepted_result_catalog` 返回精确 source、generation、READY 的
    U/RF/S variable/position/component/unit，以及有界 named-region 名称；
  - `query_accepted_result` 要求显式 variable、component、position、region、
    aggregation、expected source 和 expected generation；
  - `AgentResultQueryBridge`、只读端口协议和 Fake Port 均只跨层传递不可变 DTO。
- 主窗口持有 A7 bridge，但没有增加结果显示命令、确认入口或视口投影调用。
- GUI Session 适配器复用原生
  `ResultProvider.catalog()/validate_query()/query()`；只查询当前 generation 的 READY
  字段，不调用 `materialize()`，也不复制结果恢复算法。
- `ResultProvider` 增加本地 named-region 节点/单元解析 API：
  - node set、edge、surface 和 element set 均按首次出现顺序去重；
  - 多个 nodal collection 同名、空集合、未知名称和错误实体类型均用稳定 code
    fail closed；
  - 解析出的 ID 只在本地构造原生 `ResultQuery`，不进入 Agent DTO。
- 结果目录只发布 Session `named_regions` 中前 127 个有界名称以及
  `all_nodes/all_elements`；查询不能枚举或猜测内部模型集合。
- 位移、反力和应力分别使用持久化 `UnitContext.length/force/stress`，不转换数值；
  GUI shallow projection 补齐了已有单位上下文，避免查询从其他字段猜测单位。
- maximum/minimum/absolute-extreme 从原生 records 确定性选择位置；
  absolute-extreme 返回原始带符号值。RF sum 使用去重后的节点过滤并以
  `math.fsum` 合计，每个节点最多计一次。
- 查询开始、原生返回和聚合完成后均核对 source/generation；结果切换、
  materialization generation 推进或调用末尾身份改变均返回 stale，不返回标量。
- Fake Port 解释器只格式化响应中的真实 value/unit/location/run/step；失败响应没有
  工程标量。

## 验证记录

- A7 新增聚焦测试：
  - `tests/test_agent_authoring_phase_a7.py`
  - `tests/gui/test_agent_result_query_phase_a7.py`
  - 结果：`18 passed`。
- A7、原生 query contracts/evaluator/provider、A6 DTO/GUI solve 和 A1 bridge 聚焦
  组合：`83 passed`。
- 额外运行 solve result workflow、result materialization 和既有 typed main-window
  query 组合：`20 passed, 9 failed`。9 个失败都位于
  `tests/gui/test_typed_main_window_result_query.py`；该旧 fixture 没有
  `OutputRequest`，accepted provider 目录为空，因此仍断言非空
  `default_selection/result_selection` 时失败。`git diff e47bb95 --` 已确认 A7
  没有修改该测试或 result workflow/execution/output-request 路径，A7 的 provider
  改动只增加 named-region 公共解析方法，故未扩大范围修改该既有
  characterization。
- A7 目标文件 `compileall` 通过；临时 pycache 已清理。
- 全局 Ruff 对全部 A7 新增/修改 Python 文件通过；项目 venv 未安装 Ruff，按项目
  规则使用全局可用命令。
- `git diff --check` 通过。
- 未运行全量测试，未使用 computer use，未修改 README 或顶层 boundary 文档，
  未创建 commit。

## 残余风险

- A7 已建立模型可调用的严格 tool schema、bridge 和主窗口本地端口；当前
  `QtAgentRuntime` 的动态工具自动发现/注册仍由 A8 端到端接线完成。本阶段不声称
  生产 Provider 已自动发现这两个工具。
- A7 只查询当前 accepted generation 的 READY 字段。未请求或尚未物化的结果返回
  诊断；A7 不以只读查询名义推进 materialization generation。
- 极值并列时保留原生 record 顺序中的第一个位置；标量仍为真实极值，A7 不返回
  无界的全部并列位置。
- `ResultProvider` 的 named-region ID API 可能在本地返回较大 identity tuple，但
  它不跨 `fem_agent` 协议边界；Agent 只看到名称、一个聚合标量和一个有界位置。

## 主 Agent 审查

- 审查结论：通过。
- 独立复跑 A7 结果目录、DTO/Fake bridge 和 GUI/native 查询测试：`18 passed`。
- 独立复跑原生 query contracts/evaluator/provider、A6 DTO/GUI solve 和 A1 bridge
  直接依赖回归：`65 passed`。
- 审查确认 tool catalog 和 query schema 均为封闭字段集；查询必须从目录取得精确
  source、generation、READY field、component、unit 和有界 region 名称。
- 审查确认 source/generation 在开始、原生查询返回和聚合完成后三次核对；结果切换
  或 generation 推进均不返回旧标量。
- 审查确认 named-region 解析位于 application `ResultProvider` 公共只读 API，
  node/edge/surface/element set 均去重，GUI 不访问 provider 私有模型。
- 审查确认 RF 使用去重节点和 `math.fsum`，S absolute-extreme 保留原始带符号值与
  位置，U/RF/S 单位精确来自持久化 `UnitContext`。
- 查询后视口字段、分量、相机、选择和变形状态零变化；payload 不含 records、数组、
  坐标、连接、VTK 或绝对路径。
- 全局 Ruff 与 `git diff --check` 通过；未运行全量测试，未使用 computer use。

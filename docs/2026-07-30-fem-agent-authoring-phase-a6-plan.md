# FEM Agent Authoring Phase A6 实施计划

## 状态

- 阶段：A6
- 基线：`5831667`
- 状态：已实现，主 Agent 审查通过
- 上游边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

在 A5 已接受、已网格化且具有完整单步线性静力定义的 native 模型上，完成从确定性
模型预检到 GUI 明示确认、后台求解和精确结果归属的本地控制链：

- 自动调用既有类型化静力预检，不需要求解确认；
- 只有不存在阻塞诊断时生成有界求解摘要和 `SOLVE` 提案；
- 提案同时绑定精确 document、session revision、artifact/model revision、
  validation stamp、分析步、作业名和 proposal hash；
- “开始求解”只能由当前 GUI 提案卡控件签发一次性授权；
- 一次点击只向既有 GUI job/session 后台入口提交一个作业；
- 复用既有进度、合作取消、成功、失败和陈旧结果丢弃路径；
- 成功结果继续由 application session 绑定精确 artifact、run 和 model revision。

## 允许项

- 增加纯 Python 的 validation stamp、有界预检诊断、求解摘要和求解提案 DTO。
- 从当前 `ValidationRecord` 的精确身份和有界确定性报告字段计算 SHA-256 指纹。
- 复用 `AgentProposal`、`REQUEST_SOLVE`、proposal hash/idempotency 和 V1 生命周期。
- 让 `SessionGeometryAuthoringPort` 接受严格的 `SOLVE` 提案，并在 GUI 授权消费后
  调用注入的现有 GUI 作业启动回调。
- 为既有 `_begin_submit_run` 增加精确 expected session revision 和生命周期回调，
  保持普通 GUI 作业入口行为兼容。
- 使用直接 DTO、Fake 启动回调、application session 和聚焦 Qt 测试。

## 禁止项

- 不把 `confirm_solve`、`accept_proposal` 或其他确认能力注册为 Provider 工具。
- 不接受自然语言、Provider 工具调用、旧事件、重放日志或后台线程作为求解授权。
- 不在阻塞诊断、缺失单位、缺失当前 validation、陈旧 session/artifact/model/stamp
  或 GUI 后台控制器忙时提交求解。
- 不把完整模型、节点、单元、结果数组、VTK、绝对路径或原始 ModelPatch 放进
  Provider payload、提案卡或事件。
- 不在 Qt 主线程执行预检、装配、求解或结果物化。
- 不修改或复制求解算法，不建立第二个后台 task controller。
- 不实现 A7 的结果查询、聚合或工程解释。
- 不实现 A8 的自然语言端到端流程。
- 不修改 README 或顶层 boundary 文档，不运行全量测试，不使用 computer use，
  不创建 commit。

## Validation stamp 与失效规则

1. `SolveValidationStamp` 包含 session ID、artifact ID、model revision、step name 和
   `report_hash`；自身再计算一个完整 `stamp_hash`。
2. `report_hash` 只使用预检报告中有界、确定性的字段：精确 provenance、是否执行
   数值稳定性检查、固定字段的 `PreflightFacts`，以及按报告顺序保存的诊断
   code/severity/stage/message/path/remediation。它不包含时间、绝对路径、模型数组、
   任意 GUI 对象或无界 details。
3. 提案注册、按钮可用性和 GUI 点击消费时都重新从当前 session validation record
   计算 stamp。任一 document/session revision、artifact、model revision、step 或
   stamp 不一致均 fail closed。
4. 预检重跑会产生新的 session revision；即使确定性报告内容相同，旧提案仍因
   base session revision 不一致而陈旧。报告内容变化时 `stamp_hash` 也变化。

## GUI 唯一授权与后台复用

1. Provider 只能请求本地生成求解提案，不能获得接受、拒绝、取消或确认工具。
2. `AgentAuthoringBridge.accept_from_gui_control` 继续在 bridge owner/GUI 线程创建并
   立即消费不可伪造的一次性授权；直接调用 `accept_proposal` 缺少该对象时拒绝。
3. solve port 在消费 GUI 授权后最后一次核对 revision 和 validation stamp，然后将
   严格的 `AgentSolveTaskRequest` 交给主窗口。
4. 主窗口调用既有 `_begin_submit_run`。该入口仍使用 session
   `prepare_solve -> begin_run -> accept_run_succeeded/failed/cancelled` 和同一个
   `BackgroundTaskController`，因此不增加 Qt 主线程求解或第二套结果安装逻辑。
5. proposal 生命周期随现有后台终态进入 running/succeeded/failed/cancelled/stale；
   重复点击和晚到回调由 proposal 状态、task token 和 session revision 三层拒绝。

## 目标文件

- `src/fem_agent/solve_authoring.py`（新增）
  - 有界预检视图、确定性 validation stamp、求解摘要和严格 `SOLVE` 提案构造。
- `src/fem_agent/__init__.py`
  - 导出 A6 纯契约。
- `src/fem_gui/agent_authoring.py`
  - 严格 solve proposal 端口、stamp/revision 门、一次性启动和生命周期。
- `src/fem_gui/main_window.py`
  - 注入 Agent solve 启动回调并复用既有作业后台入口。
- `src/fem_gui/widgets/agent_chat.py`
  - 提案按钮同时检查当前 validation stamp；保持 GUI 控件为唯一授权。
- `tests/test_agent_authoring_phase_a6.py`（新增）
  - DTO、阻塞诊断、stamp、隐私和工具目录边界。
- `tests/gui/test_agent_solve_phase_a6.py`（新增）
  - GUI 授权、revision/stamp 陈旧、重复点击和 Fake 后台终态。
- `tests/gui/test_agent_solve_main_window_phase_a6.py`（如确有必要新增）
  - 既有 job/session 后台入口复用、取消、失败和晚到结果。

最终实现按现有职责保持最小修改；不为满足清单制造空改动。

## 实施批次

### 批次 1：确定性预检视图和 stamp

- 建立固定字段、有界、严格类型的诊断/事实/单位/求解摘要。
- 从 current `ValidationRecord` 计算稳定 hash，并验证阻塞诊断无法生成提案。
- 使用现有 `run_static_preflight`/GUI `check_step`，不复制 FEM 检查算法。

### 批次 2：严格求解提案

- 生成单一 `REQUEST_SOLVE` operation，包含 step、job、artifact/model revision 和
  validation stamp。
- 本地摘要显示模型、分析步、作业、节点/单元/单元类型、材料、截面、约束、载荷、
  结果请求、单位、警告和阻塞诊断计数。
- 严格拒绝未知字段、非当前 stamp、缺失单位和非 native 模型。

### 批次 3：GUI 授权和既有后台生命周期

- port 只接受一个严格 solve operation。
- GUI 点击后只启动一次；重复、重放、busy 和启动异常有确定终态。
- 复用现有 job/session task，映射进度、成功、失败、取消和 stale/discarded。
- session token 继续阻止取消后、失败后或 model revision 变化后的晚到结果安装。

### 批次 4：聚焦验收

- 成功、阻塞诊断、陈旧 revision、陈旧 stamp、重复确认、启动异常、取消、失败和
  晚到结果。
- 求解工作负载不在 Qt 主线程，成功 provenance 精确。
- Provider 工具目录没有确认能力；有界摘要不含完整数组、绝对路径或 GUI 对象。

## 聚焦测试

计划运行：

- `tests/test_agent_authoring_phase_a6.py`
- `tests/gui/test_agent_solve_phase_a6.py`
- application 预检/stamp/task token 回归：
  - `tests/application/test_preflight.py`
  - `tests/application/test_validation_stamps.py`
  - `tests/application/test_task_tokens.py`
- GUI job/task/bridge 回归：
  - `tests/gui/test_analysis_jobs.py`
  - `tests/gui/test_task_lifecycle.py`
  - `tests/gui/test_agent_authoring_bridge.py`
- A5 直接依赖回归：
  - `tests/test_agent_authoring_phase_a5.py`
  - `tests/gui/test_agent_analysis_patch_phase_a5.py`

不运行全量测试；真实求解如需验证，使用最小线性静力 fixture 单独运行。

## 主审交付

- 精确文件和契约清单；
- 聚焦测试、Ruff 和 `git diff --check` 结果；
- 取消、失败、晚到和 provenance 的覆盖证据；
- Provider 工具目录确认边界证据；
- 残余风险；
- 本计划状态更新为“实现完成，等待主 Agent 审查”；
- 不创建提交。

## 实施结果

- 新增确定性的 `SolveValidationStamp`、有界 `SolveSummary` 和严格
  `REQUEST_SOLVE` 提案构造；阻塞诊断、缺失当前 validation、缺失单位或非 native
  当前模型均无法生成提案。
- `REQUEST_SOLVE` 保留 A1 两字段旧格式的严格反序列化，同时新增 A6 五字段严格
  身份；两种格式都拒绝混合、缺失和未知字段。
- 自动预检复用 GUI 现有后台任务控制器和 application validation token，不签发
  GUI 确认能力；通过后从当前 session 记录生成精确 stamp。
- 求解接受路径只允许 GUI 控件入口，点击前再次核对 session revision、artifact、
  model revision 和 validation stamp；一个提案只启动一个既有 GUI 作业。
- 求解继续复用 session 的 run/task token，成功结果通过 application session 的
  当前 run provenance 查询验证精确归属；取消、失败、陈旧和晚到回调不能覆盖
  当前结果。
- 未实现 A7/A8，未修改 README 或顶层 boundary 文档，未运行全量测试，未使用
  computer use，未创建提交。

## 验收记录

- A6 新增聚焦测试：
  `pytest -q tests/test_agent_authoring_phase_a6.py tests/gui/test_agent_solve_phase_a6.py`
  → `14 passed`。
- validation stamp、task token、authoring bridge、analysis job 和 task lifecycle
  核心回归 → `68 passed`。
- A5、application preflight 和陈旧 solve callback 直接依赖回归 → `36 passed`。
- 目标文件 `compileall` 通过。
- 全局 `ruff check` 通过；项目虚拟环境未安装 Ruff，按项目约定使用全局命令。
- `git diff --check` 通过。

## 残余风险

- 自动预检和求解仍依赖 GUI owner thread 调度入口；本阶段没有把这些入口注册为
  Provider 可直接调用的确认工具。
- 大模型数值稳定性检查继续遵循既有 application preflight 的预算与降级规则；
  A6 没有复制或放宽该算法。

## 主 Agent 审查

- 审查结论：通过。
- 独立复跑 A6 新增纯契约与 GUI 后台链路测试：`14 passed`。
- 独立复跑 A5、application preflight/validation/task token、GUI authoring bridge、
  analysis job 和 task lifecycle 直接依赖回归：`103 passed`。
- 审查确认自动预检无需确认且只更新 validation/status；求解提案同时绑定 session
  revision、artifact、model revision 和确定性 validation stamp。
- 审查确认通用 `REQUEST_SOLVE` 仍严格接受 A1 两字段旧格式，A6 port 仅接受五字段
  新格式，混合字段集拒绝。
- 审查确认求解仅由 GUI 控件一次性授权，Provider 工具目录无确认工具，busy、启动
  失败、取消、失败、discarded 和晚到结果均进入确定终态。
- 成功终态使用 application session 的权威 run provenance 核对 artifact、run、
  step 和 model revision，不依赖当前视口是否显示结果。
- 全局 Ruff 与 `git diff --check` 通过；未运行全量测试，未使用 computer use。

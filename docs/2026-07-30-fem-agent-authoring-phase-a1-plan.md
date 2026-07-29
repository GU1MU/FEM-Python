# FEM Agent V1 Authoring Phase A1 实施计划

## 状态

- 日期：2026-07-30
- 阶段：A1
- 状态：实现、聚焦测试与主 Agent 审查已完成
- 上游边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`
- 完成条件：本文的聚焦测试通过，并由主 Agent 审阅实现与更新上游状态
- 主审结论：通过；未发现真实 `ModelSession` 写入或 A2 能力前置实现
- 主审验证：A1 契约/bridge/事件/聊天 37 项通过，runtime/主窗口布局 23 项通过

## 目标

A1 只建立自主建模的本地控制面，不写入真实工程模型：

1. 从 GUI 的只读 session snapshot 生成有界 `AuthoringContext`。
2. 建立空白/native 文档的本地绑定和绑定失效。
3. 建立 `RequirementLedger`、`RequirementReview` 和确定性阶段门。
4. 建立具有文档身份、基准 revision 和 `draft_revision` 的 `AgentDraft`。
5. 建立严格的 `ModelPatch`、`AgentProposal`、完整 hash 和 idempotency key。
6. 建立不拥有 `ModelSession` 的 Fake `AuthoringPort` 和
   `AgentAuthoringBridge` 骨架。
7. 增加严格 proposal 事件、事件重放和最小聊天卡片/绑定状态展示。

## 允许项

- 读取空白或 native session 的有界摘要。
- 在 Agent 本地状态中记录需求、审查、草稿和无副作用提案。
- 使用 Fake Port 展示静态提案并记录接受、拒绝、陈旧、失败和重放结果。
- GUI 控件调用本地 bridge 的授权入口。
- 文档/session/revision 改变后立即陈旧化待确认提案。
- 为 A1 纯契约和最小 Qt 控件接入增加聚焦测试。

## 禁止项

- 修改 `ModelSession` 或增加任何真实 session 写入口。
- 创建、替换或删除真实几何、网格、作用域、定义、作业或结果。
- 调用 Gmsh、求解器、VTK 投影或主窗口私有建模方法。
- 向 Provider 发送完整数组、绝对路径、GUI/Qt/VTK 对象或原始 patch。
- 注册 Provider 可调用的确认工具，或用自然语言/工具调用授权。
- 提前实现 A2 的单位持久化、命名分配、recipe 提交和项目事务。
- 修改 README、运行全量测试或提交 git commit。

## 目标文件

- `src/fem_agent/authoring.py`
  - 有界 DTO、需求账本、审查、草稿、patch/proposal、Fake Port。
- `src/fem_agent/__init__.py`
  - A1 公共纯 Python 契约导出。
- `src/fem_gui/agent_authoring.py`
  - session snapshot 到有界上下文的适配及 bridge 骨架。
- `src/fem_gui/agent_events.py`
  - 严格 proposal 生命周期事件和可重放投影。
- `src/fem_gui/widgets/agent_chat.py`
  - 绑定状态与最小 proposal 卡片。
- `src/fem_gui/widgets/viewport_toolbar.py`
  - 可选 bridge 注入。
- `src/fem_gui/main_window.py`
  - 只读 snapshot 绑定刷新，不增加 session 写入。
- `tests/test_agent_authoring_contracts.py`
  - 纯 DTO、账本、草稿、hash、idempotency 和 Fake Port。
- `tests/gui/test_agent_authoring_bridge.py`
  - bridge、事件、GUI 授权、陈旧和线程边界。

## 聚焦测试

计划只运行：

```text
pytest -q tests/test_agent_authoring_contracts.py
pytest -q tests/gui/test_agent_authoring_bridge.py
pytest -q tests/gui/test_agent_event_contract.py tests/gui/test_agent_chat_overlay.py
```

覆盖：

1. 切换 document/session/revision 后待确认提案立即陈旧。
2. 自然语言、Provider 工具路径和缺失 GUI 授权不能接受提案。
3. `RequirementReview` 未经 GUI 确认时阶段门返回
   `clarification_required`。
4. `AuthoringContext` 不包含完整数组、绝对路径或 GUI 对象。
5. proposal 事件严格校验、严格顺序并可得到相同重放快照。
6. Provider/Agent 工具只通过后台 runtime；GUI bridge 仅由主线程控件调用。
7. 成功、拒绝、陈旧、异常、重复点击和幂等重放均有确定终态。

## 交付审查

完成后核对：

- `git diff` 不包含 README 或 `src/fem/application/session.py`。
- 没有真实建模、网格、定义或求解调用。
- proposal 卡片只使用本地安全摘要，不展示原始 patch。
- 测试命令均为上述聚焦范围。
- 上游设计文档的当前状态由主 Agent 在审阅所有 phase 后统一记录和提交。

## 实施结果

- 纯契约测试：
  `pytest -q tests/test_agent_authoring_contracts.py`，6 passed。
- bridge/事件/GUI 测试：
  `pytest -q tests/gui/test_agent_authoring_bridge.py`，6 passed。
- 既有事件与聊天覆盖层回归：
  `pytest -q tests/gui/test_agent_event_contract.py
  tests/gui/test_agent_chat_overlay.py`，25 passed。
- 既有 runtime 与主窗口布局回归：
  `pytest -q tests/gui/test_agent_runtime.py
  tests/gui/test_main_window_layout.py`，23 passed。
- 未运行全量测试，未调用 Computer Use，未修改 README，未修改真实
  `ModelSession`，未创建 git commit。

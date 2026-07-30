# FEM Agent Authoring Phase A8 实施计划

## 状态

- 阶段：A8
- 基线：`68f2c61`
- 状态：已实现，主 Agent 审查通过
- 上游边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

以“帮我建立一个偏心的带孔平板模型，孔的位置偏离板的中心”为首个确定性验收
提示，贯通 A1–A7 已有本地契约，并加固动态工具、确认、失败、恢复、隐私和资源边界：

1. 模糊提示只进入需求澄清，不触发几何、Gmsh 或求解。
2. 多轮追问后形成完整 `RequirementReview`，工程值只有 GUI 确认后进入
   `confirmed`。
3. `AgentSessionEngine` 支持注入严格动态工具目录；`QtAgentRuntime` 正式持有
   A8 controller，并把工具执行封送到 GUI owner thread。
4. 几何、网格和求解只创建本地提案，执行授权仍只来自既有 GUI 控件。
5. 作用域、材料、截面和分析定义复用 A4/A5 自动可逆 patch。
6. 预检、求解和结果查询复用 A6/A7 bridge，不复制 FEM、Gmsh、求解或结果算法。
7. Fake Provider 通过真实 engine/runtime 动态定义与 dispatch seam 驱动确定性
   多轮端到端流程。

## 端到端架构

### 动态工具目录

- 在 `fem_agent` 增加纯 Python 的 A8 workflow controller：
  - 封闭的阶段枚举；
  - 每阶段允许的 `ToolDefinition`；
  - 严格字段校验和 handler dispatch；
  - Provider-safe `ToolResult`；
  - GUI 确认、拒绝、失败、取消和 stale 的本地终态通知；
  - 动态目录中永远没有 `accept_proposal`、`confirm_mesh`、
    `confirm_solve` 或 `confirm_requirement_review`。
- 现有 `AgentToolRegistry` 保留全部 V0 工具，并可组合一个严格动态目录。
- `AgentSessionEngine` 每次 Provider round trip 重新读取动态 definitions，阶段推进
  后不继续发布上一阶段的写工具。

### Qt owner-thread 封送

- `QtAgentRuntime` 持有 A8 controller，而不是持有 `ModelSession`。
- engine 后台线程调用动态工具时，只把工具名、JSON 参数和有界执行上下文发送给
  runtime。
- runtime 通过 queued Qt signal 把调用封送到 owner thread；owner thread 上的
  controller handler 才能调用注入的 `AgentAuthoringBridge` 或
  `AgentResultQueryBridge`。
- 返回 engine 的内容必须先通过 JSON、大小、深度、集合数量、路径和禁止字段检查。
- GUI bridge、`ModelSession`、Qt/VTK、原始 patch 和结果 records 不跨线程进入
  engine 或 Provider payload。

### Fake Provider 脚本

- 第一轮收到模糊提示，仅输出几何/材料/网格/载荷/结果缺项追问，不调用工具。
- 后续轮次显式调用需求更新和 review 请求；测试在 GUI owner thread 确认 review。
- Provider 随 controller 阶段只看见当前允许工具，依次准备几何、网格、自动定义、
  预检、求解和 A7 结果查询。
- 几何、网格和求解的实际执行由测试模拟 GUI 控件调用既有 bridge，再把权威终态
  通知 controller；Provider 工具本身不能授权。
- Fake Provider 捕获的每个 request 都检查不含完整节点/单元/结果数组、绝对路径、
  Qt/VTK/`ModelSession`、raw patch 或凭据。

## 允许项

- 为 engine/registry/runtime 增加最小可注入动态定义与 handler seam。
- 增加 A8 controller、严格工具 schema、阶段状态和有界诊断。
- 让主窗口把既有 authoring/result bridge 注入 runtime owner-thread handler。
- 复用 A1–A7 DTO、proposal、patch、preflight、solve 和 result query bridge。
- 增加 Fake Provider 纯本地端到端测试和最相关 V0 `.inp`/GUI 回归。
- 可选真实 Provider 冒烟仅在现有凭据和现有配置可用时显式选择/跳过。

## 禁止项

- 不增加任何 Provider 可调用的确认、接受、拒绝、取消或 GUI click 工具。
- 不让 engine 后台线程直接访问 `ModelSession`、GUI、Qt、VTK 或 Gmsh。
- 不复制几何、网格、求解、预检或结果聚合算法。
- 不发送完整几何、节点、单元、结果数组、records、绝对路径、raw patch、
  Agent 私有路径或凭据。
- 不在模糊需求、未确认 review、stale revision 或阻塞预检下继续。
- 不自动安装依赖、修改 Provider 配置、运行全量测试或使用 computer use。
- 不修改 README、顶层 boundary 文档，不创建 commit。

## 目标文件

- `src/fem_agent/authoring_runtime.py`（新增）
  - A8 阶段、动态 tool catalog、严格 dispatch、状态通知和 payload 加固。
- `src/fem_agent/tools/registry.py`
  - 组合 V0 与注入动态工具，拒绝重名和未知调用。
- `src/fem_agent/engine.py`
  - 注入动态目录并在每个 Provider 回合刷新 definitions。
- `src/fem_gui/agent_runtime.py`
  - 正式持有 controller，owner-thread tool invocation 和有界 shutdown。
- `src/fem_gui/widgets/agent_chat.py`
  - 向 runtime 注入 controller/bridge，并在 GUI proposal 操作后报告终态。
- `src/fem_gui/widgets/viewport_toolbar.py`
  - 透传 A8 controller/result bridge。
- `src/fem_gui/main_window.py`
  - 用已有 authoring/result bridge 建立 controller handler，不暴露私有方法。
- `tests/test_agent_authoring_phase_a8.py`（新增）
  - controller 阶段、schema、确认缺失、终态、隐私与资源边界。
- `tests/gui/test_agent_authoring_e2e_phase_a8.py`（新增）
  - Fake Provider、真实 engine/runtime seam、三类 GUI 确认和成功/失败/恢复链。
- 本计划文档。

最终实现按现有职责做最小调整；不为满足清单制造空改动。

## 确认点

1. `RequirementReview`：只能由 `confirm_requirement_review_from_gui` 确认。
2. 几何：工具只准备并展示 proposal；“加入模型”GUI 控件执行。
3. 网格：工具只准备并展示 proposal；“开始划分”GUI 控件启动后台 Gmsh。
4. 求解：工具只准备并展示 proposal；“开始求解”GUI 控件启动既有后台作业。
5. 自动 patch：仅 A4/A5 已允许的纯新增、无结果、可逆编辑。

## 失败、取消与恢复

- 文档/session/revision 改变：待 review/proposal 进入 stale；下一次工具调用先重新读取
  context。
- Provider turn 取消：终止本轮，不撤销已经接受的 patch。
- 网格/求解取消或失败：保留上游已接受模型，丢弃晚到结果，并记录唯一终态。
- 动态 handler 超时、runtime shutdown 或 payload 越界：返回稳定本地诊断，不扩大
  能力。
- 保存、关闭、重开：通过 schema v10 和既有 session/project API 验证已接受几何、
  网格意图、作用域和分析定义可再次生成网格、预检和求解；未执行提案不恢复为
  pending。

## 聚焦测试

计划运行：

- `tests/test_agent_authoring_phase_a8.py`
- `tests/gui/test_agent_authoring_e2e_phase_a8.py`
- A1–A7 直接契约：
  - `tests/test_agent_authoring_contracts.py`
  - `tests/test_agent_authoring_phase_a2.py`
  - `tests/test_agent_authoring_phase_a3.py`
  - `tests/test_agent_authoring_phase_a4.py`
  - `tests/test_agent_authoring_phase_a5.py`
  - `tests/test_agent_authoring_phase_a6.py`
  - `tests/test_agent_authoring_phase_a7.py`
- runtime、bridge 和 V0 最相关回归：
  - `tests/gui/test_agent_runtime.py`
  - `tests/gui/test_agent_authoring_bridge.py`
  - `tests/test_agent_e2e.py` 中单个 Fake Provider `.inp` 流程。
- schema v10 保存/重开和重编译聚焦用例。
- A8 目标文件 Ruff、编译检查及 `git diff --check`。

不运行全量测试；真实 Gmsh/求解只在 A8 fixture 确实需要时单独运行。

## 完成条件

1. A8 十二条验收由确定性测试覆盖。
2. 动态工具目录按阶段变化，确认能力永不发布。
3. engine/runtime/GUI owner-thread 的真实调用 seam 有自动化覆盖。
4. 成功、拒绝、stale、取消、网格失败、求解失败和恢复均有唯一终态。
5. Provider payload 隐私和资源预算有正反测试。
6. V0 `.inp` Agent 和最相关 GUI runtime/bridge 回归通过。
7. 状态更新为“实现完成，等待主 Agent 审查”，不创建提交。

## 实施记录

- 新增 `AuthoringWorkflowController`，按需求审查、几何、网格、A4、A5、
  预检、求解和结果阶段动态发布严格工具；确认、接受、拒绝和取消能力不进入
  Provider catalog。
- `AgentSessionEngine` 仅在注入动态目录时追加 A8 authoring contract，并在每次
  tool round 后刷新目录；未注入时 V0 system prompt 保持原样。
- `QtAgentRuntime` 通过 queued signal 把动态调用封送到 GUI owner thread；
  runtime 只传递工具名、JSON 参数、执行上下文和 Provider-safe 结果。
- 生产 controller 复用 A1–A7 现有 bridge 与 DTO。A4 和 A5 分为两个原子、
  可逆 patch，A5 构造失败不会部分写入该阶段定义。
- 首里程碑 schema 只发布实际支持的 2D 组合：
  `plane_stress`/`plane_strain`、DOF 1..2、edge traction/pressure；review 前
  校验载荷方向、压力符号和 DOF 连续性。整数数组 item 的
  `minimum`/`maximum` 按 schema 执行。
- 生产 controller 创建时立即建立 document/session/revision 基线。主动 GUI
  几何、网格、A4/A5、预检和求解 revision 可继续推进；外部 revision、
  文档/session 切换或 stale proposal 会立即清空 ledger、进入 `STALE`，只发布
  `read_authoring_context`。首个 native project 导致 session ID 切换的合法
  revision 也有专门覆盖。
- payload 在离开 owner thread 前限制 JSON 类型、大小、深度和集合数量，并拒绝
  绝对路径、节点/单元/连接/结果数组、`ModelSession`、Qt/VTK/Gmsh、
  raw patch 和凭据字段；本地异常只返回稳定类型诊断。

## 验证记录

- A8 controller/runtime/生产注入：`22 passed in 39.78s`；随后对
  RequirementReview 拒绝按钮的可见尺寸调整单独复验通过。
- A1–A7 关键合同、几何/网格零刷新、求解终态、schema v10 保存重开、
  V0 registry/engine prompt/Fake Provider `.inp`/GUI runtime：
  `73 passed in 20.14s`。
- 目标文件 Ruff：通过（项目 venv 未安装 Ruff，按项目约定使用全局 Ruff）。
- 目标 Python 文件内存编译检查：通过。
- `git diff --check`：通过；仅有仓库既有的 LF/CRLF 提示。
- 未运行全量测试，未使用 computer use，未创建 commit。
- 可选真实 Provider 冒烟未运行；本阶段没有修改 Provider 配置，也不依赖真实
  凭据完成确定性验收。
- 主 Agent 独立复验：A8 controller/runtime/生产注入 `22 passed in 40.20s`；
  A1–A7、V0 engine/runtime、GUI bridge、保存重开与求解生命周期精选回归
  `11 passed in 22.56s`；目标 Ruff、compileall 与 `git diff --check` 均通过。

## 主审增量修正

- queued owner-thread invocation 使用 per-invocation 锁仲裁 `claim`、`cancel` 与
  `finish`；timeout 或 shutdown 会先原子完成取消，晚到 Qt slot 在 dispatch 前
  跳过，不能执行本地 handler。
- RequirementReview 卡片不再查询未注册的 AgentProposal。按钮严格核对
  pending review ID/hash、controller 阶段以及 bridge 当前
  document/session/revision；QTest 已分别点击确认和拒绝并验证阶段前进/回退。
- `set_authoring_requirements` 先验证整个字段批次，再进行第二遍 ledger 写入；
  混合有效/无效调用失败时 revision 与 entries 均保持不变。
- 新 Agent session 成功切换时，runtime 在 GUI owner thread 清空 controller
  ledger/stage；drawer 将旧 pending proposals 标记 stale 并重新 seed 当前本地
  binding，新 engine session 不继承旧 confirmed requirements 或 pending
  operation。
- 清理了 V0 system prompt 尾部两行意外前导空格。

## 主审关注点

- 原生 Gmsh/求解调用一旦进入既有后端，其可中断粒度仍由 A3/A6 现有任务边界
  决定；A8 只消费权威终态。
- controller 的需求 ledger 是 Agent 私有会话状态；保存重开验证的是已接受的
  schema v10 模型、网格意图、作用域和分析定义，未执行 proposal 不恢复为
  pending。

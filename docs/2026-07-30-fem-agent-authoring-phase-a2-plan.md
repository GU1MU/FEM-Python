# FEM Agent V1 Authoring Phase A2 实施计划

## 状态

- 日期：2026-07-30
- 基线：A1 提交 `3e8037e`
- 阶段：A2（单位、命名与几何草稿/提交）
- 状态：实现、聚焦测试与主 Agent 审查已完成
- 绑定边界：`2026-07-30-fem-agent-autonomous-authoring-boundary.md`

## 目标

1. 为 native 项目增加类型化单位上下文，并支持旧项目迁移、保存与重开。
2. 建立统一 `NamePolicy` 和 `NameAllocator`，生成满足
   `{type}-{function}`、规范化唯一且可重放的名称。
3. 提供矩形、圆盘、偏心带孔平板、长方体、圆柱及必要平移/旋转的纯后端几何草稿工具。
4. 为几何草稿生成有界静态预览和 revision 绑定的几何提案。
5. 经 GUI 本地确认后，原子完成“空白会话创建 native 项目和首部件”或
   “向现有 native 项目增加部件”，成功后只执行一次完整投影刷新。

## 允许项

- 扩展应用层 native 项目、项目编解码和迁移契约。
- 扩展 A1 的纯 authoring DTO、工具和 bridge。
- 增加受控的 `ModelSession` 原子几何提交入口。
- 使用现有类型化 geometry recipe；只传递稳定 part ID、recipe DTO 和受控工程参数。
- 使用与 GUI 视口隔离的有界静态预览。
- 增加 A2 专用聚焦单元和 GUI bridge 测试。

## 禁止项

- 不生成网格，不引入 `MeshIntent`，不提前实现 A3。
- 不删除、替换或合并用户已有部件。
- 不暴露 Gmsh tag、OCC handle、任意脚本或任意 `gmsh.option`。
- 不让 Provider、Agent engine 或工具直接操作 Qt、VTK、主窗口或
  `ModelSession`。
- 不在确认前修改 GUI revision、模型树、Actor、相机或选择。
- 不修改 `README.md` 或 `README_Zh.md`。
- 不运行全量测试，不提交 commit。

## 目标文件

- `src/fem/application/`：单位上下文、项目状态和原子部件安装。
- `src/fem/io/`：`.femproj` 编解码与旧项目迁移。
- `src/fem/geometry/`：复用并补齐首批类型化 recipe 的严格验证。
- `src/fem_agent/authoring.py`：命名、几何草稿、静态预览和 proposal payload。
- `src/fem_gui/agent_authoring.py`：A1 bridge 的真实几何提交端口。
- `src/fem_gui/` 相关投影接点：接受后一次刷新。
- `tests/`：A2 专用聚焦验收、拒绝、陈旧和失败原子性测试。
- `docs/2026-07-30-fem-agent-autonomous-authoring-boundary.md`：完成后记录状态和聚焦验证。

具体文件以现有职责边界为准，保持最小修改面。

## 实施批次

1. 单位上下文、native 项目保存和旧项目迁移。
2. `NamePolicy`、`NameAllocator` 及确定性冲突测试。
3. 首批几何草稿、偏心孔严格验证和有界静态预览。
4. 几何 proposal、真实 authoring port、`ModelSession` 原子提交及一次刷新。
5. A2 八条验收与拒绝、陈旧、失败原子性聚焦测试。
6. 更新设计文档状态和验证记录。

## 聚焦测试

- 单位与项目迁移/保存重开测试。
- 命名策略的 Unicode/大小写规范化、命名空间、稳定数字后缀和非法名称测试。
- 几何 recipe、孔心坐标/中心偏移、孔完整位于板内及预览有界性测试。
- bridge 在确认前、拒绝、陈旧、提交失败及成功路径的 revision/状态/刷新次数测试。
- 空白会话创建项目与首部件的原子性、native 项目只新增一个部件测试。

只运行上述新增测试及直接受影响的既有 A1、project codec/session 聚焦测试。

## 完成条件

1. A2 八条验收均由自动化测试覆盖并通过。
2. 拒绝、陈旧和失败路径保持 GUI 状态原子不变。
3. 不包含网格生成或其他 A3 能力。
4. 设计文档准确记录 A2 状态、测试命令、结果和剩余风险。

## 实施记录

- 已增加应用层 `UnitContext`；长度、力和应力必填，密度、加速度和约定名称可明确
  为不适用。
- 当前项目格式升级为 schema v8；v8 在严格 v7 canonical Part authoring 上增加
  `unit_context`，v1–v7 仍可读取，缺失单位迁移为 `None`。
- 已增加 `NamePolicy` 和 `NameAllocator`，覆盖 NFKC/大小写冲突、显式命名空间、
  稳定短数字后缀、系统名冒充和本地最终分配校验。
- 已增加矩形、圆盘、偏心带孔平板、长方体、圆柱、平移和旋转草稿工具，以及最多
  64 点的静态线框预览。
- 偏心孔支持孔心坐标或板中心偏移，两者必须且只能提供一种；使用既有严格 recipe
  校验保证孔边界完整位于板内。
- 已增加空白会话“native 项目 + 单位 + 首部件”单 revision 原子事务，以及向
  native 项目增加部件的真实 A2 port。
- 主窗口已从 A1 fake port 切换至真实几何 port；确认成功后执行一次完整投影重建。
- 几何 port 明确传入 `mesh_settings=None`，未增加网格意图、Gmsh 网格调用或 A3
  能力。

## 聚焦验证记录

通过：

- `python -m pytest tests/test_agent_authoring_phase_a2.py
  tests/gui/test_agent_geometry_commit_phase_a2.py
  tests/gui/test_agent_authoring_bridge.py tests/gui/test_main_window_layout.py -q`
  ：26 项通过。
- `python -m pytest tests/io/test_project_v8.py tests/io/test_project_router.py
  tests/io/test_project_codec.py tests/application/test_project_v1_session.py -q`
  ：50 项通过。

补充回归：

- `tests/application/test_native_multi_part.py`、`tests/gui/test_project_io.py` 和
  `tests/io/test_project_v7.py` 在组合运行中通过。
- 同批加入 `tests/application/test_session_revisions.py` 时有 2 项既有失败：
  `can_save` 对空 native 项目的现有行为与测试预期不一致，以及既有
  `replace_geometry`/重命名 characterization 的 active recipe 预期不一致。对比
  A1 基线 `3e8037e`，相关产品实现与 A2 前相同，因此未在 A2 扩大范围修复。

## 剩余风险

- 静态预览为确定性有界线框数据，未尝试替代精确 OCC/VTK 视口投影。
- 三维部件接受仍复用既有单实体认证；A2 未新增 Gmsh 进程协调器，该协调器属于 A3。
- 旧项目单位保持缺失；Agent 后续建模前必须通过需求确认补充，A2 不猜测单位。

## 主 Agent 审查

- 结论：通过。
- 独立验证：A2 新增验收 13 项通过；A1、project router/v7、native
  multi-part、GUI project I/O 和主窗口布局聚焦回归 67 项通过。
- 边界核对：没有 README 修改、Gmsh 调用、`MeshIntent` 或网格生成。
- 已知基线：`tests/application/test_session_revisions.py` 中空 native 项目的
  `can_save` 预期和 legacy `replace_geometry` 重命名 characterization 在 A1
  基线实现中已存在，A2 未修改对应行为。

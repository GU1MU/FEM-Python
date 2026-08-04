# Abaqus B31 INP Phase 0 冻结记录

日期：2026-08-04

对应计划：[2026-08-04-abaqus-b31-inp-import-expansion-plan.md](2026-08-04-abaqus-b31-inp-import-expansion-plan.md)

范围：仅 Phase 0；本记录不改变生产行为。

## 1. Phase 0 边界

本阶段冻结目标契约、现有调用方、source orientation 词汇和可重复的当前行为
characterization。以下内容明确留在后续阶段：

- 不新增或修改 `fem.io.inp` 生产 facade；
- 不移动 `fem.abaqus` 模块、不删除旧公开符号；
- 不实现节点法向、orientation node、`*NORMAL` 或 element-end frame 算法；
- 不改变 B31 topology gate、`*PREPRINT` 处理或 output execution 行为；
- 不把当前拒绝结果升级为永久产品契约。

## 2. 冻结的最终公开入口

Phase 1 以后，完整 Abaqus INP 模型导入只通过下列入口提供：

```text
fem.io.inp.read(path: str | Path) -> FEMModel
fem.io.inp.read_with_report(path: str | Path) -> InpImportResult
```

`fem.io.inp` 的公开 report value 冻结为以下最小形状。字段必须 detached、owned，
并且不能持有 parser、deck 或 builder 的可变状态：

| value | 最小公开字段 | 约束 |
| --- | --- | --- |
| `InpImportResult` | `model`, `notices` | `model` 为完整 `FEMModel`；`notices` 为不可变序列 |
| `InpImportNotice` | `code`, `message`, `locations` | notice 是非权威限制/近似报告，不进入物理模型 |
| `InpSourceLocation` | `path`, `line`, `keyword` | `line` 为正整数；`keyword` 可为空 |
| optional summary | `source_summary`、`orientation_summary` | 只有在成为只读 typed value 后才可公开；不是 parser state |

`notices` 的最小 item shape 与当前 `AbaqusImportNotice` 的三个字段一致。Phase 0 不
引入第二份实现，当前类型仍由 `fem.abaqus` 提供；迁移阶段再把该 value 归属到
`fem.io.inp`。

## 3. 冻结的公开错误族

最终 facade 保留一个 INP 输入错误基类和三个可区分的语义层次：

| 角色 | 最终 facade 名称 | 语义 |
| --- | --- | --- |
| base | `InpInputError` | 所有可定位的输入错误基类 |
| lexical/structural | `InpParseError` | 文本、keyword、record 或 source shape 错误 |
| semantic construction | `InpBuildError` | 已解析 source 无法构造当前模型 |
| valid but unsupported | `UnsupportedInpFeatureError` | 合法 Abaqus 语义超出当前 capability |

错误族的最小稳定属性为 `code`、`message`、`location`、`locations`、`record` 和
`remediation`。`path`、`line`、`keyword` 是从首个 source location 暴露的便捷属性；
错误字符串必须包含可读的 source location 和 remediation（若有）。

Phase 0 继续保留现有 `AbaqusInputError`、`AbaqusParseError`、`AbaqusBuildError`、
`UnsupportedAbaqusFeatureError` 和 `AbaqusSourceLocation`，不建立兼容 wrapper，也不
在本阶段变更它们。上表是 Phase 1/7 的目标归属和命名冻结。

## 4. Source orientation 术语表

下表中的 “source” 指输入文件中写出的证据；“effective” 指经过语义解析后交给
core element 的结果。二者不得互相覆盖或伪装。

| 术语 | 冻结含义 | identity / provenance 约束 |
| --- | --- | --- |
| default n1 | source 没有写 assignment-scoped n1 时，adapter 按明确规则选择的参考方向 | 必须标记为 default/generated，不能显示成用户显式输入 |
| section n1 | `*BEAM SECTION` geometry record 后的 approximate n1 数据记录 | assignment-scoped；保留其 source span |
| orientation node | B31 connectivity 中用于方向定义的额外节点 | source-only reference；不成为 Beam2 结构节点、连接或额外自由度 |
| nodal normal | 节点记录附带的 normal 分量，或 `*NORMAL` 对节点/单元端给出的 normal | 至少按 node、element 和 local end 保留身份；不能只放松散 metadata |
| generated normal | adapter 根据拓扑、切线和已冻结规则推导的方向 | 必须与显式 source normal 区分，并带 generated evidence |
| effective element frame | 供 core Beam2 消费的已验证正交局部 frame | 至少含 local x/y/z 和 element identity；不能反向覆盖 source provenance |
| element-end frame | 以 `(element_id, local_end, node_id)` 为身份的端部 frame | 允许同一节点对不同 element-end 拥有不同方向；当前 Beam2 尚未消费该 field |

相关约束：element connectivity 的 source 顺序保持不变；共享节点不得通过复制节点
来规避 frame 问题；Abaqus B31 与当前 Euler–Bernoulli Beam2 的 shear-deformation
差异继续由 `abaqus.b31.euler_bernoulli_approximation` notice 表达。

## 5. 当前行为 characterization 基线

基线测试位于
[`tests/characterization/test_phase0_abaqus_b31_baseline.py`](../tests/characterization/test_phase0_abaqus_b31_baseline.py)。
每个输入由测试中的最小字符串写入 pytest `tmp_path`；没有读取或派生仓库
`data/` 文件。

| case | 当前结果 | 当前 code / evidence | 后续处理 |
| --- | --- | --- | --- |
| `preprint` | build 拒绝 | `abaqus.line.keyword_unsupported`；`*PREPRINT` 被视为未接受 keyword | Phase 1 改为 harmless ignored，并保留 occurrence |
| `kink` | build 拒绝 | `abaqus.b31.nodal_normal_averaging_unsupported` | Phase 2/4 改为逐单元 frame 或 resolver acceptance |
| `t_junction` | build 拒绝 | `abaqus.b31.nodal_normal_averaging_unsupported`；branch/junction | Phase 2/4 改为 acceptance |
| `closed_loop` | build 拒绝 | `abaqus.b31.nodal_normal_averaging_unsupported`；closed loop | Phase 2/4 改为 acceptance |
| `orientation_node` | parse 拒绝 | `abaqus.b31.orientation_node_unsupported` | Phase 3 解析并保留 source-only node |
| `nodal_normal` | build 拒绝 | `abaqus.b31.nodal_normals_unsupported` | Phase 3/4 解析 typed source 并交给 resolver |
| `normal_keyword` | build 拒绝 | `abaqus.line.keyword_unsupported`；`*NORMAL` 不在当前 line subset | Phase 3/4 按语义和 oracle 冻结后支持 |
| `output_parent_child` | build 成功 | parent/child parameters、flags、variables 保留；B31 仍有 Euler–Bernoulli notice | Phase 6 按变量投影 execution report，不阻断 physics import |

这些结果是迁移前的观察值，不是后续阶段必须保留的 rejection contract。保留的
永久约束只有计划中明确的 fail-closed 条件，例如无效/非有限/平行方向和事务性不变。

## 6. 测试隔离规则

- Phase 0 新增测试只使用内联最小文本、pytest `tmp_path` 或 `tests/fixtures/inp/`
  中独立维护的最小 fixture；本阶段新增测试只使用内联文本和 `tmp_path`。
- 不读取、复制、裁剪、转换或派生 `data/` 中的任何文件。
- 门式框架未来使用新的小型两跨或 T 形拓扑，不复用产品数据的节点号、单元号、集合
  或完整拓扑。
- characterization 的拒绝断言必须标记为当前基线语义；能力阶段应更新为新的
  acceptance/negative 测试，而不是把旧错误码继续当作永久目标。

## 7. 调用方清单

机器可读的调用方 ledger 见
[`2026-08-04-abaqus-b31-inp-usage-ledger.json`](2026-08-04-abaqus-b31-inp-usage-ledger.json)。
扫描范围是 `src/`、`tests/`、`examples/`，明确排除 `data/`；ledger 覆盖 GUI、Agent、
application 测试、普通 tests 和 examples，并把当前 mesh-only `fem.io.inp` reader 的
测试调用与完整模型导入入口分开记录。

# 任务2 3D节点应力插值修正实施说明

## 目标

修正 `Hex8`、`Tet4`、`Tet10` 的 3D 节点应力计算和导出链路，使项目结果能清楚区分：

1. 积分点应力
2. 单元节点应力
3. region-aware 节点平均应力
4. CSV / legacy VTK 导出应力

本任务优先修正 3D 实体单元。2D 平面单元、梁单元、杆单元的应力导出逻辑不作为本任务改动范围。

## 已确定的默认规则

### 节点平均

- 平均权重：算术平均。
- averaging threshold：默认 `75`，可配置。
- threshold 判断：只使用 `mises`。
- cluster 一旦确定，所有应力分量共用同一套 `cluster_id`。
- cluster 内先平均 6 个应力分量，再由平均后的分量重新计算输出 `mises`。
- 默认不跨材料平均。
- 默认不跨 section / property 平均。
- 默认不跨 element type 平均。
- 暂不考虑局部坐标系、orientation、材料方向转换。

### region 与 cluster

`region_key` 用于表达不允许跨越的平均边界：

```text
region_key = (material_id, section_id, element_type_id)
```

后续如果需要支持 instance，可扩展为：

```text
region_key = (instance_id, material_id, section_id, element_type_id)
```

`cluster_id` 用于表达同一节点、同一 region 内，因为 averaging threshold 不通过而拆分出的平均组：

```text
vtk_point_key = (original_node_id, region_key, cluster_id)
```

含义：

- 不同 `region_key` 永远不平均。
- 同一 `region_key` 内，threshold 通过的贡献进入同一个 cluster。
- threshold 不通过的贡献拆成多个 cluster，各自平均、各自导出。

### VTK 导出

保留单文件 legacy `.vtk`。

不引入 `.vtm + .vtu` MultiBlock。region-aware 节点应力通过 duplicate boundary points 表达：

```text
vtk_point_key = (original_node_id, region_key, cluster_id)
```

同一个物理节点如果属于不同材料、section、element type，或者同一 region 内 threshold 分裂成多个 cluster，则在 VTK 中写成多个坐标重合但 PointData 不同的 point。

### 不作为本任务重点

- 不新增公开诊断表。
- 不新增网格畸变质量检查。
- 不专门设计无贡献节点策略，沿用现有行为。
- 不重构整体求解器或材料系统。

## 应力计算链路

目标链路：

```text
kernel.nodal_stress()
  -> element nodal contributions
  -> group by original_node_id + region_key
  -> threshold clustering
  -> averaged region nodal stress rows
  -> CSV
  -> legacy VTK duplicate points
```

现有 `vtk.export.from_result()` 先生成 CSV 再写 VTK 的流程保留。

## 3D单元插值策略

### Hex8

当前问题：

```text
2x2x2 Gauss 点应力 -> 求平均 -> 复制到 8 个节点
```

这不是 Abaqus `ELEMENT_NODAL` 口径。

修正为：

```text
2x2x2 Gauss 点应力 -> Hex8 外推矩阵 -> 8 个单元节点应力
```

实现方法：

1. 使用 8 个 Gauss 点：

```text
g = 1 / sqrt(3)
xi, eta, zeta = +/- g
```

2. 使用 8 个自然节点：

```text
xi, eta, zeta = +/- 1
```

3. 构造矩阵：

```text
S_gp = N_gp @ S_node
S_node = E_hex8 @ S_gp
```

这里 `E_hex8` 是 Hex8 的 Gauss 点到节点外推矩阵。实现时不在每个单元内求逆，而是在模块级预计算一次：

```text
HEX8_EXTRAPOLATION_MATRIX
node_stress = HEX8_EXTRAPOLATION_MATRIX @ gp_stress
```

4. 对 6 个应力分量分别外推。

### Tet4

保持现有策略：

```text
重心常应力 -> 复制到 4 个节点
```

原因：`Tet4 / C3D4` 是常应变常应力单元，单元内节点应力差异主要来自跨单元平均，而不是单元内部外推。

### Tet10

当前问题：

```text
10 个自然节点位置直接 stress_at()
```

修正为：

```text
4 个 Hammer 积分点应力
  -> 拟合一次应力场
  -> 评价到 10 个 Tet10 自然节点
```

拟合形式：

```text
S(xi, eta, zeta) = a0 + a1 * xi + a2 * eta + a3 * zeta
```

对 6 个应力分量分别拟合。

## CSV设计

3D nodal stress CSV 改为 region-aware 节点应力结果。

CSV 每行表示一个 `element local node` 对应的 region-aware averaged nodal stress，而不是一个物理节点只写一行。这样 VTK 可以从 CSV 重建 duplicate point connectivity。

固定表头：

```text
source_elem_id
source_local_node
original_node_id
region_id
cluster_id
material_id
section_id
element_type_id
x
y
z
sig_x
sig_y
sig_z
tau_xy
tau_yz
tau_zx
mises
```

说明：

- `source_elem_id` 是贡献来源单元号。
- `source_local_node` 是贡献来源局部节点号，建议使用 1-based 编号以便与 CSV 人工检查一致。
- `original_node_id` 是原始 mesh 节点号。
- `region_id` 是 `region_key` 的整数编码。
- `cluster_id` 是 threshold clustering 后的平均组编号。
- `material_id`、`section_id`、`element_type_id` 是整数编码。
- 同一个 cluster 内可能出现多行，它们的平均应力值相同，但 `source_elem_id/source_local_node` 不同。
- `source_elem_id/source_local_node` 不是额外诊断表字段，而是 legacy VTK 重建 cell connectivity 必需字段。
- 字符串到整数的映射可以先只在内部稳定生成；如需审计，可后续补充映射导出。

## VTK设计

legacy `.vtk` 仍输出 `UNSTRUCTURED_GRID`。

### Points

VTK point 不再直接等于 mesh node。

构建规则：

```text
vtk_point_key = (original_node_id, region_key, cluster_id)
```

如果同一物理节点属于多个 region 或多个 cluster，则写成多个坐标相同的 VTK point。

### PointData

建议写入：

```text
original_node_id
region_id
cluster_id
material_id
section_id
element_type_id
sig_x
sig_y
sig_z
tau_xy
tau_yz
tau_zx
mises
```

### CellData

建议写入：

```text
original_element_id
region_id
material_id
section_id
element_type_id
```

### Cell connectivity

每个 element 的每个 local node 需要从 CSV 找到它对应的 averaged row：

```text
(source_elem_id, source_local_node)
  -> original_node_id
  -> region_id
  -> cluster_id
```

然后使用：

```text
vtk_point_key = (original_node_id, region_id, cluster_id)
```

构建 cell connectivity。

## threshold clustering 规则

同一个 `original_node_id + region_key` 内收集多个单元节点应力贡献。

本任务固定使用 `mises` 作为 threshold clustering 的唯一判断量：

```text
对每个贡献先计算 mises；
只用 mises variation 判断是否需要拆分平均组；
拆出的 cluster 对所有应力分量共用。
```

原因：

- CSV 和 VTK 一个 point 只能对应一组 PointData。
- 只用 `mises` 判断，规则简单，且适合一次性导出完整 3D 应力场。

建议算法：

1. 对候选贡献按 `mises` 排序。
2. 尝试加入当前 cluster。
3. 如果加入后 cluster 内 `mises` variation 超过 threshold，则新开 cluster。
4. cluster 内对 6 个应力分量做算术平均。
5. 由平均后的 6 个应力分量重新计算输出 `mises`。

variation 建议：

```text
variation = 100 * (max_mises - min_mises) / max(abs(max_mises), abs(min_mises), tiny)
```

## 代码改动位置

### `src/fem/elements/hexahedron.py`

改动点：

- 修改 `Hex8Kernel.nodal_stress()`。
- 从 Gauss 平均复制改为 8 点外推。
- 可新增私有 helper 构造外推矩阵。

### `src/fem/elements/tetrahedron.py`

改动点：

- 保持 `Tet4Kernel.nodal_stress()` 不变。
- 修改 `Tet10Kernel.nodal_stress()`。
- 从节点直接评价改为 Hammer 积分点一次场拟合。

### `src/fem/post/stress/averaging.py`

新增内部模块。

建议职责：

- 定义 averaging policy。
- 收集 element nodal contributions。
- 构造 region key。
- 执行 threshold clustering。
- 输出 region-aware nodal stress rows。
- 输出带 `source_elem_id/source_local_node` 的 CSV row，供 VTK connectivity 使用。

### `src/fem/post/stress/export.py`

改动点：

- `nodal()` 增加可选参数：

```python
model=None
averaging_policy=None
```

- 老接口保持兼容。

### `src/fem/post/stress/nodal.py`

改动点：

- 2D `_plane()` 和 `_plane_multi()` 暂不动。
- 3D `_solid()` 和 `_solid_multi()` 改为调用 averaging 模块。
- 3D CSV 表头改成带 `source_elem_id/source_local_node` 的 region-aware 表头。

### `src/fem/post/vtk/fields.py`

改动点：

- 保留旧 `read_nodal_stress()`。
- 新增或内部识别 region-aware nodal stress CSV。
- 新格式需要读取 `source_elem_id`、`source_local_node`、`original_node_id`、`region_id`、`cluster_id` 等字段。

### `src/fem/post/vtk/cells.py`

改动点：

- 保留旧 `build()`。
- 新增 region-aware cell builder。
- 根据 `(source_elem_id, source_local_node) -> (original_node_id, region_id, cluster_id)` 构建 duplicate point connectivity。

### `src/fem/post/vtk/writer.py`

改动点：

- 保留旧 `write()`。
- 新增更通用的 legacy unstructured writer，支持：

```text
points
cells
cell_types
point_data
cell_data
```

### `src/fem/post/vtk/export.py`

改动点：

- `from_result()` 调用 stress export 时传入 `result.model`。
- `from_csv()` 识别 region-aware nodal stress CSV。
- region-aware 时走 duplicate point VTK builder / writer。
- 旧格式仍走旧路径。

## 模型信息来源

有 `ModelResult` 时：

```text
result.model -> model_element_info(model, elem.id)
```

可获得：

- material
- section_type
- element type
- effective properties

只有 mesh 时：

```text
elem.props.get("material")
elem.props.get("section") / elem.props.get("section_type")
elem.type
```

作为兼容退回。

## 测试计划

### 单元算法测试

1. `Hex8` 非均匀位移场下，8 个节点应力不再全部相等。
2. `Tet4` 节点应力保持 4 节点相等。
3. `Tet10` 节点应力来自积分点拟合外推。

### averaging 测试

1. 同一节点、不同材料：不平均，输出不同 region。
2. 同一节点、不同 element type：不平均，输出不同 region。
3. 同一节点、同一 region，threshold 通过：同一 cluster。
4. 同一节点、同一 region，threshold 不通过：拆成多个 cluster。

### CSV 测试

1. 3D nodal stress CSV 包含 `source_elem_id`、`source_local_node`、`original_node_id`、`region_id`、`cluster_id`。
2. 混合单元共享节点出现多行 region-aware 结果。
3. 旧 2D stress CSV 不受影响。

### VTK 测试

1. region-aware VTK 的 `POINTS` 数量可大于 mesh 节点数。
2. VTK PointData 包含 `original_node_id`、`region_id`、`cluster_id`。
3. VTK CellData 包含 `original_element_id`、`region_id`。
4. 共享边界节点在不同 region 中使用不同 point index。

### Abaqus 对齐测试

分层对比：

```text
ELEMENT_NODAL:
  检查 Hex8 / Tet10 外推逻辑

NODAL / averaged:
  检查 region-aware 平均与 threshold 分裂逻辑
```

Abaqus 测试可作为可选集成测试；无 Abaqus 环境时跳过或使用 baseline CSV。

## 实施顺序

0. 先写/改 `Hex8`、`Tet10` 的单元节点应力测试。
1. 修改 `Hex8` 和 `Tet10` 的 `kernel.nodal_stress()`。
2. 新增 `post.stress.averaging` 内部模块并完成纯函数测试。
3. 改造 3D `stress.nodal` CSV 导出。
4. 新增 region-aware VTK CSV reader、cell builder 和 writer。
5. 改造 `vtk.export.from_result()` / `from_csv()` 的 region-aware 分支。
6. 补齐混合单元、threshold cluster、duplicate point VTK 回归测试。

## 开放确认点

当前仍需在实现前最终确认：

1. `section_id` 是否使用 `section_type`，还是需要从 section assignment 生成更细的 property id。
2. 字符串到整数 ID 的映射是否需要额外写入一个 mapping CSV。

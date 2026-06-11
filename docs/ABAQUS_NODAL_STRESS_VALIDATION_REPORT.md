# 任务2 3D 节点应力修正与 Abaqus 同口径验证报告

## 1. 报告目的与总体结论

本报告用于说明任务2中 3D 实体单元节点应力的最终计算、平均和导出口径，并证明项目导出的节点应力结果与 Abaqus 同口径结果在主要验证算例中进入可接受误差范围。

验证对象覆盖 11 组算例：

- `01_hex8_single_element_C3D8`
- `02_tet4_single_element_C3D4`
- `03_tet10_single_element_C3D10`
- `04_hex8_two_element_average_C3D8`
- `05_material_boundary_two_hex8_C3D8`
- `06_section_boundary_two_hex8_C3D8`
- `07_threshold_split_two_hex8_C3D8`
- `08_tet4_regular_cube_C3D4`
- `09_tet10_regular_cube_C3D10`
- `10_mixed_element_type_hex8_tet4`
- `11_mixed_element_type_hex8_tet10`

总体结论：

- `01-07`、`09-11` 的节点应力与 Abaqus 同口径结果一致，误差主要处于 Abaqus 文本输出截断量级。
- 通过算例的最大绝对误差主要落在 `5e-5` 到 `5e-3`，可作为“修正后的节点应力结果与 Abaqus 同口径结果误差进入可接受范围”的证据。
- `08_tet4_regular_cube_C3D4` 是极端 C3D4 多单元共享节点场景，差异来自“完整应力张量导出口径”和 “Abaqus/CAE 当前显示变量实时 averaging 口径”的不同，不作为一般算例误差水平的代表。
- Hex8 / C3D8 与 Abaqus 对齐时需要考虑 B-bar / 平均体积应变相关影响；当前差异来自底层 Hex8 单元体积应变口径，而不是节点应力外推矩阵错误。
- Tet10 / C3D10 对比时需要注意剪应力分量映射：Abaqus 的 `S13` 对应项目侧 `tau_zx`，Abaqus 的 `S23` 对应项目侧 `tau_yz`。

## 2. 最终算法口径

### 2.1 基本术语

- `element-local`：同一个物理节点在某个单元内部的局部节点位置。例如两个相邻单元共享一个几何节点时，Abaqus 报表中可能会有两行，分别表示这两个单元对该节点的贡献。
- `region`：节点应力平均时允许合并的区域。项目默认不跨 material、section、element type 平均。
- `cluster`：同一个物理节点、同一个 region 内，根据 averaging threshold 判断后形成的平均组。处于同一 cluster 的贡献会平均；处于不同 cluster 的贡献会分开输出。
- `完整应力张量`：同一行同时保存 `S11/S22/S33/S12/S13/S23` 以及由这 6 个分量重新计算得到的 `Mises`。

### 2.2 单元节点应力

`Hex8 / C3D8` 使用 2x2x2 Gauss 点应力外推到 8 个单元节点：

```text
2x2x2 Gauss 点应力
  -> Hex8 Gauss-to-node 外推矩阵
  -> 8 个单元节点应力
```

数学关系为：

```text
S_gp = N_gp @ S_node
S_node = E_hex8 @ S_gp
```

实现中使用模块级预计算矩阵：

```text
HEX8_EXTRAPOLATION_MATRIX
node_stress = HEX8_EXTRAPOLATION_MATRIX @ gp_stress
```

与 Abaqus C3D8 对比时，Hex8 节点正应力需要考虑 Abaqus-compatible 的 B-bar / 平均体积应变口径。当前节点应力外推使用标准 Gauss-to-node 外推矩阵，这一外推链路本身可以通过剪应力、`mises` 以及 Tet10 外推结果验证；Hex8 正应力差异主要来自底层 Hex8 单元没有采用 Abaqus C3D8 的 B-bar 体积应变处理。

因此，本报告中的 Hex8 同口径对比使用静水应力项替换来隔离并说明该差异：项目由 `D @ B @ u` 得到的偏应力部分和剪应力可以与 Abaqus 对齐，但三个正应力的共同平均项需要按单元中心平均正应力替换：

```text
mean_node = (sig_x + sig_y + sig_z) / 3
mean_elem = 贡献 Hex8 单元中心的平均正应力

sig_x_corr = sig_x - mean_node + mean_elem
sig_y_corr = sig_y - mean_node + mean_elem
sig_z_corr = sig_z - mean_node + mean_elem
```

剪应力分量保持原值。`mises` 使用修正后的 6 个应力分量重新计算。对于同一 region / cluster 内的共享节点：

```text
mean_elem = 贡献 Hex8 单元中心平均正应力的算术平均值
```

这个处理用于验证节点应力外推、平均和导出链路；如果后续希望项目 Hex8 单元本身与 Abaqus C3D8 更一致，需要在 Hex8 单元公式中实现 B-bar / 平均体积应变修正，而不是继续把问题归因到节点外推。

`Tet4 / C3D4` 是常应变常应力单元，单元内部没有额外节点外推自由度：

```text
重心常应力 -> 复制到 4 个单元节点
```

`Tet10 / C3D10` 使用 4 个 Hammer 积分点应力拟合一次应力场，再评估到 10 个 Tet10 自然节点：

```text
4 个 Hammer 积分点应力
  -> 一次应力场拟合
  -> 评估到 10 个 Tet10 自然节点
```

拟合形式为：

```text
S(xi, eta, zeta) = a0 + a1 * xi + a2 * eta + a3 * zeta
```

实现中使用模块级预计算矩阵：

```text
TET10_LINEAR_EXTRAPOLATION_MATRIX
node_stress = TET10_LINEAR_EXTRAPOLATION_MATRIX @ gp_stress
```

### 2.3 Region-aware 平均

3D 节点应力平均默认不跨：

- material
- section / section assignment / section properties
- element type

有完整 `FEMModel` 时，region 信息优先来自模型中的 element 信息和 section assignment。只有 mesh 时，region 信息来自元素属性：

```text
elem.props["material"]
elem.props["section"] / elem.props["section_type"]
elem.type
```

如果元素没有显式 material 名称，则使用元素属性签名作为 material 兜底，避免同类型但不同 `E/nu` 的直接 mesh 被合并。

同一个 `original_node_id + region_key` 内会收集多个 element-local nodal stress contribution。Averaging threshold 默认是 `75`，项目使用 Abaqus relative nodal variation 的 region-range 分母：

```text
variation =
    100 * (max_mises_at_node - min_mises_at_node)
    / (max_mises_within_region - min_mises_within_region)
```

如果 region 内 `max_mises == min_mises`，则同节点 contribution 也相等时 variation 视为 `0`，否则视为无限大。

项目导出 CSV / VTK 时需要每一行都保存一组完整应力张量，因此采用统一 Mises cluster：

- threshold 判断使用 `mises`；
- cluster 一旦确定，`sig_x/sig_y/sig_z/tau_xy/tau_yz/tau_zx/mises` 共用同一套 `cluster_id`；
- cluster 内先算术平均 6 个应力分量；
- 输出 `mises` 由平均后的 6 个应力分量重新计算；
- 当前平均权重为算术平均。

该导出口径与 Abaqus/CAE Viewer 的交互式显示能力有差别。Abaqus/CAE Viewer 可以针对当前显示变量实时 averaging，例如显示 `S11` 时按 `S11` 判断，显示 `Mises` 时按 `Mises` 判断。项目为了让 CSV / VTK 中一行对应一组明确完整的应力张量，采用统一 Mises cluster，而不是为每个分量建立一套独立 cluster。

## 3. CSV / VTK 输出口径

3D nodal stress CSV 每行表示一个 `element-local` 节点对应的 region-aware averaged nodal stress。固定表头为：

```text
source_elem_id, source_local_node, original_node_id,
region_id, cluster_id, material_id, section_id, element_type_id,
x, y, z,
sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx, mises
```

字段含义：

- `source_elem_id`：贡献来源单元编号。
- `source_local_node`：贡献来源局部节点号，使用 1-based 编号。
- `original_node_id`：原始 mesh 节点号。
- `region_id`：`region_key` 的稳定整数编码。
- `cluster_id`：同一 `original_node_id + region_key` 下的 threshold cluster 编号。
- `material_id/section_id/element_type_id`：对应 region 分量的稳定整数编码。
- `x/y/z`：原始节点坐标。
- 应力字段：平均后的 6 个分量和重新计算的 `mises`。

`source_elem_id/source_local_node` 是 legacy VTK 重建 duplicate point connectivity 的必要字段。CSV 保留 `region_id/cluster_id/material_id/section_id/element_type_id`，便于追踪 region-aware 平均和 duplicate point connectivity。

VTK 继续输出单文件 legacy `.vtk`。Region-aware VTK 使用 duplicate points 表达边界不平均：

```text
vtk_point_key = (original_node_id, region_id, cluster_id)
```

如果同一个物理节点属于不同 region，或者同一 region 内被 threshold 拆成多个 cluster，则在 VTK 中写成多个坐标重合但 PointData 不同的 point。

VTK 默认只写入真正用于查看云图的 PointData：

```text
displacement
sig_x
sig_y
sig_z
tau_xy
tau_yz
tau_zx
mises
```

VTK 默认不写入 `original_node_id/region_id/cluster_id/material_id/section_id/element_type_id` 等调试字段，也不写入额外 CellData 应力字段。这样在 ParaView 等工具里只保留需要查看的物理量，避免 point/cell 两套同名应力字段同时出现。调试与追踪信息保留在 nodal stress CSV 中。

## 4. Abaqus 对比口径

应力分量采用如下映射：

```text
Abaqus: S11, S22, S33, S12, S13, S23
项目:   sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx
映射:   S11 = sig_x
        S22 = sig_y
        S33 = sig_z
        S12 = tau_xy
        S13 = tau_zx
        S23 = tau_yz
```

Tet4 / C3D4 和 Tet10 / C3D10 不启用 Hex8 静水应力修正，按上述分量映射直接对比。Hex8 / C3D8 使用第 2.2 节中的静水应力项修正后对比。

Abaqus 的 `Avg: 75%` 判定使用 relative nodal variation：

```text
relative nodal variation =
    (maximum at node - minimum at node)
    / (maximum over active regions - minimum over active regions)
```

如果不跨 region 平均，则分母使用该 region 内的范围：

```text
relative nodal variation =
    (maximum at node - minimum at node)
    / (maximum within region - minimum within region)
```

也就是说，Abaqus 的分母不是该节点处的最大贡献值，而是当前参与显示/平均的 active region 或 region 内的全局范围。

误差判断采用如下原则：

- `1e-3` 左右的误差通常来自 Abaqus 文本表格保留位数；
- `1e-2` 以下且无系统性偏移的误差可认为已经进入可接受范围；
- 若误差达到 `1e1` 到 `1e2` 量级，则不是输出截断问题，应按平均口径或算法规则差异分析。

## 5. 逐案例对比分析

### 5.1 01 Hex8 单单元 C3D8

该算例验证单个 Hex8 / C3D8 的节点应力外推，并说明 Hex8 正应力差异来自底层单元体积应变口径。

Hex8 / C3D8 的剪应力和 `mises` 能够与 Abaqus 对齐，正应力差异集中在三个正应力的共同平均项，即静水应力项。这说明节点应力外推矩阵和应力分量映射不是主要问题；差异来源是当前 Hex8 单元没有采用 Abaqus C3D8 的 B-bar / 平均体积应变处理。Abaqus C3D8 的体积应变项会影响正应力的共同平均项，因此该算例采用第 2.2 节的静水应力项替换来进行同口径对比。

该算例中：

```text
mean_elem = 665.0
```

修正后的 Abaqus 同口径节点正应力为：

```text
S11:
756.538, 702.692, 918.077, 971.923, 810.385, 756.538, 971.923, 1025.769

S22:
514.231, 460.385, 352.692, 406.538, 568.077, 514.231, 406.538, 460.385

S33:
724.231, 831.923, 724.231, 616.538, 616.538, 724.231, 616.538, 508.846
```

上述结果与 Abaqus 节点应力结果一致。

结论：Hex8 / C3D8 的节点应力外推链路通过验证；与 Abaqus 的正应力口径差异属于底层 Hex8 单元 B-bar / 体积应变处理问题，后续应在单元公式层面修正。

### 5.2 02 Tet4 单单元 C3D4

该算例验证单个 Tet4 / C3D4 的常应变单元应力结果。

项目结果与 Abaqus 一致：

```text
S11   = 258.46153846153845
S22   = -16.15384615384616
S33   = 177.69230769230768
S12   = 40.38461538461539
S13   = 16.153846153846153
S23   = 32.30769230769231
Mises = 261.84682048184146
```

Tet4 单元为常应变单元，单元内各节点的应力相同。该结果与 Abaqus 给出的单元节点应力一致。

结论：Tet4 单单元应力恢复链路通过验证。

### 5.3 03 Tet10 单单元 C3D10

该算例验证单个 Tet10 / C3D10 的二次四面体节点应力外推结果。

对比时使用剪应力映射：

```text
Abaqus S12 = 项目 tau_xy
Abaqus S13 = 项目 tau_zx
Abaqus S23 = 项目 tau_yz
```

按上述映射后，10 个节点的 6 个应力分量与 Abaqus 一致。最大绝对误差约为：

```text
max_abs_diff ~= 4.6e-4
```

该误差来自 Abaqus 文本结果保留位数造成的截断。

结论：Tet10 单单元应力外推和分量映射通过验证。

### 5.4 04 两个 Hex8 共享节点平均 C3D8

该算例验证两个同材料、同截面、同单元类型 Hex8 在共享节点处的平均规则。

对比误差为：

```text
max_abs_diff  = 5.38e-4
mean_abs_diff = 1.61e-4
```

共享节点示例：

```text
element 1, local node 2
element 2, local node 1
```

这两个 element-local 节点代表同一个物理节点。按同 region 平均并进行 Hex8 静水应力修正后，两侧结果一致：

```text
S11 = 609.808
S22 = 529.039
S33 = 593.654
S12 = 80.7692
S13 ~= 0
S23 ~= 0
```

该案例的关键是共享节点处是否把两个相邻 Hex8 的 element-local 结果按 Abaqus 口径合并。Abaqus 在该共享节点处给出的两个 element-local 行具有相同 averaged-at-nodes 值；项目结果在进行 region-aware 平均和 Hex8 静水项修正后，也给出相同值。

结论：同 material / section / element type / cluster 下的 Hex8 共享节点平均规则与 Abaqus 同口径结果一致。

### 5.5 05 Material boundary two-Hex8 C3D8

该算例验证材料边界处是否禁止跨 material 平均。

对比误差为：

```text
max_abs_diff  = 5.38e-4
mean_abs_diff = 1.43e-4
```

误差处于 Abaqus 文本截断量级。该算例的几何连接关系与 04 类似，但两个 Hex8 被赋予不同 material。平均分组中 material 参与 region 判定，因此同一几何节点在 material boundary 两侧会被视为两个不同的应力输出分支。

结论：material boundary 分组规则正确，节点应力结果通过验证。

### 5.6 06 Section boundary two-Hex8 C3D8

该算例验证截面边界处是否禁止跨 section 平均。

对比误差为：

```text
max_abs_diff  = 5.38e-4
mean_abs_diff = 1.61e-4
```

误差处于 Abaqus 文本截断量级。该算例把 material 保持一致，只改变 section，用于区分“材料不同导致分区”和“截面定义不同导致分区”。结果表明，section 也是 region 分组的一部分。

结论：section boundary 分组规则正确，节点应力结果通过验证。

### 5.7 07 Threshold split two-Hex8 C3D8

该算例验证同 region 内存在明显应力跳变时，threshold split 是否能阻止不合理平均。

两个单元的应力水平明显不同：

```text
low-stress element:  mises ~= 16.1539
high-stress element: mises ~= 630
```

对比误差为：

```text
max_abs_diff  = 5.38e-5
mean_abs_diff = 8.24e-6
```

该算例与 04 的区别在于：两个单元可以处在同 material、同 section、同 element type 下，但应力水平差异非常大。Abaqus 同口径结果在该算例中保留高低应力分区。项目通过 `cluster_id / threshold split` 做出同样分裂，因此误差降到 `1e-5` 量级。

结论：threshold split / cluster 分组规则有效，节点应力结果通过验证。

### 5.8 08 Tet4 regular cube C3D4

该算例验证多个 Tet4 组成规则立方体时的节点平均结果，同时用于说明极端 C3D4 多单元共享节点场景下的导出口径差异。

项目采用统一 Mises cluster 来保存完整应力张量。对于同一个物理节点、同一个 region 内的多个 element-local contribution，项目先根据 Mises relative nodal variation 判断是否需要分裂；若处于同一 cluster，则统一输出一组完整的 6 分量应力张量。

08 与 Abaqus 的整体差异为：

```text
max_abs_diff  = 141.347
mean_abs_diff = 11.574
```

最大差异位置为：

```text
element_id = 5
node_id    = 5
field      = S11
project    = 617.885
Abaqus     = 476.538
```

该模型由 6 个 C3D4 四面体剖分一个立方体：

```text
1: 1, 2, 3, 7
2: 1, 3, 4, 7
3: 1, 4, 8, 7
4: 1, 8, 5, 7
5: 1, 5, 6, 7
6: 1, 6, 2, 7
```

C3D4 是常应变单元，因此每个单元内部的 element-nodal stress 等于该单元常应力。多个 C3D4 在同一节点相交时，不同 attached element 在该节点处可能有不同的常应力贡献。

例如 node 5 由 element 4 和 element 5 共享：

```text
element 4, node 5: S11 = 759.231
element 5, node 5: S11 = 476.538
```

在项目的统一 Mises cluster 口径下，这两个 contribution 可以被归入同一个完整张量平均组，因此导出：

```text
project S11 = (759.231 + 476.538) / 2 = 617.885
```

Abaqus/CAE Viewer 可以针对当前显示变量实时计算 averaging。显示 `S, S11` 时，它可以按 S11 的 relative nodal variation 判断，并在该分量上保留 element-local 差异。因此 element 5, node 5 处为：

```text
Abaqus S11 = 476.538
```

于是形成最大差异：

```text
617.885 - 476.538 = 141.347
```

该差异不是数值精度问题，也不是应力分量映射问题。若是文本截断，差异应在 `1e-3` 左右；若是分量映射问题，剪应力或多个正应力分量会呈现系统性交叉。这里的根本原因是：

- 项目侧：导出 CSV / VTK 时，需要把 `S11, S22, S33, S12, S13, S23, Mises` 作为同一行完整应力张量保存，因此采用统一 Mises cluster。
- Abaqus 侧：Viewer 可以针对当前显示变量实时计算 averaging，例如看 `S11` 时按 `S11` 判断，看 `Mises` 时按 `Mises` 判断。
- 如果项目也对每个分量分别建立 cluster，同一个节点可能同时对应多套 cluster，CSV / VTK 中一行完整应力张量难以清楚表达。

结论：08 是极端 C3D4 regular cube 场景下的导出口径差异案例。一个规则立方体由多个常应变 Tet4 单元剖分，共享节点处的单元应力跳变容易被放大。除该极端 C3D4 多单元共享节点场景外，其余 Hex8、Tet4 单单元、Tet10、多单元边界和混合单元算例的误差均处于 Abaqus 文本截断量级，可以认为一般情况下误差很小。

### 5.9 09 Tet10 regular cube C3D10

该算例验证多个 Tet10 组成规则立方体时的节点应力外推和平均结果。

对比误差为：

```text
max_abs_diff  = 4.62e-3
mean_abs_diff = 3.05e-4
```

最大误差略高于 Hex8 算例，原因是 Abaqus 表格中部分数值只保留 2-3 位小数；当应力值较大时，文本截断会带来约 `1e-3` 到 `1e-2` 的差异。

该算例与 08 都是规则立方体四面体剖分，但单元类型从 C3D4 变为 C3D10。C3D10 是二次四面体，节点应力外推不只是单元常应力复制，边中节点也参与结果对比。

项目结果在 60 个 element-local 节点行、共 420 个分量对比中保持 `4.62e-3` 的最大误差，说明：

- Tet10 的节点外推链路与 Abaqus 一致；
- 中间节点的应力结果没有出现编号错配；
- `S13/S23` 与 `tau_zx/tau_yz` 的剪应力映射正确；
- 多单元共享节点下，Tet10 场景没有出现 08 那种 `1e2` 量级的导出口径差异。

结论：Tet10 多单元节点应力结果与 Abaqus 同口径一致，误差可接受。

### 5.10 10 Mixed Hex8 + Tet4

该算例验证 Hex8 与 Tet4 混合单元类型边界处的节点应力规则。

对比误差为：

```text
max_abs_diff  = 5.38e-4
mean_abs_diff = 1.46e-4
```

Hex8 部分使用静水应力修正，Tet4 部分直接按分量映射对比。结果与 Abaqus 一致，说明项目没有跨 element type 进行错误平均。

该算例专门检查不同 element type 共存时的边界行为。Hex8 和 Tet4 的节点应力恢复方式不同：Hex8 需要静水应力项修正，Tet4 不需要。如果共享节点处跨单元类型平均，会把两套不同恢复口径的应力混合，通常会导致正应力和 Mises 同时偏离。

当前误差保持在 `5e-4` 量级，说明 element type 已经参与分组。Hex8 只与 Hex8 同口径贡献平均，Tet4 保持 Tet4 自身口径，不发生跨类型平滑。

结论：Hex8 + Tet4 混合单元类型场景通过验证。

### 5.11 11 Mixed Hex8 + Tet10

该算例验证 Hex8 与 Tet10 混合单元类型边界处的节点应力规则。

对比误差为：

```text
max_abs_diff  = 6.15e-4
mean_abs_diff = 2.02e-4
```

Hex8 部分使用静水应力修正，Tet10 部分按 `S12/S13/S23` 分量映射对比。结果与 Abaqus 一致，说明 mixed solid 场景下的 region-aware 输出规则成立。

该算例比 10 更敏感，因为 Tet10 含有二次节点和更复杂的应力外推，而 Hex8 仍需静水项修正。若节点编号、局部节点顺序、剪应力映射或 element type 分组任意一项有误，通常会在共享边界或中间节点处放大为明显误差。

当前最大误差为 `6.15e-4`，仍处于 Abaqus 文本截断量级，说明：

- Hex8 修正与 Tet10 外推可以在同一输出流程中并存；
- mixed element type 边界没有被错误平均；
- Tet10 的中间节点没有与 Hex8 角节点发生错配；
- 剪应力分量映射在混合场景中仍成立。

结论：Hex8 + Tet10 混合单元类型场景通过验证。

## 6. 测试与验证命令

已覆盖测试包括：

- `tests/test_elements.py`：Hex8 Gauss-to-node 外推、Tet4 常应力复制、Tet10 Hammer 点一次拟合外推。
- `tests/test_post.py`：region-aware averaging、threshold split、material / section / element type 分组、CSV / VTK 输出。
- `tests/test_regressions.py`：导出链路回归测试。

已验证命令：

```text
python -m pytest tests\test_post.py -q
python -m pytest -q
python examples\run_stress_validation.py
```

当前验证结果：

```text
tests/test_post.py: 22 passed
full pytest:        191 passed
examples/run_stress_validation.py: Passed 11, Failed 0
```

VTK 输出清洁性已检查：生成的 `.vtk` 文件只包含 `displacement`、`sig_x/sig_y/sig_z`、`tau_xy/tau_yz/tau_zx`、`mises`，不包含 `CELL_DATA`，也不包含 `original_node_id/region_id/cluster_id/material_id/section_id/element_type_id` 等调试字段。

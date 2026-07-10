# FEM Python

这是一个面向教学、实验和逐步扩展的有限元项目。项目保留清晰的有限元主流程：读取mesh，构建model，定义材料和分析步，装配刚度，施加载荷和约束，求解，导出后处理结果

## 安装和运行


```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

运行示例：

```powershell
python examples\cantilever_beam_hex8.py
python examples\cantilever_beam_hex8_abaqus.py
python examples\mixed_hex8_tet4.py
```

运行pytest测试：

```powershell
pip install -e ".[test]"
python -m pytest -q
python -m pytest -q tests/test_solvers.py
python -m pytest -q tests/test_solvers.py::test_static_linear_solver_builds_step_boundary_and_solves_case
```

## 当前能力

网格读取：

- `fem.io.inp`读取Abaqus inp中的mesh拓扑和坐标
- `fem.io.csv`读取CSV格式mesh
- `fem.abaqus`读取Abaqus inp中的完整模型数据

单元：

- `Truss2D`
- `Beam2D`
- `Tri3`
- `Quad4`
- `Quad8`
- `Hex8`
- `Hex20
- `Tet4`
- `Tet10`

求解：

- 稀疏全局刚度装配
- 支持`Hex8`、`Hex20`、`Tet4`和`Tet10`混合实体网格装配
- 支持各单元的节点力、边力、体力、重力装配
- 线性静力分析流程


后处理：

- 节点位移CSV
- 单元应力CSV
- 节点平均应力CSV
- VTK文件导出

## 模块职责

`core`是数据结构层，只保存模型结构，不负责装配、求解或导出

- `core.mesh`：节点、单元、mesh容器
- `core.dof`：节点到全局自由度的映射
- `core.model`：`FEMModel`、set、surface、材料定义、section、step和载荷声明
- `core.result`：求解结果数据

`io`是mesh读取层

- `io.inp`读取Abaqus inp中的网格数据
- `io.csv`读取CSV网格
- `io.materials`读取独立材料表

`abaqus`是Abaqus适配层

- `abaqus.parser`把inp解析成中间deck
- `abaqus.builder`把deck转换为`FEMModel`
- `abaqus.read()`是完整模型读取入口

`materials`是材料定义和赋值层

- `materials.linear_elastic`定义线弹性材料和本构矩阵
- `materials.assignment`把材料按element set赋给模型
- 求解前由`materials.apply_sections(model)`把section信息写入单元求解属性

`steps`是分析步声明层

- 创建`AnalysisStep`
- 向step加入位移约束、节点力、surface traction、surface pressure和输出请求

`boundary`是求解边界解析层

- `boundary.step`把`AnalysisStep`中的声明解析为求解器使用的`BoundaryCondition`
- `boundary.loads`从`BoundaryCondition`构造全局载荷向量
- `boundary.constraints`施加Dirichlet约束
- `boundary.body`、`boundary.nodal`、`boundary.traction`分别处理体力、节点力和边/面力

`elements`是单元kernel层

- 每类单元提供刚度矩阵、等效载荷和应力计算
- `elements.registry`负责按单元类型查找kernel
- `Truss2D`提供轴向应变/应力；`Beam2D`当前提供位移和刚度响应，截面力与弯曲应力导出仍需单独实现

`assemble`是全局装配层

- `assemble.stiffness`根据mesh和element kernel装配全局刚度矩阵

`solvers`是求解流程层

- `solvers.linear.solve()`求解稀疏线性方程组
- `solvers.static_linear.solve()`执行线性静力流程：材料赋值，step解析，装配，载荷向量，约束处理，线性求解，生成`ModelResult`

`post`是后处理层

- `post.displacement.export.nodal()`导出节点位移
- `post.stress.export.element()`导出单元应力
- `post.stress.export.nodal()`导出带阈值判断的节点应力
- `post.vtk.export.from_result()`从`ModelResult`导出CSV和VTK
- `post.vtk.export.from_csv()`从已有CSV导出VTK

`selection`是几何选择层

- `selection.nodes`按坐标筛选节点
- `selection.elements`筛选单元
- `selection.edges`筛选2D边
- `selection.faces`筛选3D面并生成surface

## 一般流程

一般流程从mesh开始，逐步补齐有限元求解所需的数据

```text
io.csv/io.inp
    -> mesh
    -> FEMModel(mesh=mesh)
    -> node_sets/element_sets/surfaces
    -> materials + sections
    -> AnalysisStep
    -> boundary.step.boundary_for_step(model, step)
    -> BoundaryCondition
    -> assemble.stiffness
    -> boundary.loads + boundary.constraints
    -> solvers.linear
    -> ModelResult
    -> post
```

这条链路中，各层职责如下：

- `mesh`保存节点、单元和自由度映射
- `FEMModel`保存mesh、set、surface、材料、section和step
- `AnalysisStep`保存一个分析阶段中的约束、载荷和输出请求
- `BoundaryCondition`保存解析到当前mesh后的全局DOF约束、全局节点力和单元局部边/面载荷
- `ModelResult`保存求解后的位移和反力
- `post`使用`mesh`、`U`、`ModelResult`或已有CSV，生成后处理文件


## 一般流程示例

一般流程示例在`examples/cantilever_beam_hex8.py`

这个脚本展示了：

- 用`fem.io.inp.read_hex8()`读取`examples/examples_data/cantilever_beam_hex8.inp`中的mesh。
- 创建`FEMModel`。
- 用`selection`构造node set和element set
- 用`materials.linear_elastic`定义材料
- 用`materials.assign()`把材料赋给element set
- 用`steps.static()`创建分析步
- 用`steps.displacement()`和`steps.nodal_load()`定义约束和载荷
- 用`solvers.static_linear.solve()`求解
- 用`post.vtk.export.from_result()`导出结果


## Abaqus流程示例

Abaqus流程示例在`examples/cantilever_beam_hex8_abaqus.py`

这个脚本展示了：

- 用`abaqus.read()`读取完整inp模型
- 从模型中取得分析步
- 用`solvers.static_linear.solve()`求解
- 用`post.vtk.export.from_result()`导出结果

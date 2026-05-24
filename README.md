# FEM Python

这是一个面向教学、实验和逐步扩展的有限元项目。项目保留清晰的有限元主流程：读取mesh，构建model，定义材料和分析步，装配刚度，施加载荷和约束，求解，导出后处理结果。

项目有两条使用路线：

- 一般流程：用`io.csv`或`io.inp`只读取mesh，然后显式组装`FEMModel`、材料、set、surface和step。
- Abaqus流程：用`abaqus.read()`读取inp中的节点、单元、set、surface、材料、section、step和载荷，直接构建`FEMModel`后求解。

一般流程是核心路线。`abaqus`模块用于把Abaqus输入文件转换到同一套项目数据结构，后续求解和后处理仍走统一流程。

## 安装和运行

当前仓库没有`pyproject.toml`或`setup.py`，运行示例前需要把`src`加入`PYTHONPATH`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

运行示例：

```powershell
python examples\cantilever_beam_hex8.py
python examples\cantilever_beam_hex8_abaqus.py
```

运行 unittest 测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover tests
python -m unittest tests.test_solvers
python -m unittest tests.test_solvers.StaticLinearSolverTests.test_static_linear_solver_builds_step_boundary_and_solves_case
```

Tests 已完成模块化迁移，常用测试位于 `tests/test_*.py`。旧入口 `tests/test_regressions.py` 暂时保留为兜底回归，因此 `discover tests` 会同时运行旧入口和新模块化测试；迁移对照表见 `docs/TEST_MIGRATION_MAP.md`。

## 当前能力

网格读取：

- `fem.io.inp`读取Abaqus inp中的mesh拓扑和坐标。
- `fem.io.csv`读取CSV格式mesh。
- `fem.abaqus`读取Abaqus inp中的完整模型数据。

单元：

- `Truss2D`
- `Beam2D`
- `Tri3`
- `Quad4`
- `Quad8`
- `Hex8`
- `Tet4`
- `Tet10`

求解：

- 稀疏全局刚度装配。
- 线性静力分析流程。
- `steps`声明入口支持位移约束、集中力、surface traction和surface pressure。
- 低层`BoundaryCondition`和element kernel支持体力、重力、边力和面力组装。

后处理：

- 节点位移CSV。
- 单元应力CSV。
- 节点平均应力CSV。
- VTK文件导出。

## 模块职责

`core`是数据结构层，只保存模型结构，不负责装配、求解或导出。

- `core.mesh`：节点、单元、mesh容器。
- `core.dof`：节点到全局自由度的映射。
- `core.model`：`FEMModel`、set、surface、材料定义、section、step和载荷声明。
- `core.result`：求解结果数据。

`io`是mesh读取层。

- `io.inp`只读取Abaqus inp中的网格数据。
- `io.csv`读取CSV网格。
- `io.materials`读取独立材料表。

`abaqus`是Abaqus适配层。

- `abaqus.parser`把inp解析成中间deck。
- `abaqus.builder`把deck转换为`FEMModel`。
- `abaqus.read()`是完整模型读取入口。

`materials`是材料定义和赋值层。

- `materials.linear_elastic`定义线弹性材料和本构矩阵。
- `materials.assignment`把材料按element set赋给模型。
- 求解前由`materials.apply_sections(model)`把section信息写入单元求解属性。

`steps`是分析步声明层。

- 创建`AnalysisStep`。
- 向step加入位移约束、节点力、surface traction、surface pressure和输出请求。

`boundary`是求解边界解析层。

- `boundary.step`把`AnalysisStep`中的声明解析为求解器使用的`BoundaryCondition`。
- `boundary.loads`从`BoundaryCondition`构造全局载荷向量。
- `boundary.constraints`施加Dirichlet约束。
- `boundary.body`、`boundary.nodal`、`boundary.traction`分别处理体力、节点力和边/面力。

`elements`是单元kernel层。

- 每类单元提供刚度矩阵、等效载荷和应力计算。
- `elements.registry`负责按单元类型查找kernel。

`assemble`是全局装配层。

- `assemble.stiffness`根据mesh和element kernel装配全局刚度矩阵。

`solvers`是求解流程层。

- `solvers.linear.solve()`求解稀疏线性方程组。
- `solvers.static_linear.solve()`执行线性静力流程：材料赋值，step解析，装配，载荷向量，约束处理，线性求解，生成`ModelResult`。

`post`是后处理层。

- `post.displacement.export.nodal()`导出节点位移。
- `post.stress.export.element()`导出单元应力。
- `post.stress.export.nodal()`导出节点平均应力。
- `post.vtk.export.from_result()`从`ModelResult`导出CSV和VTK。
- `post.vtk.export.from_csv()`从已有CSV导出VTK。

`selection`是几何选择层。

- `selection.nodes`按坐标筛选节点。
- `selection.elements`筛选单元。
- `selection.edges`筛选2D边。
- `selection.faces`筛选3D面并生成surface。

## 一般流程中数据流转

一般流程从mesh开始，逐步补齐有限元求解所需的数据。

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

- `mesh`保存节点、单元和自由度映射。
- `FEMModel`保存mesh、set、surface、材料、section和step。
- `AnalysisStep`保存一个分析阶段中的约束、载荷和输出请求。
- `BoundaryCondition`保存解析到当前mesh后的全局DOF约束、全局节点力和单元局部边/面载荷。
- `ModelResult`保存求解后的位移和反力。
- `post`消费`mesh`、`U`、`ModelResult`或已有CSV，生成后处理文件。

这个流转关系使新增能力可以落到明确模块中。新增单元主要改`elements`和`post`，新增材料主要改`materials`，新增载荷或约束主要改`steps`和`boundary`。

## 一般流程示例

一般流程示例在`examples/cantilever_beam_hex8.py`。

这个脚本展示了：

- 用`fem.io.inp.read_hex8()`读取`examples/cantilever_beam_hex8.inp`中的mesh。
- 创建`FEMModel`。
- 用`selection`构造node set和element set。
- 用`materials.linear_elastic`定义材料。
- 用`materials.assign()`把材料赋给element set。
- 用`steps.static()`创建分析步。
- 用`steps.displacement()`和`steps.nodal_load()`定义约束和载荷。
- 用`solvers.static_linear.solve()`求解。
- 用`post.vtk.export.from_result()`导出结果。

阅读和修改这个脚本时，可以把它作为新增功能的首选接入点。一般流程能跑通后，再考虑Abaqus适配。

## Abaqus流程示例

Abaqus流程示例在`examples/cantilever_beam_hex8_abaqus.py`。

这个脚本展示了：

- 用`abaqus.read()`读取完整inp模型。
- 从模型中取得分析步。
- 用`solvers.static_linear.solve()`求解。
- 用`post.vtk.export.from_result()`导出结果。

Abaqus流程适合复用已有inp文件。它的目标是把Abaqus文件转换为项目内部的`FEMModel`，求解和后处理继续使用相同核心模块。

当前`abaqus`模块只支持项目已有kernel能正确表达的单元和载荷语义。不支持的Abaqus单元变体应报错，等增加对应kernel后再支持。

## 如何增加新的element类型

新增单元的核心工作是增加element kernel。reader、装配、载荷和后处理只通过kernel或registry接入。

先确定3件事：

- mesh维度：2D单元使用`Element2D`和2D mesh，3D单元使用`Element3D`和3D mesh。
- 单元类型名：reader写入`elem.type`，registry用`type_names`匹配这个名字。
- 单元属性：材料、厚度、截面、面积等数据统一从`elem.props`读取。

必须完成：

1. 在`src/fem/elements/`中选择族模块。例如线单元放`line.py`，四边形放`quadrilateral.py`，四面体放`tetrahedron.py`，六面体放`hexahedron.py`。
2. 实现kernel类，提供`type_names`和`stiffness(mesh, elem, node_lookup=None)`。
3. 在`elements.registry`中调用`register_element_kernel(...)`注册kernel。
4. 更新`io.csv`或`io.inp`，让reader能创建对应`Element2D`或`Element3D`，并写入正确`elem.type`和`elem.props`。
5. 增加测试，至少覆盖刚度矩阵尺寸、自由度顺序、关键数值和异常输入。

按能力继续补充：

1. 支持体力时，在kernel中增加`body_force(mesh, elem, vector, node_lookup=None)`。
2. 支持2D边力时，增加`edge_traction(mesh, elem, local_index, vector, node_lookup=None)`。
3. 支持3D面力时，增加`face_traction(mesh, elem, local_index, vector, node_lookup=None)`。
4. 支持单元应力时，增加`element_stress(...)`或`stress_at(...)`。
5. 支持节点应力时，增加`nodal_stress(...)`或在`post.stress`中增加外推逻辑。

后处理接入点：

1. VTK需要新增单元类型映射时，更新`post.vtk.cells`。
2. 应力导出需要识别新类型时，更新`post.stress.dispatch`。
3. 若现有`post.stress.export.element()`或`post.stress.export.nodal()`无法直接消费新kernel，补充`post.stress.element`或`post.stress.nodal`中的分发逻辑。

Abaqus适配在一般流程可用之后再做。新增Abaqus单元时，更新`abaqus.parser`/`abaqus.builder`，把Abaqus类型名转换到项目内部`elem.type`。

## 如何增加新的boundary类型

新增边界时先明确它属于声明层还是求解输入层。

声明层面向用户和输入文件，写入`AnalysisStep`：

1. 在`core.model`中增加轻量数据结构，或扩展已有`DisplacementConstraint`、`NodalLoad`、`SurfaceLoad`。
2. 在`steps`包中增加便捷函数。函数实现放在`constraints.py`、`loads.py`或新的清晰模块中，`steps/__init__.py`只导出公共入口。
3. 如果条件依赖set或surface，声明里保存set名或surface名，不直接保存解析后的DOF。
4. 在`boundary.step`中解析`AnalysisStep`，把set、surface和component转换成`BoundaryCondition`。

求解输入层面向装配和线性求解，写入`BoundaryCondition`：

1. 在`boundary.condition`中增加求解器需要的数据结构。
2. 在`boundary.loads`或`boundary.constraints`中增加组装或约束处理。
3. 如果载荷需要单元积分，把积分公式放进对应element kernel。
4. 如果载荷依赖几何选择，优先复用`selection.nodes`、`selection.edges`和`selection.faces`生成set或surface。

Abaqus适配作为独立入口处理：

1. 在`abaqus.parser`中解析关键字。
2. 在`abaqus.builder`中把Abaqus对象转成`core.model`中的声明。
3. 复用`boundary.step`进入求解流程。

新增boundary不直接写进`solvers.static_linear`。solver只编排材料赋值、step解析、装配、约束、求解和结果生成。

## 如何增加新的materials类型

材料扩展要同时考虑材料定义、section赋值和kernel消费方式。

必须完成：

1. 在`src/fem/materials/`中新增材料模块，例如`orthotropic.py`或`plasticity.py`。
2. 提供材料工厂函数，返回`MaterialDefinition(name, properties)`。
3. 设计清晰的`properties`键名，element kernel只读取这些约定字段。
4. 在`materials/__init__.py`中导出新模块或便捷函数。
5. 用`materials.add(model, material)`把材料放入`model.materials`。
6. 用`materials.assign(model, material=..., element_set=...)`把材料赋给element set。
7. 确认`materials.apply_sections(model)`能把section信息写入目标单元的`elem.props`。

如果材料需要本构矩阵：

1. 在线性材料模块中提供矩阵函数。
2. 在需要该材料的element kernel中显式调用它。
3. 不让kernel根据字段缺省值猜材料模型。

如果材料来自文件：

1. 独立CSV材料表更新`io.materials`。
2. Abaqus材料更新`abaqus.parser`和`abaqus.builder`。
3. reader或builder只负责转换数据，不负责求解。

当前线弹性材料入口在`src/fem/materials/linear_elastic.py`，调用示例见`examples/cantilever_beam_hex8.py`。

## 如何增加新的后处理方式

后处理按结果类型分包。公共调用风格保持`post.<result_type>.export.<function>()`。

现有入口：

- `post.displacement.export.nodal(mesh, U, path)`
- `post.stress.export.element(mesh, U, path)`
- `post.stress.export.nodal(mesh, U, path)`
- `post.vtk.export.from_result(result, output_dir)`
- `post.vtk.export.from_csv(mesh, disp_csv_path, elem_csv_path, vtk_path, nodal_stress_csv_path=None)`

新增一种结果类型：

1. 创建`src/fem/post/<name>/`包。
2. 新增`export.py`，把文件导出函数放在这里。
3. 在`post/<name>/__init__.py`中导出`export`。
4. 在`post/__init__.py`中导出新子包。
5. 函数输入优先使用`mesh`、`U`、`ModelResult`或已有CSV。

扩展VTK：

1. 新字段读取或转换放在`post.vtk.fields`。
2. 新VTK单元拓扑放在`post.vtk.cells`。
3. 写文件细节放在`post.vtk.writer`。
4. 从结果自动生成CSV和VTK的流程放在`post.vtk.export`。

后处理不修改`FEMModel`、`mesh`或`elem.props`。需要求解前数据时，在solver或材料赋值阶段准备好。

## 推荐开发顺序

新增一个有限元能力时，按这个顺序推进：

1. 明确它属于哪个层：core数据、element kernel、material、boundary、solver还是post。
2. 先补最小测试，固定期望行为。
3. 新增或修改数据结构。
4. 实现底层计算逻辑。
5. 接入registry或dispatch。
6. 更新reader或builder。
7. 更新example或README。

对于大功能，优先保证一般流程可用，再考虑Abaqus适配。这样可以避免把Abaqus解析细节误认为核心架构。

## 重要约定

- `core`只放数据结构，不放装配、求解、导出方法。
- `steps`包负责创建和填充`AnalysisStep`，`AnalysisStep`保存分析步声明。
- `boundary`保存求解器输入解析逻辑。
- `abaqus`只负责适配输入文件，不承担核心求解职责。
- `solvers.static_linear`负责流程编排，但不直接实现单元积分或set/surface选择。
- 新增功能优先通过明确模块和dispatch/registry接入，不恢复旧式大函数入口。

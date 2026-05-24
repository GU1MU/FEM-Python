# unittest 测试模块化迁移对照表

本文档记录 `tests/test_regressions.py` 中旧单文件 unittest 回归测试，在模块化测试结构中的归属与当前迁移状态。

## 当前状态

- 任务状态：Tests 模块重构已完成实体迁移。
- 旧回归入口：`tests/test_regressions.py`
- 旧测试数量：79
- 新模块化 unittest 测试数量：79
- `python -m unittest discover tests` 会同时运行旧入口和新模块化测试，因此当前总数为 158。这是因为旧入口仍保留，并不是迁移遗漏。
- 本阶段不删除旧 `tests/test_regressions.py`，它继续作为兜底回归测试。
- 如果后续要让全量 discover 只运行 76 个模块化测试，需要删除、改名或排除 `tests/test_regressions.py`。

## 当前结构

```text
tests/
  helpers/
    file_builders.py
    mesh_builders.py
    model_builders.py
    result_builders.py
  test_core.py
  test_elements.py
  test_assemble.py
  test_boundary.py
  test_solvers.py
  test_post.py
  test_io.py
  test_abaqus.py
  test_integration.py
  test_regressions.py
```

## 实体迁移进度

| 新测试文件 | 数量 | 当前状态 | 备注 |
|---|---:|---|---|
| `tests/test_core.py` | 15 | 已实体迁移 | core / dof / selection / materials |
| `tests/test_solvers.py` | 6 | 已实体迁移 | linear solver 与 static linear 流程 |
| `tests/test_assemble.py` | 13 | 已实体迁移 | dense/sparse 全局刚度装配 |
| `tests/test_boundary.py` | 3 | 已实体迁移 | boundary load vector 与包入口 |
| `tests/test_io.py` | 2 | 已实体迁移 | io 包入口与 inp mesh reader 职责 |
| `tests/test_post.py` | 8 | 已实体迁移 | 使用临时目录隔离 CSV/VTK 输出 |
| `tests/test_integration.py` | 2 | 已实体迁移 | 手写 FEMModel 到 static solver 的完整流程 |
| `tests/test_abaqus.py` | 12 | 已实体迁移 | 使用临时目录隔离 inp 文件 |
| `tests/test_elements.py` | 18 | 已实体迁移 | line / plane / solid element kernel |

已实体迁移的新模块测试数量：79。

## 公共测试构造

重复的测试数据构造已经集中到 `tests/helpers/`：

| helper 文件 | 职责 |
|---|---|
| `tests/helpers/mesh_builders.py` | 构造 truss、beam、tri3、quad4、quad8、hex8、tet4、tet10 等测试 mesh |
| `tests/helpers/model_builders.py` | 构造带材料、section、step、边界和载荷的测试 model / workflow |
| `tests/helpers/result_builders.py` | 构造后处理测试需要的 result 对象 |
| `tests/helpers/file_builders.py` | 写入临时 inp 等文件型测试数据 |

## 运行命令

只运行新模块化测试：

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_core tests.test_elements tests.test_assemble tests.test_boundary tests.test_solvers tests.test_post tests.test_io tests.test_abaqus tests.test_integration
```

运行当前仓库全部可发现测试。注意：因为旧入口仍保留，这条命令当前会运行 158 个测试，其中 79 个来自新模块化文件，79 个来自旧 `test_regressions.py`：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover tests
```

单独运行旧回归入口：

```powershell
python -m unittest tests.test_regressions
```

单独运行某个新模块：

```powershell
python -m unittest tests.test_core
python -m unittest tests.test_solvers
python -m unittest tests.test_assemble
python -m unittest tests.test_boundary
python -m unittest tests.test_io
python -m unittest tests.test_post
python -m unittest tests.test_abaqus
python -m unittest tests.test_elements
python -m unittest tests.test_integration
```

## 旧测试归属清点

| 旧测试类 | 测试数量 | 新文件 | 新测试类 | 状态 |
|---|---:|---|---|---|
| `DofMapRegressionTests` | 7 core + 3 solver + 1 integration | `test_core.py` / `test_solvers.py` / `test_integration.py` | `DofMapCoreTests` / `StaticLinearSolverTests` / `ModelWorkflowIntegrationTests` | 已实体迁移 |
| `SelectionRegressionTests` | 6 | `test_core.py` | `SelectionTests` | 已实体迁移 |
| `MaterialsRegressionTests` | 2 | `test_core.py` | `MaterialsTests` | 已实体迁移 |
| `SolverRegressionTests` | 3 | `test_solvers.py` | `LinearSolverTests` | 已实体迁移 |
| `LineElementKernelRegressionTests` | 3 elements + 4 assemble | `test_elements.py` / `test_assemble.py` | `LineElementKernelTests` / `LineAssemblyTests` | 已实体迁移 |
| `Quad4StiffnessRegressionTests` | 2 elements + 2 assemble | `test_elements.py` / `test_assemble.py` | `Quad4ElementKernelTests` / `PlaneAssemblyTests` | 已实体迁移 |
| `PlaneElementKernelRegressionTests` | 2 elements + 2 assemble | `test_elements.py` / `test_assemble.py` | `PlaneElementKernelTests` / `PlaneAssemblyTests` | 已实体迁移 |
| `BoundaryKernelRegressionTests` | 3 elements + 3 boundary | `test_elements.py` / `test_boundary.py` | `PlaneLoadKernelTests` / `BoundaryKernelTests` | 已实体迁移 |
| `PlaneStressKernelRegressionTests` | 1 | `test_elements.py` | `PlaneStressKernelTests` | 已实体迁移 |
| `Hex8LoadRegressionTests` | 3 elements + 2 assemble | `test_elements.py` / `test_assemble.py` | `Hex8KernelTests` / `SolidAssemblyTests` | 已实体迁移 |
| `TetElementKernelRegressionTests` | 2 elements + 2 assemble | `test_elements.py` / `test_assemble.py` | `TetKernelTests` / `SolidAssemblyTests` | 已实体迁移 |
| `SolidStressKernelRegressionTests` | 1 | `test_elements.py` | `SolidStressKernelTests` | 已实体迁移 |
| `VtkExportRefactorTests` | 1 | `test_post.py` | `VtkExportTests` | 已实体迁移 |
| `IoPackageRefactorTests` | 2 | `test_io.py` | `IoPackageTests` | 已实体迁移 |
| `AbaqusModelReaderTests` | 12 | `test_abaqus.py` | `AbaqusModelReaderTests` | 已实体迁移 |
| `PostPackageRefactorTests` | 7 | `test_post.py` | `PostPackageTests` | 已实体迁移 |
| `ElementsPackageRefactorTests` | 1 | `test_elements.py` | `ElementsPackageTests` | 已实体迁移 |
| `StiffnessModuleRemovalTests` | 1 | `test_assemble.py` | `StiffnessModuleRemovalTests` | 已实体迁移 |
| `ManualWorkflowRefactorTests` | 1 | `test_integration.py` | `ModelWorkflowIntegrationTests` | 已实体迁移 |

## 后续建议

1. 所有新模块测试已经实体迁移完成。
2. 下一步可以先让新模块化测试稳定运行一段时间。
3. 确认没有遗漏后，再考虑删除 `tests/test_regressions.py`。

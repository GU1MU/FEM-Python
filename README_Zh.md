# FEM Python

[English](README.md)

FEM Python 是一个用于脚本化有限元建模与线性静力分析的 Python 项目
项目覆盖几何建模、网格生成或导入、模型定义、求解、结果查询以及 CSV/VTK 导出

支持两类主要入口：

- 使用 OCC/Gmsh 在 Python 中创建几何和网格
- 将受支持的 Abaqus `.inp` 内容构建为 `FEMModel`

## 安装

需要 Python 3.13 或更高版本，以下命令在仓库根目录执行，以
PowerShell 为例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[cad]"
```

`cad` 可选依赖用于 OCC/Gmsh 建模和网格功能

## 示例

仓库保留三个代表性示例：

| 示例 | 内容 |
| --- | --- |
| [`frame.py`](examples/frame.py) | Beam2 刚架、OCC 建模、Gmsh 自动网格、固定约束、多静力工况 |
| [`perforated_plate.py`](examples/perforated_plate.py) | 一次 Tri3 平面应力开孔板、圆孔曲线选择、局部网格加密和边牵引 |
| [`cantilever_beam.py`](examples/cantilever_beam.py) | 导入 Abaqus `.inp` 中的二次 Hex20 悬臂梁、面载荷和重力 |

运行示例：

```powershell
.\.venv\Scripts\python.exe examples\frame.py
.\.venv\Scripts\python.exe examples\perforated_plate.py
.\.venv\Scripts\python.exe examples\cantilever_beam.py
```

结果默认写入 `results/`

## 工作流程

```text
OCC/Gmsh 建模或 Abaqus .inp 导入
    → FEMModel
    → 集合、材料与截面
    → AnalysisStep
    → static_linear.solve()
    → ModelResult / ModelResults
    → CSV / VTK
```

求解一个工况时返回 `ModelResult`；使用 `steps=` 求解一组工况时返回可迭代的 `ModelResults`

## 当前能力

| 类别 | 支持内容 |
| --- | --- |
| 几何与网格 | OCC 几何与布尔运算、CAD 实体选择、Gmsh 自动网格与局部加密、Gmsh/CSV 网格读取 |
| 模型导入 | 将受支持的 Abaqus `.inp` 关键字和分析数据构建为 `FEMModel` |
| 一维单元 | `Truss2`、`Beam2` |
| 二维单元 | `Tri3`、`Tri6`、`Quad4`、`Quad8`，支持平面应力和平面应变 |
| 三维单元 | `Tet4`、`Tet10`、`Hex8`、`Hex20` |
| 材料 | 各向同性线弹性材料 |
| 约束与载荷 | 零或非零规定位移、节点力、Beam2 线载荷、二维边牵引/压力、三维面牵引/压力、重力 |
| 求解 | 稀疏装配、单个或多个独立线性静力工况、共享 `Initial` 约束 |
| 后处理 | 节点位移、反力、按单元类型提供的单元或节点应力、Beam2 应力包络、CSV 和 VTK |

Beam2 支持圆形、空心圆形和矩形截面

## 能力边界

- 当前求解范围为各向同性线弹性、小变形、线性静力分析
- Abaqus 适配层只解析项目明确支持的关键字和分析数据
- 项目不强制单位制，也不执行单位换算；所有输入必须使用一致单位

## 测试

安装测试依赖并运行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[cad,test]"
.\.venv\Scripts\python.exe -m pytest -q
```

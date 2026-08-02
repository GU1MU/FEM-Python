# FEM Python

FEM Python 是一个用于脚本化有限元建模与线性静力分析的 Python 项目

项目覆盖几何建模、网格生成或导入、模型定义、求解、结果查询以及 CSV/VTK 导出

项目同时提供 Python API 和桌面 GUI，GUI 内集成 FEM Agent，模型既可自主创建，也可从受支持的 Abaqus `.inp` 文件导入

## 安装

需要 Python 3.13 或更高版本和 `uv`，以下命令均在仓库根目录的 PowerShell 中执行

创建虚拟环境：

```powershell
uv venv --python 3.13
```

安装桌面 GUI、FEM Agent 以及网格功能：

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[cad,gui,agent]"
```

如果只使用 Python API 和示例，可以仅安装网格依赖：

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[cad]"
```

## 桌面 GUI

### 启动

激活虚拟环境并启动 GUI：

```powershell
.\.venv\Scripts\Activate.ps1
fem-gui
```

### 功能

桌面 GUI 将项目管理、前处理、分析和后处理组织在同一会话中

- 新建、打开和保存自主项目，或导入 Abaqus `.inp` 模型
- 创建和编辑几何、作用域、材料、截面及分析定义
- 设置并生成网格，完成质量评估、模型检查和后台分析
- 显示、查询和导出结果

## FEM Agent

### 配置

在仓库根目录创建 `fem-agent.config.json`：

```json
{
  "enabled": true,
  "api_key": "your-api-key"
}
```

启动 GUI 后，点击模型视口右上角的 `FA` 按钮打开 FEM Agent

### 功能

FEM Agent 根据当前 GUI 会话和用户提供的工程参数，协助完成受支持的自主建模与分析流程

- 读取当前模型状态，并可通过 `@` 引用工作区文件
- 创建或修改几何、网格
- 定义并赋予材料、分析步及边界条件
- 提交求解并查询分析结果

## Python API 示例

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
| 求解 | 稀疏装配、单个或多个独立线性静力工况 |
| 后处理 | 节点位移、反力、单元或节点应力、CSV 和 VTK 导出 |

Beam2 支持圆形、空心圆形和矩形截面

## 能力边界

- 当前求解范围为各向同性线弹性、小变形、线性静力分析
- Abaqus 适配层只解析项目明确支持的关键字和分析数据
- 项目不强制单位制，也不执行单位换算；所有输入必须使用一致单位

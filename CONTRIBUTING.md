# 贡献指南 (Contributing Guide)

感谢你对 NeuroGenesis Lab 的关注！本文档描述了如何参与项目开发。

## 开发环境搭建

### 一步搭建（推荐）

克隆仓库后，从干净状态一步搭好环境（脚本会在 `.venv` 不存在时自动创建）：

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# Linux / macOS
bash scripts/bootstrap.sh
```

默认会构建 Rust 扩展并执行运行时一致性检查。若本机没有 Rust 工具链，
可只搭建纯 Python 开发环境：

```bash
# Windows
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -SkipRust

# Linux / macOS
bash scripts/bootstrap.sh --skip-rust
```

如需手动逐步搭建，可按以下步骤操作。

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/neurogenesis-lab.git
cd neurogenesis-lab
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. 安装开发依赖

```bash
pip install -e ".[dev,viz]"
```

### 4. （可选）安装 Rust 扩展

```bash
pip install -e ".[rust]"
python -m maturin develop --release
```

## 分支策略

- `main` — 稳定发布分支，仅通过 PR 合并
- `dev` — 日常开发分支
- 功能分支命名：`feature/<描述>`（例如 `feature/vectorized-corpse-pool`）
- 修复分支命名：`fix/<描述>`（例如 `fix/checkpoint-migration`）

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]
```

常用类型：
- `feat` — 新功能
- `fix` — 修复 Bug
- `refactor` — 重构（不改变外部行为）
- `test` — 测试相关
- `docs` — 文档更新
- `ci` — CI/CD 变更
- `chore` — 构建/工具相关

示例：
```
feat(vectorized): 实现向量化尸体回收系统
fix(network): 修复 checkpoint 加载时的字段缺失
test(learning): 补充 RuleComposer 边界测试
```

## 测试要求

- 所有 PR 必须通过 CI（测试 + lint + 覆盖率）
- 覆盖率要求 ≥ 90%
- 新功能必须附带对应的单元测试
- 运行测试：

```bash
# 运行全部测试
pytest

# 运行特定测试
pytest tests/test_network.py

# 查看覆盖率报告
pytest --cov=src/neurogenesis_lab --cov-report=html
```

## 代码风格

- 使用 [ruff](https://docs.astral.sh/ruff/) 进行 lint 和格式化
- 行宽上限：100 字符
- 类型标注要求：所有公开 API 必须有完整的类型标注
- Docstring 风格：Google style

```bash
# 检查代码风格
ruff check src/

# 自动格式化
ruff format src/

# 类型检查
mypy src/
```

## 项目结构

```
src/neurogenesis_lab/
├── cli/           # CLI 子包（命令行接口）
├── config.py      # 配置系统
├── neuron.py      # 神经元模型
├── network.py     # 标准网络引擎
├── vectorized.py  # 向量化网络引擎
├── learning.py    # 学习引擎
├── encoding.py    # 编码系统
├── observability.py  # 可观测性
├── protocols.py   # 统一接口协议
└── ...
```

## 问题反馈

- 使用 GitHub Issues 报告 Bug 或提出功能建议
- Bug 报告请包含：复现步骤、期望行为、实际行为、环境信息
- 功能建议请描述：用例场景、期望效果、可能的实现思路

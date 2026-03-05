# 分支保护规则配置指南

本文档说明如何为 NeuroGenesis Lab 仓库配置分支保护规则，确保所有合并到主分支的代码都通过 CI 检查。

## 配置步骤

### 1. 访问仓库设置

1. 打开 GitHub 仓库页面
2. 点击 **Settings**（设置）标签
3. 在左侧菜单中找到 **Branches**（分支）

### 2. 添加分支保护规则

1. 点击 **Add branch protection rule**（添加分支保护规则）
2. 在 **Branch name pattern** 中输入：`main`（或 `master`，取决于你的主分支名称）

### 3. 配置保护选项

勾选以下选项：

#### 必需的状态检查（Required status checks）

- ✅ **Require status checks to pass before merging**（合并前要求状态检查通过）
  - ✅ **Require branches to be up to date before merging**（合并前要求分支是最新的）
  - 在搜索框中添加以下必需的检查：
    - `Lint & Format Check` (来自 lint.yml)
    - `Test (Python 3.10, ubuntu-latest)` (来自 python-test.yml)
    - `Test (Python 3.11, ubuntu-latest)` (来自 python-test.yml)
    - `Test (Python 3.12, ubuntu-latest)` (来自 python-test.yml)
    - `Test (Python 3.10, windows-latest)` (来自 python-test.yml)
    - `Test (Python 3.11, windows-latest)` (来自 python-test.yml)
    - `Test (Python 3.12, windows-latest)` (来自 python-test.yml)

#### 其他推荐选项

- ✅ **Require a pull request before merging**（合并前要求 Pull Request）
  - **Required approvals**: 1（对于个人项目可选，团队项目推荐）
  - ✅ **Dismiss stale pull request approvals when new commits are pushed**（新提交时撤销旧审批）

- ✅ **Require conversation resolution before merging**（合并前要求解决所有对话）

- ✅ **Do not allow bypassing the above settings**（不允许绕过上述设置）

### 4. 保存规则

点击页面底部的 **Create**（创建）或 **Save changes**（保存更改）按钮。

## 验证配置

配置完成后：

1. 创建一个新分支并进行修改
2. 提交 Pull Request
3. 确认 PR 页面显示所有必需的状态检查
4. 在所有检查通过前，**Merge** 按钮应该是禁用状态
5. 所有检查通过后，才能合并

## 当前 CI 工作流

NeuroGenesis Lab 配置了以下 CI 工作流：

| 工作流 | 文件 | 触发条件 | 检查内容 |
|--------|------|----------|----------|
| Python Tests | `python-test.yml` | push/PR to main | 多平台×多版本测试，覆盖率≥90% |
| Lint | `lint.yml` | push/PR to main | ruff check + format + mypy |
| Coverage Report | `coverage.yml` | PR to main | 覆盖率报告（PR 评论） |
| Docs | `docs.yml` | push to main | 文档构建与部署 |
| Rust Runtime Check | `rust-runtime-check.yml` | push/PR | Rust 扩展运行时检查 |
| Rust Wheel | `rust-wheel.yml` | push/PR | Rust wheel 构建 |

## 注意事项

1. **首次配置**：如果这是第一次配置分支保护，需要先运行一次 CI 让 GitHub 识别所有的状态检查名称
2. **状态检查名称**：必须与 workflow 文件中的 `jobs.<job_id>.name` 完全匹配
3. **个人仓库**：如果是个人开发，可以给自己的账号添加 bypass 权限，但不推荐
4. **紧急修复**：管理员可以临时禁用保护规则进行紧急修复，修复后应立即重新启用

## 故障排查

### 问题：找不到状态检查选项

**解决方案**：
1. 确保至少有一次 PR 或 push 触发了 CI
2. 等待 CI 运行完成
3. 刷新分支保护规则页面

### 问题：某些检查一直不通过

**解决方案**：
1. 检查 Actions 标签页查看失败原因
2. 本地运行 `pytest` 和 `ruff check src/` 确保代码质量
3. 修复问题后重新推送

### 问题：需要临时绕过保护规则

**解决方案**：
1. 不推荐绕过，应该修复 CI 失败
2. 如果确实紧急，管理员可以临时禁用规则
3. 修复后立即重新启用保护

## 相关文档

- [GitHub 分支保护文档](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [NeuroGenesis Lab CONTRIBUTING.md](../CONTRIBUTING.md)
- [开发清单07](../docs/开发清单07.md)

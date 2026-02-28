# learning — 学习引擎

混合学习系统：符号规则发现 + 神经动力学预测。

## 核心类

- **`AutoLearner`** — 自动学习包装器，集成仿真流与学习引擎，提供预测奖励/惩罚
- **`RuleDiscovery`** — 基于 trigram/bigram 的规则发现
- **`RuleStore`** — LRU OrderedDict 规则库
- **`RuleComposer`** — 规则组合与缺失 token 填充
- **`PatternMiner`** — 表达式模式挖掘
- **`QueryEngine`** — 表达式查询接口
- **`MemoryBank`** — 表达式存储

::: neurogenesis_lab.learning

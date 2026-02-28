# encoding — 编码系统

Token 与 Event 编码系统，用于将外部输入转换为脉冲神经网络可处理的事件。

## 核心类

- **`Event`** — 事件数据类（token_id, t, src_id, meta）
- **`Sequence`** — Token 序列
- **`Vocabulary`** — 双向符号映射
- **`Tokenizer`** — 文本→Token 编码器
- **`Encoder`** — 表达式→Event 编码器

::: neurogenesis_lab.encoding

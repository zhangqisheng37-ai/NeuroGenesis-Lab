# protocols — 协议接口

双引擎统一接口定义。

## 类型定义

- **`TickSnapshot`** (TypedDict) — `tick()` 返回的统一快照类型
- **`EngineSummary`** (TypedDict) — `get_summary()` 返回的统一摘要类型
- **`SimEngine`** (Protocol, runtime_checkable) — 仿真引擎协议，约束 `tick()` 和 `get_summary()`

`Network` 和 `VectorizedNetwork` 均实现此协议。

::: neurogenesis_lab.protocols

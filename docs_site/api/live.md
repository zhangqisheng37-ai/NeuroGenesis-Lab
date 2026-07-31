# live — 实时可视化 (WebSocket)

Phase 8 / P0 的实时交互式可视化能力，由三部分组成：协议模块 `live_protocol`、
FastAPI 服务器 `server`、自包含前端页面 `web/live.html`。P5 Web 管理界面复用
同一套 WebSocket 协议与 FastAPI 应用扩展 REST 端点。

安装可选依赖：

```bash
pip install "neurogenesis-lab[web]"   # fastapi + uvicorn
```

## CLI 用法

```bash
# 边跑边看：在 run 中内嵌实时监控并自动打开浏览器
python -m neurogenesis_lab run --live-monitor --port 8765

# 独立服务模式：流式推送一次仿真，结束后保持服务器供持续查看
python -m neurogenesis_lab serve --port 8765 --neurons 12

# 常用开关
#   --host / --port         绑定地址与端口
#   --ui-port               浏览器打开的端口（反向代理场景）
#   --live-tick-ms          每 tick 推送间隔（调速，毫秒）
#   --no-browser            不自动打开浏览器
#   --wait-for-client       启动后等待浏览器连接再开始仿真
#   --client-wait-timeout   等待客户端的超时秒数
```

浏览器端支持：实时渲染神经元网络（位置/能量/放电高亮）、种群指标面板、
暂停 / 继续 / 停止 / 调速控制。

## 消息协议

| 方向 | 类型 | 说明 |
| --- | --- | --- |
| 服务端 → 客户端 | `init` | 连接后首个消息，含配置与神经元总数 |
| 服务端 → 客户端 | `tick_update` | 每 tick 的神经元、连接与统计 |
| 服务端 → 客户端 | `status` | 运行状态变更（running/paused/finished） |
| 客户端 → 服务端 | `pause` / `resume` / `stop` | 仿真控制 |
| 客户端 → 服务端 | `speed_change` | 调速（`tick_ms`） |

## live_protocol 模块

::: neurogenesis_lab.live_protocol

## server 模块

::: neurogenesis_lab.server

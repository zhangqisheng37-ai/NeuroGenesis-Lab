/**
 * NeuroGenesis Lab — P5 实时监控 D3.js 可视化与控制逻辑
 *
 * 从 live.html 提取的核心渲染逻辑，封装为 NetworkMonitor 类，
 * 通过 WebSocket 接收仿真数据并实时更新力导向网络图与统计面板。
 */

"use strict";

const STAT_FIELDS = [
  ["tick", "Tick"],
  ["alive_neurons", "存活"],
  ["total_neurons", "总数"],
  ["tick_spikes", "放电/tick"],
  ["births", "出生"],
  ["deaths", "死亡"],
  ["mean_energy", "平均能量"],
  ["prediction_error", "预测误差"],
  ["rule_count", "规则数"],
  ["prediction_accuracy", "预测准确率"],
];

/**
 * 实时监控主类 — 管理 WebSocket 连接、D3 渲染与控制交互。
 */
class NetworkMonitor {
  constructor() {
    /** @type {WebSocket|null} */
    this.ws = null;
    /** @type {object} */
    this.stats = {};
    /** @type {string} */
    this.simState = "idle";

    // D3 SVG 初始化
    this.svg = d3.select("#network-canvas");
    this.gLinks = this.svg.append("g").attr("stroke", "#2b3040").attr("stroke-opacity", 0.5);
    this.gNodes = this.svg.append("g");
    this.width = 0;
    this.height = 0;
    this.xScale = null;
    this.yScale = null;

    this._initSVG();
    this._initControls();
  }

  /** 初始化 SVG 画布尺寸与窗口监听。 */
  _initSVG() {
    const wrap = document.querySelector(".monitor-canvas-wrap");
    if (!wrap) return;
    this.width = wrap.clientWidth;
    this.height = wrap.clientHeight;
    this.svg.attr("viewBox", `0 0 ${this.width} ${this.height}`);
    window.addEventListener("resize", () => {
      this.width = wrap.clientWidth;
      this.height = wrap.clientHeight;
      this.svg.attr("viewBox", `0 0 ${this.width} ${this.height}`);
    });
  }

  /** 绑定控制面板按钮事件。 */
  _initControls() {
    const btnPause = document.getElementById("btn-pause");
    const btnResume = document.getElementById("btn-resume");
    const btnStop = document.getElementById("btn-stop");
    const speed = document.getElementById("speed");

    if (btnPause) btnPause.onclick = () => { this._sendCommand("pause"); this._setStatus("paused"); };
    if (btnResume) btnResume.onclick = () => { this._sendCommand("resume"); this._setStatus("running"); };
    if (btnStop) btnStop.onclick = () => this._sendCommand("stop");
    if (speed) {
      speed.oninput = () => {
        const ms = parseInt(speed.value, 10);
        document.getElementById("speed-val").textContent = ms;
        this._sendCommand("speed_change", { tick_ms: ms });
      };
    }
  }

  /**
   * 建立 WebSocket 连接。
   * @param {string} url — WebSocket 地址（如 ws://localhost/ws）
   */
  connect(url) {
    if (this.ws) {
      this.ws.close();
    }
    this.ws = new WebSocket(url);
    const connEl = document.getElementById("conn-state");

    this.ws.onopen = () => {
      if (connEl) { connEl.textContent = "已连接"; connEl.className = "conn-indicator online"; }
    };

    this.ws.onclose = () => {
      if (connEl) { connEl.textContent = "连接断开"; connEl.className = "conn-indicator offline"; }
      // 自动重连
      setTimeout(() => this.connect(url), 1500);
    };

    this.ws.onerror = () => {
      if (connEl) { connEl.textContent = "连接错误"; connEl.className = "conn-indicator offline"; }
    };

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        this._handleMessage(msg);
      } catch (e) {
        console.warn("消息解析失败:", e);
      }
    };
  }

  /**
   * 根据消息类型分发处理。
   * @param {object} msg
   */
  _handleMessage(msg) {
    if (msg.type === "tick_update") {
      this._updateNetwork(msg.neurons || [], msg.connections || []);
      this._updateStats(msg.stats || {});
    } else if (msg.type === "status") {
      this._setStatus(msg.state);
      if (typeof msg.tick_ms === "number") {
        const speedEl = document.getElementById("speed");
        const speedValEl = document.getElementById("speed-val");
        if (speedEl) speedEl.value = msg.tick_ms;
        if (speedValEl) speedValEl.textContent = msg.tick_ms;
      }
    } else if (msg.type === "init") {
      // 初始化消息 — 可在此显示配置信息
      console.log("仿真初始化:", msg.total_neurons, "个神经元");
    }
  }

  /**
   * 更新 D3 网络可视化。
   * @param {Array} neurons — 神经元列表
   * @param {Array} connections — 连接列表
   */
  _updateNetwork(neurons, connections) {
    if (!neurons.length) return;

    // 更新坐标比例尺
    const xs = neurons.map((n) => n.x);
    const ys = neurons.map((n) => n.y);
    const pad = 40;
    const xExtent = d3.extent(xs);
    const yExtent = d3.extent(ys);
    if (xExtent[0] === xExtent[1]) { xExtent[0] -= 1; xExtent[1] += 1; }
    if (yExtent[0] === yExtent[1]) { yExtent[0] -= 1; yExtent[1] += 1; }
    this.xScale = d3.scaleLinear().domain(xExtent).range([pad, this.width - pad]);
    this.yScale = d3.scaleLinear().domain(yExtent).range([pad, this.height - pad]);

    const byId = new Map(neurons.map((n) => [n.id, n]));

    // 更新连接线
    const links = connections.filter(
      (c) => byId.has(c.source) && byId.has(c.target)
    );
    const lineSel = this.gLinks
      .selectAll("line")
      .data(links, (d) => d.source + "->" + d.target);
    lineSel.exit().remove();
    lineSel
      .enter()
      .append("line")
      .merge(lineSel)
      .attr("x1", (d) => this.xScale(byId.get(d.source).x))
      .attr("y1", (d) => this.yScale(byId.get(d.source).y))
      .attr("x2", (d) => this.xScale(byId.get(d.target).x))
      .attr("y2", (d) => this.yScale(byId.get(d.target).y));

    // 更新神经元节点
    const nodeSel = this.gNodes
      .selectAll("circle")
      .data(neurons, (d) => d.id);
    nodeSel.exit().remove();
    nodeSel
      .enter()
      .append("circle")
      .attr("r", 6)
      .merge(nodeSel)
      .attr("cx", (d) => this.xScale(d.x))
      .attr("cy", (d) => this.yScale(d.y))
      .attr("r", (d) => (d.spiked ? 9 : d.alive ? 6 : 4))
      .attr("fill", (d) =>
        !d.alive ? "#3b4252" : d.spiked ? "#f59e0b" : this._energyColor(d.energy)
      )
      .attr("stroke", (d) => (d.spiked ? "#fde68a" : "none"))
      .attr("stroke-width", (d) => (d.spiked ? 2 : 0));
  }

  /**
   * 根据能量值映射颜色。
   * @param {number} energy
   * @returns {string} D3 插值颜色
   */
  _energyColor(energy) {
    const t = Math.max(0, Math.min(1, energy / 3));
    return d3.interpolateViridis(0.25 + t * 0.6);
  }

  /**
   * 更新统计面板。
   * @param {object} stats
   */
  _updateStats(stats) {
    this.stats = stats;
    const box = d3.select("#stats");
    const sel = box.selectAll(".stat").data(STAT_FIELDS, (d) => d[0]);
    const enter = sel.enter().append("div").attr("class", "stat");
    enter.append("div").attr("class", "stat-label").text((d) => d[1]);
    enter.append("div").attr("class", "stat-value");
    enter.merge(sel).select(".stat-value").text((d) => this._fmt(stats[d[0]] ?? 0));
  }

  /**
   * 格式化数值显示。
   * @param {*} v
   * @returns {string|number}
   */
  _fmt(v) {
    if (typeof v !== "number") return v;
    return Number.isInteger(v) ? v : v.toFixed(4);
  }

  /**
   * 更新仿真状态指示器。
   * @param {string} state — running / paused / finished
   */
  _setStatus(state) {
    this.simState = state;
    const badge = document.getElementById("status-badge");
    const map = { running: "运行中", paused: "已暂停", finished: "已结束", idle: "空闲" };
    if (badge) {
      badge.textContent = map[state] || state;
      badge.className = "status-badge " + state;
    }

    const btnPause = document.getElementById("btn-pause");
    const btnResume = document.getElementById("btn-resume");
    const btnStop = document.getElementById("btn-stop");
    const finished = state === "finished";

    if (btnPause) {
      btnPause.classList.toggle("active", state === "paused");
      btnPause.disabled = finished;
    }
    if (btnResume) {
      btnResume.classList.toggle("active", state === "running");
      btnResume.disabled = finished;
    }
    if (btnStop) btnStop.disabled = finished;
  }

  /**
   * 通过 WebSocket 发送控制命令。
   * @param {string} type — 命令类型
   * @param {object} payload — 附加参数
   */
  _sendCommand(type, payload = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...payload }));
    }
  }
}

// ── 页面加载时初始化 ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const monitor = new NetworkMonitor();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  monitor.connect(`${proto}://${location.host}/ws`);
});

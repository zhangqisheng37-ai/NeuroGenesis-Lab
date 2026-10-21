/**
 * NeuroGenesis Lab — 实验运行工作台前端逻辑
 *
 * 覆盖完整闭环：参数配置/预设加载 → 启动仿真（POST /api/run）→
 * WebSocket 实时网络图 + 统计 + 时序曲线 → 结束显示实验详情入口。
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

const SERIES_MAX_POINTS = 200;

/**
 * 实时时序曲线 — D3 折线图，环形缓冲滚动窗口。
 */
class TimeSeriesChart {
  constructor(svgSelector) {
    this.svg = d3.select(svgSelector);
    this.maxPoints = SERIES_MAX_POINTS;
    this.series = { alive: [], energy: [], pe: [] };
    this.metrics = [
      { key: "alive", label: "存活", color: "#14b8a6" },
      { key: "energy", label: "平均能量", color: "#f59e0b" },
      { key: "pe", label: "预测误差", color: "#8b5cf6" },
    ];
    this.width = 0;
    this.height = 170;
    this._measure();
    const parent = this.svg.node()?.parentElement;
    if (parent && typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => this._measure()).observe(parent);
    } else {
      window.addEventListener("resize", () => this._measure());
    }
  }

  _measure() {
    const el = this.svg.node();
    if (!el || !el.parentElement) return;
    const width = Math.max(el.parentElement.clientWidth || 300, 200);
    if (width === this.width) return;
    this.width = width;
    this.height = 170;
    this.svg.attr("viewBox", `0 0 ${this.width} ${this.height}`);
    if (this.series.alive.length) this.render();
  }

  /** 追加一个数据点并裁剪窗口。 */
  push(tick, stats) {
    this.series.alive.push({ x: tick, y: Number(stats.alive_neurons) || 0 });
    this.series.energy.push({ x: tick, y: Number(stats.mean_energy) || 0 });
    this.series.pe.push({ x: tick, y: Number(stats.prediction_error) || 0 });
    for (const key of ["alive", "energy", "pe"]) {
      this.series[key] = this.series[key].slice(-this.maxPoints);
    }
    this.render();
  }

  /** 重新渲染全部折线。 */
  render() {
    const margin = { top: 10, right: 10, bottom: 20, left: 34 };
    const w = this.width - margin.left - margin.right;
    const h = this.height - margin.top - margin.bottom;
    if (w <= 0 || h <= 0) return;

    this.svg.selectAll("*").remove();
    const g = this.svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const xDomain = d3.extent(this.series.alive, (d) => d.x);
    const xScale = d3
      .scaleLinear()
      .domain(xDomain[0] === undefined ? [0, 1] : [xDomain[0], Math.max(xDomain[0] + 1, xDomain[1])])
      .range([0, w]);
    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([h, 0]);

    // 三个指标各自归一化（独立 y 轴便于观察趋势）
    for (const m of this.metrics) {
      const pts = this.series[m.key];
      if (pts.length < 2) continue;
      const max = d3.max(pts, (d) => d.y) || 1;
      const domain = [0, max === 0 ? 1 : max * 1.1];
      const localY = d3.scaleLinear().domain(domain).range([h, 0]);
      const line = d3
        .line()
        .x((d) => xScale(d.x))
        .y((d) => localY(d.y));
      g.append("path")
        .datum(pts)
        .attr("fill", "none")
        .attr("stroke", m.color)
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.9)
        .attr("d", line);
    }

    // 坐标轴
    const axisX = d3.axisBottom(xScale).ticks(4);
    const axisY = d3.axisLeft(yScale).ticks(3);
    g.append("g")
      .attr("transform", `translate(0,${h})`)
      .call(axisX)
      .selectAll("text")
      .style("fill", "#8b93a7")
      .style("font-size", "10px");
    g.append("g")
      .call(axisY)
      .selectAll("text")
      .style("fill", "#8b93a7")
      .style("font-size", "10px");
    g.selectAll(".domain, .tick line").style("stroke", "#262b38");
  }

  /** 清空所有数据。 */
  reset() {
    this.series = { alive: [], energy: [], pe: [] };
    this.render();
  }
}

/**
 * 实验工作台主类 — 管理启动流程、WebSocket、D3 渲染与控制交互。
 */
class NetworkMonitor {
  constructor() {
    this.ws = null;
    this.stats = {};
    this.simState = "idle";
    this.experimentId = null;
    this.presetConfig = null;
    this.chart = new TimeSeriesChart("#series-chart");

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
    this._initLaunchPanel();
    this._loadPresets();
    this._initSpeedInput();
    this._applyStatus("idle");

    // 页面加载时检测是否有正在运行的仿真
    this._checkRunning();
  }

  /** 初始化 SVG 画布尺寸，并跟随容器尺寸变化（不依赖 window resize）。 */
  _initSVG() {
    const wrap = document.querySelector(".monitor-canvas-wrap");
    if (!wrap) return;
    const sync = () => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      if (!w || !h || (w === this.width && h === this.height)) return;
      this.width = w;
      this.height = h;
      this.svg.attr("viewBox", `0 0 ${w} ${h}`);
    };
    sync();
    // 启动/暂停会隐藏启动面板改变列宽，需监听容器自身尺寸
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(sync).observe(wrap);
    } else {
      window.addEventListener("resize", sync);
    }
  }

  /** 绑定控制面板按钮事件。 */
  _initControls() {
    const btnPause = document.getElementById("btn-pause");
    const btnResume = document.getElementById("btn-resume");
    const btnStop = document.getElementById("btn-stop");

    if (btnPause) btnPause.onclick = () => { this._sendCommand("pause"); this._applyStatus("paused"); };
    if (btnResume) btnResume.onclick = () => { this._sendCommand("resume"); this._applyStatus("running"); };
    if (btnStop) btnStop.onclick = () => this._sendCommand("stop");
  }

  /** 绑定启动面板：预设加载与启动按钮。 */
  _initLaunchPanel() {
    const btnLaunch = document.getElementById("btn-launch");
    if (btnLaunch) {
      btnLaunch.onclick = () => this._startSimulation();
    }
    const preset = document.getElementById("preset-select");
    if (preset) {
      preset.onchange = () => this._loadPresetConfig(preset.value);
    }
  }

  /** 速度输入：运行时实时调速，空闲时作为启动参数。 */
  _initSpeedInput() {
    const speed = document.getElementById("f-tick-ms");
    if (!speed) return;
    speed.oninput = () => {
      const ms = parseInt(speed.value, 10) || 0;
      const valEl = document.getElementById("speed-val");
      if (valEl) valEl.textContent = ms;
      if (this.simState === "running" || this.simState === "paused") {
        this._sendCommand("speed_change", { tick_ms: ms });
      }
    };
  }

  /** 加载预设列表到下拉框。 */
  async _loadPresets() {
    const select = document.getElementById("preset-select");
    if (!select) return;
    try {
      const data = await apiGet("/api/config/presets", { silent: true });
      for (const p of data || []) {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.name;
        select.appendChild(opt);
      }
    } catch (e) {
      console.warn("加载预设失败:", e);
    }
  }

  /** 加载指定预设内容并回填表单。 */
  async _loadPresetConfig(name) {
    if (!name) {
      this.presetConfig = null;
      return;
    }
    try {
      const cfg = await apiGet(`/api/config/presets/${name}`, { silent: true });
      if (cfg.error) {
        showToast(cfg.error, "error");
        return;
      }
      this.presetConfig = cfg;
      const sim = cfg.simulation || {};
      this._setValue("f-ticks", sim.ticks);
      this._setValue("f-seed", sim.seed);
      this._setValue("f-expr", sim.expr);
      this._setValue("f-input-weight", sim.input_weight);
      this._setValue("f-stim-interval", sim.stim_interval);
      showToast(`已加载预设「${name}」`, "success");
    } catch (e) {
      showToast("加载预设失败: " + e.message, "error");
    }
  }

  /** 安全设置表单值（跳过空值）。 */
  _setValue(id, value) {
    if (value === undefined || value === null) return;
    const el = document.getElementById(id);
    if (el) el.value = value;
  }

  /** 更新顶部实验 ID 显示。 */
  _updateExperimentId() {
    const expEl = document.getElementById("exp-id");
    if (expEl) expEl.textContent = this.experimentId || "—";
  }

  /** 检查是否有正在运行的仿真（页面刷新后恢复）。 */
  async _checkRunning() {
    try {
      const status = await apiGet("/api/run/status", { silent: true });
      if (status.running) {
        this.experimentId = status.experiment_id || null;
        this._updateExperimentId();
        this._applyStatus(status.state || "running");
        this._connect();
      } else if (status.state === "finished" && status.experiment_id) {
        // 仿真在页面加载前已结束：恢复结果入口与启动面板
        this.experimentId = status.experiment_id;
        this._updateExperimentId();
        this._applyStatus("finished");
      }
    } catch (e) {
      console.warn("查询运行状态失败:", e);
    }
  }

  /** 收集启动参数并调用 POST /api/run。 */
  async _startSimulation() {
    const btnLaunch = document.getElementById("btn-launch");
    if (this.simState === "running" || this.simState === "paused") {
      showToast("仿真已在运行中", "warning");
      return;
    }
    const tickMs = Math.max(0, parseInt(document.getElementById("f-tick-ms").value, 10) || 50);

    let body;
    if (this.presetConfig) {
      // 预设模式：完整配置 + 表单覆盖的公共参数
      body = {
        config: this.presetConfig,
        ticks: parseInt(document.getElementById("f-ticks").value, 10) || 100,
        seed: parseInt(document.getElementById("f-seed").value, 10) || 42,
        neurons: Math.max(2, parseInt(document.getElementById("f-neurons").value, 10) || 8),
        expr: document.getElementById("f-expr").value || "1+2=3",
        input_weight: parseFloat(document.getElementById("f-input-weight").value) || 40,
        stim_interval: parseInt(document.getElementById("f-stim-interval").value, 10) || 5,
        tick_ms: tickMs,
        experiment_name: this.presetConfig.simulation?.name || null,
      };
    } else {
      body = {
        ticks: parseInt(document.getElementById("f-ticks").value, 10) || 100,
        seed: parseInt(document.getElementById("f-seed").value, 10) || 42,
        neurons: Math.max(2, parseInt(document.getElementById("f-neurons").value, 10) || 8),
        expr: document.getElementById("f-expr").value || "1+2=3",
        input_weight: parseFloat(document.getElementById("f-input-weight").value) || 40,
        stim_interval: parseInt(document.getElementById("f-stim-interval").value, 10) || 5,
        tick_ms: tickMs,
      };
    }

    try {
      const result = await withLoading(btnLaunch, async () => {
        return await apiPost("/api/run", body);
      });
      if (result.error) {
        showToast(result.error, "error");
        return;
      }
      this.experimentId = result.experiment_id || null;
      this._updateExperimentId();
      this.chart.reset();
      this._applyStatus("running");
      this._connect();
      showToast(`仿真已启动（实验 ${this.experimentId || "—"}）`, "success");
    } catch (e) {
      showToast("启动失败: " + e.message, "error");
    }
  }

  /**
   * 建立 WebSocket 连接。
   * @param {string} [url] — WebSocket 地址，默认当前 host 的 /ws
   */
  connect(url) {
    if (this.ws) {
      this.ws.close();
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(url || `${proto}://${location.host}/ws`);
    const connEl = document.getElementById("conn-state");

    this.ws.onopen = () => {
      if (connEl) { connEl.textContent = "已连接"; connEl.className = "conn-indicator online"; }
    };

    this.ws.onclose = () => {
      if (connEl) {
        if (this.simState === "finished") {
          // 仿真正常结束后的连接关闭不是错误
          connEl.textContent = "已结束";
          connEl.className = "conn-indicator finished";
        } else {
          connEl.textContent = "连接断开";
          connEl.className = "conn-indicator offline";
        }
      }
      // 自动重连（已结束时不再重连）
      if (this.simState !== "finished") {
        setTimeout(() => this.connect(), 1500);
      }
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

  /** 连接别名 — 页面恢复运行状态时使用。 */
  _connect() {
    this.connect();
  }

  /**
   * 根据消息类型分发处理。
   * @param {object} msg
   */
  _handleMessage(msg) {
    if (msg.type === "tick_update") {
      this._updateNetwork(msg.neurons || [], msg.connections || []);
      this._updateStats(msg.stats || {});
      this._updateChart(msg.stats || {});
      const tickEl = document.getElementById("tick-val");
      if (tickEl) tickEl.textContent = msg.tick;
    } else if (msg.type === "status") {
      this._applyStatus(msg.state);
      if (typeof msg.tick_ms === "number") {
        const speedEl = document.getElementById("f-tick-ms");
        const speedValEl = document.getElementById("speed-val");
        if (speedEl) speedEl.value = msg.tick_ms;
        if (speedValEl) speedValEl.textContent = msg.tick_ms;
      }
    } else if (msg.type === "init") {
      console.log("仿真初始化:", msg.total_neurons, "个神经元");
    }
  }

  /** 更新时序曲线。 */
  _updateChart(stats) {
    this.chart.push(Number(stats.tick) || 0, stats);
  }

  /**
   * 更新 D3 网络可视化。
   * @param {Array} neurons — 神经元列表
   * @param {Array} connections — 连接列表
   */
  _updateNetwork(neurons, connections) {
    if (!neurons.length) return;
    const placeholder = document.getElementById("canvas-placeholder");
    if (placeholder) placeholder.style.display = "none";

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
   * 应用仿真状态：徽标、控制按钮、启动面板与结果链接。
   * @param {string} state — idle / running / paused / finished
   */
  _applyStatus(state) {
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
    const launchPanel = document.getElementById("launch-panel");
    const resultLink = document.getElementById("result-link");
    const finished = state === "finished";
    const running = state === "running" || state === "paused";

    if (btnPause) {
      btnPause.classList.toggle("active", state === "paused");
      btnPause.disabled = !running || state === "paused";
    }
    if (btnResume) {
      btnResume.classList.toggle("active", state === "running");
      btnResume.disabled = !running || state === "running";
    }
    if (btnStop) btnStop.disabled = !running;

    // 启动面板在空闲与结束时可见（结束后可立即配置再次运行）；结果入口在结束时显示
    const showLaunch = state === "idle" || state === "finished";
    if (launchPanel) launchPanel.style.display = showLaunch ? "" : "none";
    // 首列随之收缩，避免画布被自动放置到窄列
    const layout = document.querySelector(".workbench-layout");
    if (layout) layout.classList.toggle("is-running", !showLaunch);
    if (resultLink) resultLink.style.display = finished ? "" : "none";
    if (finished && this.experimentId) {
      const link = document.getElementById("result-link-a");
      if (link) link.href = `/experiments/${this.experimentId}`;
    }
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
  new NetworkMonitor();
});

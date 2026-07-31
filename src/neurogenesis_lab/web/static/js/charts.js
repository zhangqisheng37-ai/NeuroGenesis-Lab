/**
 * NeuroGenesis Lab — D3 图表组件
 *
 * 提供分组柱状图与折线图骨架，用于实验指标对比可视化。
 */

/**
 * 渲染分组柱状图 — 每个实验一组，每组内按 metric 分柱。
 *
 * @param {string} containerId - 目标容器 DOM id。
 * @param {Array<{label: string, values: Object<string, number>}>} data - 实验数据。
 * @param {Object} [options] - 可选配置。
 * @param {string} [options.title] - 图表标题。
 */
function renderBarChart(containerId, data, options) {
  const container = document.getElementById(containerId);
  if (!container || !data.length) return;

  options = options || {};

  // 收集所有指标名
  const metricSet = new Set();
  data.forEach(d => Object.keys(d.values).forEach(k => metricSet.add(k)));
  const metrics = Array.from(metricSet);
  if (!metrics.length) {
    container.textContent = "无指标数据";
    return;
  }

  // 尺寸计算
  const rect = container.getBoundingClientRect();
  const margin = { top: 20, right: 20, bottom: 60, left: 50 };
  const width = Math.max(rect.width || 600, 300) - margin.left - margin.right;
  const height = 280 - margin.top - margin.bottom;

  // 清空并创建 SVG
  d3.select("#" + containerId).selectAll("*").remove();
  const svg = d3.select("#" + containerId)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // 比例尺
  const x0 = d3.scaleBand().domain(data.map(d => d.label)).range([0, width]).padding(0.2);
  const x1 = d3.scaleBand().domain(metrics).range([0, x0.bandwidth()]).padding(0.05);

  const maxVal = d3.max(data, d => d3.max(metrics, k => Number(d.values[k]) || 0)) || 1;
  const y = d3.scaleLinear().domain([0, maxVal * 1.1]).nice().range([height, 0]);

  // 颜色 — 使用 CSS 变量派生色板
  const colors = ["#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#3b82f6", "#ec4899"];
  const color = d3.scaleOrdinal().domain(metrics).range(colors);

  // 绘制柱形
  const groups = svg.selectAll(".group")
    .data(data)
    .join("g")
    .attr("transform", d => `translate(${x0(d.label)},0)`);

  groups.selectAll("rect")
    .data(d => metrics.map(k => ({ key: k, val: Number(d.values[k]) || 0 })))
    .join("rect")
    .attr("x", d => x1(d.key))
    .attr("y", d => y(d.val))
    .attr("width", x1.bandwidth())
    .attr("height", d => height - y(d.val))
    .attr("fill", d => color(d.key))
    .attr("rx", 2);

  // X 轴
  svg.append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x0))
    .selectAll("text")
    .attr("transform", "rotate(-20)")
    .style("text-anchor", "end")
    .style("fill", "#8b93a7")
    .style("font-size", "11px");

  svg.selectAll(".domain, .tick line").style("stroke", "#262b38");

  // Y 轴
  svg.append("g")
    .call(d3.axisLeft(y).ticks(5))
    .selectAll("text")
    .style("fill", "#8b93a7")
    .style("font-size", "11px");

  // 图例
  const legend = svg.append("g")
    .attr("transform", `translate(${width - metrics.length * 80}, -10)`);

  metrics.forEach((m, i) => {
    const g = legend.append("g").attr("transform", `translate(${i * 80}, 0)`);
    g.append("rect").attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", color(m));
    g.append("text").attr("x", 14).attr("y", 9).text(m)
      .style("fill", "#e6e8ee").style("font-size", "11px");
  });
}

/**
 * 渲染折线图（骨架 — Phase 4 完善）。
 *
 * @param {string} containerId - 目标容器 DOM id。
 * @param {Array<{label: string, points: Array<{x: number, y: number}>}>} series - 序列数据。
 * @param {Object} [options] - 可选配置。
 */
function renderLineChart(containerId, series, options) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.textContent = "折线图组件将在 Phase 4 实现";
}

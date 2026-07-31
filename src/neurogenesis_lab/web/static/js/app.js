/**
 * NeuroGenesis Lab — P5 Web 管理界面共享工具函数
 */

/**
 * 发起 GET 请求并解析 JSON 响应。
 * @param {string} url
 * @returns {Promise<any>}
 */
async function apiGet(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`GET ${url} 失败: ${res.status}`);
  return res.json();
}

/**
 * 发起 DELETE 请求。
 * @param {string} url
 * @returns {Promise<any>}
 */
async function apiDelete(url) {
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE ${url} 失败: ${res.status}`);
  return res.json();
}

/**
 * 发起 POST 请求（JSON body）。
 * @param {string} url
 * @param {object} data
 * @returns {Promise<any>}
 */
async function apiPost(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`POST ${url} 失败: ${res.status}`);
  return res.json();
}

/** 弹出原生确认对话框。 */
function confirmAction(message) {
  return confirm(message);
}

/**
 * 显示 Toast 通知。
 * @param {string} message
 * @param {"info"|"success"|"error"} type
 */
function showToast(message, type = "info") {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  const el = document.createElement("div");
  el.className = `toast toast-${type === "info" ? "" : type}`.trim();
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => { el.remove(); }, 3500);
}

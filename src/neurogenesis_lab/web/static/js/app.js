/**
 * NeuroGenesis Lab — P5 Web 管理界面共享工具函数
 */

// ── Toast 图标映射 ──────────────────────────────────────────
const TOAST_ICONS = { info: "ℹ️", success: "✅", error: "❌", warning: "⚠️" };

/**
 * 显示 Toast 通知（支持堆叠、关闭按钮、滑出动画）。
 * @param {string} message - 通知消息
 * @param {"info"|"success"|"error"|"warning"} type - 通知类型
 * @param {number} [duration=3500] - 自动关闭时间(ms)，0 表示不自动关闭
 */
function showToast(message, type = "info", duration = 3500) {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const icon = TOAST_ICONS[type] || TOAST_ICONS.info;
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;

  const iconSpan = document.createElement("span");
  iconSpan.className = "toast-icon";
  iconSpan.textContent = icon;
  el.appendChild(iconSpan);

  const msgSpan = document.createElement("span");
  msgSpan.className = "toast-message";
  msgSpan.textContent = message;
  el.appendChild(msgSpan);

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast-close";
  closeBtn.setAttribute("aria-label", "关闭");
  closeBtn.textContent = "\u00d7";
  el.appendChild(closeBtn);

  // 关闭按钮
  closeBtn.addEventListener("click", () => dismissToast(el));

  container.appendChild(el);

  if (duration > 0) {
    setTimeout(() => dismissToast(el), duration);
  }
}

/** 带滑出动画地移除 toast 元素 */
function dismissToast(el) {
  if (el._dismissed) return;
  el._dismissed = true;
  el.style.animation = "toast-out 0.25s forwards";
  el.addEventListener("animationend", () => el.remove(), { once: true });
}

// ── API 请求封装（自动 loading + 错误 toast）─────────────────

let _apiLoadingCount = 0;

function _showGlobalLoading() {
  _apiLoadingCount++;
  let overlay = document.querySelector(".loading-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.innerHTML = '<div class="spinner"></div>';
    document.body.appendChild(overlay);
  }
  overlay.style.display = "flex";
}

function _hideGlobalLoading() {
  _apiLoadingCount = Math.max(0, _apiLoadingCount - 1);
  if (_apiLoadingCount === 0) {
    const overlay = document.querySelector(".loading-overlay");
    if (overlay) overlay.style.display = "none";
  }
}

/**
 * 发起 GET 请求并解析 JSON 响应。
 * 自动显示全局 loading，失败时自动弹 error toast。
 * @param {string} url
 * @param {object} [opts] - 可选配置
 * @param {boolean} [opts.silent=false] - 静默模式（不显示 loading / 不弹错误 toast）
 * @returns {Promise<any>}
 */
async function apiGet(url, opts = {}) {
  if (!opts.silent) _showGlobalLoading();
  try {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`GET ${url} 失败: ${res.status}`);
    return await res.json();
  } catch (err) {
    if (!opts.silent) showToast(err.message, "error");
    throw err;
  } finally {
    if (!opts.silent) _hideGlobalLoading();
  }
}

/**
 * 发起 DELETE 请求。
 * @param {string} url
 * @param {object} [opts]
 * @returns {Promise<any>}
 */
async function apiDelete(url, opts = {}) {
  if (!opts.silent) _showGlobalLoading();
  try {
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) throw new Error(`DELETE ${url} 失败: ${res.status}`);
    return await res.json();
  } catch (err) {
    if (!opts.silent) showToast(err.message, "error");
    throw err;
  } finally {
    if (!opts.silent) _hideGlobalLoading();
  }
}

/**
 * 发起 POST 请求（JSON body）。
 * @param {string} url
 * @param {object} data
 * @param {object} [opts]
 * @returns {Promise<any>}
 */
async function apiPost(url, data, opts = {}) {
  if (!opts.silent) _showGlobalLoading();
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`POST ${url} 失败: ${res.status}`);
    return await res.json();
  } catch (err) {
    if (!opts.silent) showToast(err.message, "error");
    throw err;
  } finally {
    if (!opts.silent) _hideGlobalLoading();
  }
}

// ── 确认对话框 ──────────────────────────────────────────────

/**
 * 显示自定义确认对话框（替代浏览器原生 confirm）。
 * @param {string} title - 对话框标题
 * @param {string} message - 对话框消息
 * @param {object} [options] - 可选配置
 * @param {string} [options.confirmText="确认"] - 确认按钮文本
 * @param {string} [options.cancelText="取消"] - 取消按钮文本
 * @param {"primary"|"danger"} [options.confirmStyle="primary"] - 确认按钮样式
 * @returns {Promise<boolean>}
 */
function showConfirmDialog(title, message, options = {}) {
  const {
    confirmText = "确认",
    cancelText = "取消",
    confirmStyle = "primary",
  } = options;

  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const btnClass = confirmStyle === "danger" ? "btn btn-danger" : "btn btn-primary";

    const dialog = document.createElement("div");
    dialog.className = "modal-dialog";

    const titleEl = document.createElement("div");
    titleEl.className = "modal-title";
    titleEl.textContent = title;
    dialog.appendChild(titleEl);

    const msgEl = document.createElement("div");
    msgEl.className = "modal-message";
    msgEl.textContent = message;
    dialog.appendChild(msgEl);

    const actions = document.createElement("div");
    actions.className = "modal-actions";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn modal-cancel";
    cancelBtn.textContent = cancelText;
    actions.appendChild(cancelBtn);

    const confirmBtn = document.createElement("button");
    confirmBtn.className = `${btnClass} modal-confirm`;
    confirmBtn.textContent = confirmText;
    actions.appendChild(confirmBtn);

    dialog.appendChild(actions);
    overlay.appendChild(dialog);

    function close(result) {
      overlay.style.animation = "fade-in 0.15s reverse forwards";
      const dialog = overlay.querySelector(".modal-dialog");
      if (dialog) dialog.style.animation = "modal-in 0.15s reverse forwards";
      setTimeout(() => {
        overlay.remove();
        document.removeEventListener("keydown", onKey);
        resolve(result);
      }, 140);
    }

    cancelBtn.addEventListener("click", () => close(false));
    confirmBtn.addEventListener("click", () => close(true));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });

    function onKey(e) { if (e.key === "Escape") close(false); }
    document.addEventListener("keydown", onKey);

    document.body.appendChild(overlay);
    // 聚焦取消按钮（安全默认）
    cancelBtn.focus();
  });
}

/**
 * 弹出确认对话框（向后兼容，使用自定义对话框替代原生 confirm）。
 * @param {string} message
 * @returns {Promise<boolean>}
 */
function confirmAction(message) {
  return showConfirmDialog("确认操作", message, { confirmStyle: "danger" });
}

// ── withLoading ─────────────────────────────────────────────

/**
 * 在异步操作期间显示 loading 状态。
 * @param {HTMLElement} [button] - 可选的按钮元素（禁用 + spinner）
 * @param {Function} asyncFn - 异步操作函数
 * @returns {Promise}
 */
async function withLoading(button, asyncFn) {
  if (button) {
    button.disabled = true;
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner"></span> 处理中...';
    try {
      return await asyncFn();
    } finally {
      button.disabled = false;
      button.innerHTML = originalText;
    }
  }
  return await asyncFn();
}

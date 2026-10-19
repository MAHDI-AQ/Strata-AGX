"use strict";

function escapeHtml(str) {
  if (str == null) return "";
  return String(str).replace(/[&<>"']/g, function(m) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"}[m];
  });
}

const $ = id => document.getElementById(id);
let selected = null, detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const active = r => r && !["completed", "error", "disconnected"].includes(r.state);
function text(id, value) { const el = $(id); if (el && el.textContent !== String(value)) el.textContent = value; }

async function api(path, body) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
      ...(apiKey ? {Authorization: `Bearer ${apiKey}`} : {}),
      ...(body !== undefined ? {"Content-Type": "application/json"} : {})
    },
    ...(body !== undefined ? {method: "POST", body: JSON.stringify(body)} : {})
  });
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}

function notice(message) { $("error").hidden = !message; text("error", message || ""); }

function list() {
  const searchEl = $("search"), filterEl = $("filter");
  const query = (searchEl ? searchEl.value : "").toLowerCase();
  const filter = filterEl ? filterEl.value : "all";
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(query) && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
  const container = $("requests");
  if (!container) return;

  if (!visible.length) {
    let empty = container.querySelector(".empty");
    if (!empty) {
      container.replaceChildren();
      empty = document.createElement("div");
      empty.className = "empty";
      container.append(empty);
    }
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    return;
  } else {
    const empty = container.querySelector(".empty");
    if (empty) empty.remove();
  }

  // Keyed in-place update by request ID with fingerprint check
  const existingMap = new Map();
  for (const child of Array.from(container.children)) {
    if (child.dataset?.request) existingMap.set(child.dataset.request, child);
  }

  const newIds = new Set(visible.map(r => r.id));
  // Remove items not in visible
  for (const [id, el] of existingMap.entries()) {
    if (!newIds.has(id)) el.remove();
  }

  // Insert or update in order
  let prevEl = null;
  for (const r of visible) {
    let btn = existingMap.get(r.id);
    const isSelected = selected === r.id;
    const timeStr = new Date(r.started_at * 1000).toLocaleTimeString();
    const metaStr = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    const fingerprint = `${r.state}|${timeStr}|${metaStr}|${r.path}|${isSelected}`;

    if (!btn) {
      btn = document.createElement("button");
      btn.dataset.request = r.id;
      btn.dataset.fingerprint = fingerprint;
      btn.setAttribute("aria-label", `Request ${r.id}`);
      btn.className = `request${isSelected ? " selected" : ""}`;
      btn.setAttribute("aria-pressed", String(isSelected));
      btn.innerHTML = `<div class="row"><span>${escapeHtml(timeStr)}</span><span class="tag ${escapeHtml(r.state)}">${escapeHtml(r.state)}</span></div>` +
                      `<div class="endpoint">POST ${escapeHtml(r.path)}</div>` +
                      `<div class="small">${escapeHtml(metaStr)}</div>`;
      btn.onclick = async () => {
        if (selected !== r.id) {
          selected = r.id;
          detail = null;
          list();
          await showDetail();
        }
      };
      if (prevEl) prevEl.after(btn); else container.prepend(btn);
    } else {
      if (btn.dataset.fingerprint !== fingerprint) {
        btn.dataset.fingerprint = fingerprint;
        const targetClass = `request${isSelected ? " selected" : ""}`;
        if (btn.className !== targetClass) btn.className = targetClass;
        btn.setAttribute("aria-pressed", String(isSelected));

        const rowSpan = btn.querySelector(".row > span:first-child");
        if (rowSpan && rowSpan.textContent !== timeStr) rowSpan.textContent = timeStr;

        const tagSpan = btn.querySelector(".row > span.tag");
        if (tagSpan) {
          const targetTagClass = `tag ${r.state}`;
          if (tagSpan.className !== targetTagClass) tagSpan.className = targetTagClass;
          if (tagSpan.textContent !== r.state) tagSpan.textContent = r.state;
        }

        const epDiv = btn.querySelector(".endpoint");
        const epText = `POST ${r.path}`;
        if (epDiv && epDiv.textContent !== epText) epDiv.textContent = epText;

        const metaDiv = btn.querySelector(".small");
        if (metaDiv && metaDiv.textContent !== metaStr) metaDiv.textContent = metaStr;
      }

      if (prevEl && btn.previousElementSibling !== prevEl) {
        prevEl.after(btn);
      }
    }
    prevEl = btn;
  }
}

function content() {
  if (!detail) return;
  let value = detail[pane] || (pane === "response" && detail.stream ? "Streaming response: see Output and the request error, if any." : "No content.");
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? "Raw model output retained for diagnosis. The API returned an error." : pane === "input" ? "Original request body." : pane === "output" ? "Model answer. JSON is formatted here for readability." : pane === "reasoning" ? "Separate reasoning content, when enabled." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated (preview); the API response was not shortened." : ""));
}

async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s)); text("detail-queue", seconds(value.queue_s)); text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || "text");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}


const SLOT_COLORS = ["var(--st-accent)", "#3b82f6", "#a855f7", "#f59e0b", "#ec4899", "#06b6d4", "#14b8a6", "#8b5cf6"];

function syncMonitorSlots(numSlots) {
  const container = $("slot-cards") || document.querySelector("#slot-cards");
  if (!container) return;
  container.style.gridTemplateColumns = numSlots === 1 ? "1fr" : (numSlots <= 4 ? `repeat(${numSlots}, 1fr)` : "repeat(auto-fit, minmax(280px, 1fr))");
  const existing = container.querySelectorAll(".card[id^='slot-card-']");
  if (existing.length !== numSlots) {
    let html = "";
    for (let sId = 0; sId < numSlots; sId++) {
      const col = SLOT_COLORS[sId % SLOT_COLORS.length];
      html += `
<div class="card" id="slot-card-${sId}" style="border-left: 4px solid ${col};">
  <div class="row" style="align-items: baseline;">
    <div class="label" style="font-weight: 700; color: ${col};">Slot ${sId} · Concurrency Engine</div>
    <span class="tag" id="slot${sId}-tag" style="margin-left:auto;">Idle</span>
  </div>
  <div class="value" style="font-size: 20px; margin: 4px 0;"><span id="slot${sId}-tokens">0 / 0 tokens</span></div>
  <div class="small" id="slot${sId}-speed" style="color: var(--st-ink-muted);">Waiting for request</div>
  <div style="background: var(--st-surface-2); height: 6px; border-radius: 3px; margin-top: 8px; overflow: hidden;">
    <div id="slot${sId}-bar" style="background: ${col}; height: 100%; width: 0%;"></div>
  </div>
</div>`;
    }
    container.innerHTML = html;
  }
}

async function refresh() {
  try {
    const [status, history, metrics] = await Promise.all([
      api("v1/status"),
      api("api/requests"),
      api("metrics").catch(() => null)
    ]);
    state = status;
    records = Array.isArray(history?.requests) ? history.requests : [];
    status.loaded = history?.loaded;
    status.auto_load = history?.auto_load;

    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null;
      $("selection").hidden = true;
      $("no-selection").hidden = false;
    }

    const inflight = Math.max(status.activity?.in_flight || 0, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    const conc = metrics?.engine?.concurrency || status?.concurrency?.serving || 3;
    text("model", `${status.model || "127.0.0.1:8096"} (C=${conc} Slots)`);
    text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on the next request" : "No active requests");

    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—");
    text("speed", speed(last?.timings?.predicted_per_second));

    if (metrics && typeof metrics === "object") {
      // Dynamic Slot Rendering (supports 1, 2, 3... N slots)
      const slots = Array.isArray(metrics.slots)
        ? metrics.slots
        : (metrics.slots && typeof metrics.slots === "object" ? Object.values(metrics.slots) : []);
      const eng = (metrics.engine && typeof metrics.engine === "object") ? metrics.engine : {};
      const numSlots = slots.length || (eng.concurrency || (status?.concurrency?.serving || 2));
      syncMonitorSlots(numSlots);

      for (let sId = 0; sId < numSlots; sId++) {
        const slot = slots.find(s => s && s.id === sId) || { id: sId, status: "idle", prompt_tokens: 0, generated: 0 };
        const tag = $(`slot${sId}-tag`);
        const tokEl = $(`slot${sId}-tokens`);
        const spEl = $(`slot${sId}-speed`);
        const bar = $(`slot${sId}-bar`);
        if (tag) {
          tag.className = `tag ${slot.status || "idle"}`;
          tag.textContent = slot.status === "reading" ? `Reading [${slot.cache_tier || "Fresh"}]` : (slot.status === "generating" ? (slot.phase || "Generating") : "Idle");
        }
        if (slot.status === "reading") {
          if (tokEl) tokEl.textContent = `${slot.prompt_read || slot.prompt_tokens || 0} / ${slot.prompt_tokens || 0} tokens`;
          if (spEl) spEl.textContent = slot.prefill_tok_s ? `Prefill: ${slot.prefill_tok_s} tok/s` : "Prefill in progress";
          if (bar) bar.style.width = slot.prompt_tokens ? `${Math.min(100, (100 * (slot.prompt_read || 0)) / slot.prompt_tokens)}%` : "40%";
        } else if (slot.status === "generating") {
          if (tokEl) tokEl.textContent = `${slot.generated || 0} of ${slot.max_tokens || "—"} tokens`;
          if (spEl) spEl.textContent = slot.tok_s ? `Decode: ${slot.tok_s} tok/s` : "Generating";
          if (bar) bar.style.width = slot.max_tokens ? `${Math.min(100, (100 * (slot.generated || 0)) / slot.max_tokens)}%` : "50%";
        } else {
          const lReq = slot.last_request;
          if (lReq) {
            if (tokEl) tokEl.textContent = `Last: ${lReq.output_tokens || 0} tokens (${lReq.duration_s || 0}s)`;
            const pStr = lReq.prefill_tok_s ? ` · prefill ${lReq.prefill_tok_s} t/s` : "";
            const dStr = lReq.decode_tok_s ? `${lReq.decode_tok_s} t/s` : "done";
            if (spEl) spEl.textContent = `${dStr}${pStr} · [${lReq.cache_tier || "Fresh"}]`;
          } else {
            if (tokEl) tokEl.textContent = "0 / 0 tokens";
            if (spEl) spEl.textContent = "Standing by";
          }
          if (bar) bar.style.width = "0%";
        }
      }

      // Radix 3-Tier Stats Rendering (handles object, string or null/empty)
      if (metrics.radix && typeof metrics.radix === "object") {
        const rx = metrics.radix;
        if ($("radix-mon-tag")) $("radix-mon-tag").textContent = rx.cache_tier || "3-Tier (VRAM Pool → Host RAM → NVMe Disk)";
        if ($("radix-mon-l1")) {
          const eng = (metrics.engine && typeof metrics.engine === "object") ? metrics.engine : {};
          const expStr = eng.expert_slots ? `${eng.expert_slots.toLocaleString()} Exp` : "Resident Exp";
          if (rx.l1_display) {
            $("radix-mon-l1").textContent = rx.l1_display;
          } else if (rx.l1_active_tokens != null && rx.l1_total_tokens != null) {
            $("radix-mon-l1").textContent = `${eng.concurrency || 2} slots · ${rx.l1_active_tokens.toLocaleString()} / ${rx.l1_total_tokens.toLocaleString()} toks`;
          } else {
            const poolDesc = rx.l1_vram_pool || `512k Context Pool (${(eng.kv || "q4_0").toUpperCase()})`;
            const activeTok = rx.l1_active_tokens != null ? `${rx.l1_active_tokens.toLocaleString()} toks` : "0 toks";
            const l1Text = rx.l1_vram_note || `${poolDesc} · ${activeTok} · ${expStr}`;
            $("radix-mon-l1").textContent = l1Text;
          }
        }
        if ($("radix-mon-l2")) {
          const nodeCount = rx.l2_host_nodes != null ? rx.l2_host_nodes : (rx.l2_host_active || 0);
          const hostGib = rx.l2_host_gib != null ? Number(rx.l2_host_gib).toFixed(2) : "0.00";
          $("radix-mon-l2").textContent = rx.l2_display || `${nodeCount} nodes · ${hostGib} GiB`;
        }
        if ($("radix-mon-l3")) {
          $("radix-mon-l3").textContent = rx.l3_display || `${rx.l3_nvme_snapshots || 0} snap · ${rx.l3_nvme_gib || 0} GiB`;
        }
        if ($("radix-mon-hit")) $("radix-mon-hit").textContent = `${((rx.hit_rate || 0)).toFixed(1)}%`;
        if ($("radix-mon-nodes")) $("radix-mon-nodes").textContent = `${rx.total_nodes || 0} nodes`;
      }

      // MTP Speculative Stats Rendering (handles object or null/empty)
      if (metrics.mtp && typeof metrics.mtp === "object") {
        const mt = metrics.mtp;
        if ($("mtp-mon-acc")) $("mtp-mon-acc").textContent = `${((mt.acceptance_rate || 0)).toFixed(1)}%`;
        if ($("mtp-mon-ratio")) $("mtp-mon-ratio").textContent = `${mt.drafts_accepted || 0} / ${mt.drafts_offered || 0}`;
        if ($("mtp-mon-speedup")) $("mtp-mon-speedup").textContent = `${((mt.speculative_speedup || 1.0)).toFixed(2)}x`;
        if ($("mtp-mon-rolling")) $("mtp-mon-rolling").textContent = `${((mt.rolling_acceptance_rate || 0)).toFixed(1)}%`;
      }
    }

    if (metrics && metrics.hardware && Array.isArray(metrics.hardware.gpus)) {
      const g0 = metrics.hardware.gpus[0];
      const g1 = metrics.hardware.gpus[1];
      if (g0) {
        text("gpu0-vram", (g0.mem_used / 1073741824).toFixed(1) + " GB");
        text("gpu0-temp", (g0.temp != null ? g0.temp + " °C" : "—") + (g0.util != null ? " · " + g0.util + "% util" : ""));
        const bar0 = document.getElementById("gpu0-bar");
        if (bar0 && g0.mem_total) bar0.style.width = Math.min(100, (100 * g0.mem_used / g0.mem_total)).toFixed(1) + "%";
      }
      if (g1) {
        text("gpu1-vram", (g1.mem_used / 1073741824).toFixed(1) + " GB");
        text("gpu1-temp", (g1.temp != null ? g1.temp + " °C" : "—") + (g1.util != null ? " · " + g1.util + "% util" : ""));
        const bar1 = document.getElementById("gpu1-bar");
        if (bar1 && g1.mem_total) bar1.style.width = Math.min(100, (100 * g1.mem_used / g1.mem_total)).toFixed(1) + "%";
      }
    }

    $("load").disabled = operating || Boolean(inflight) || status.loaded;
    $("unload").disabled = operating || Boolean(inflight) || !status.loaded;

    list();

    // Re-fetch detail ONLY if selected request is active/in-flight, or if detail hasn't been fetched yet
    if (selected) {
      const selRec = records.find(r => r.id === selected);
      if (!detail || (selRec && active(selRec))) {
        await showDetail();
      }
    }

    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) {
    notice(error.message);
    text("updated", "Disconnected · retrying");
    $("load").disabled = $("unload").disabled = true;
  }
}

async function control(load) {
  actionError = "";
  operating = true;
  $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try {
    await api(load ? "/load" : "/unload", {});
    notice("");
  } catch (error) {
    actionError = error.message;
    notice(actionError);
  } finally {
    operating = false;
    await refresh();
  }
}

$("load").onclick = () => control(true);
$("unload").onclick = () => control(false);
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => {
  event.preventDefault();
  apiKey = $("api-key").value.trim();
  sessionStorage.setItem("strata.monitor.key", apiKey);
  $("auth").hidden = true;
  await refresh();
};
$("theme").onclick = () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("strata.theme", theme);
};
$("search").oninput = list;
$("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => {
  button.onclick = () => {
    pane = button.dataset.pane;
    document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button)));
    content();
  };
});
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("content").textContent);
    text("copy", "Copied");
    setTimeout(() => text("copy", "Copy"), 1500);
  } catch (_) {
    notice("Clipboard unavailable; select the text to copy it.");
  }
};

async function poll() {
  await refresh();
  setTimeout(poll, 2000);
}
poll();

// App shell: hash routing between pages, the assistant sidebar / drawer /
// bottom sheet, the one map stage that moves between pages, the sliding tab
// thumb, and small motion helpers. No framework: a few listeners.

export type View = "today" | "map" | "health" | "simulate";

const VIEWS: View[] = ["today", "map", "health", "simulate"];
const TITLES: Record<View, [string, string]> = {
  today: ["Today", "What to expect on Vembanad and the Kuttanad canals"],
  map: ["Map", "Forecast, hotspots and the last satellite photo"],
  health: ["Lake health", "How the lake has changed, from satellite photos since 2025"],
  simulate: ["Try a what-if", "Change the wind or the barrage and watch the weed move"],
};
/** Pages that show the map stage, and where it goes. */
const STAGE_SLOT: Partial<Record<View, string>> = { today: "today", map: "map", simulate: "simulate" };

const KEY_DOCK = "hyacinth.askDock";
const KEY_WELCOME = "hyacinth.welcomed";

const store = {
  get(k: string): string | null {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string) {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* storage unavailable */
    }
  },
};

const mq = (q: string) => matchMedia(q);
/** Assistant is a docked column (vs. an overlay) at this width. */
const wide = mq("(min-width: 1280px)");
const phone = mq("(max-width: 767px)");
const reduced = mq("(prefers-reduced-motion: reduce)");

let current: View = "today";
const listeners: ((v: View) => void)[] = [];

/** Slide the tab pill under the selected tab. */
export function moveTabThumb(): void {
  const tabs = document.querySelector<HTMLElement>(".tabs");
  const sel = tabs?.querySelector<HTMLElement>('button[aria-selected="true"]');
  if (!tabs || !sel) return;
  tabs.style.setProperty("--x", `${sel.offsetLeft}px`);
  tabs.style.setProperty("--w", `${sel.offsetWidth}px`);
  // keep the selected tab in view when the bar scrolls (phones)
  const l = sel.offsetLeft - tabs.scrollLeft;
  if (l < 0 || l + sel.offsetWidth > tabs.clientWidth) {
    tabs.scrollTo({ left: sel.offsetLeft - 8, behavior: reduced.matches ? "auto" : "smooth" });
  }
}

export function currentView(): View {
  return current;
}

export function onViewChange(fn: (v: View) => void): void {
  listeners.push(fn);
}

function placeStage(v: View) {
  const stage = document.getElementById("stage");
  const name = STAGE_SLOT[v];
  if (!stage || !name) return;
  const slot = document.querySelector<HTMLElement>(`[data-slot="${name}"]`);
  if (!slot || stage.parentElement === slot) return;
  const video = stage.querySelector("video");
  const wasPlaying = video && !video.hidden && !video.paused;
  slot.append(stage); // the canvases keep their GPU contexts when moved
  stage.dataset.slot = name;
  if (wasPlaying) video!.play().catch(() => {});
  requestAnimationFrame(moveTabThumb);
}

/** Switch page. `focus` scrolls to an element inside it (e.g. "debris"). */
export function showView(v: View, focus?: string): void {
  const changed = v !== current;
  const apply = () => {
    current = v;
    document.body.dataset.view = v;
    for (const el of document.querySelectorAll<HTMLElement>(".view")) el.hidden = el.dataset.view !== v;
    for (const a of document.querySelectorAll<HTMLElement>("[data-nav]")) {
      if (a.dataset.nav === v) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
    const [title, sub] = TITLES[v];
    document.getElementById("page-title")!.textContent = title;
    document.getElementById("page-sub")!.textContent = sub;
    document.title = v === "today" ? "Hyacinth · കുളവാഴ" : `${title} · Hyacinth`;
    placeStage(v);
    for (const fn of listeners) fn(v);
  };
  const after = () => {
    const target = focus ? document.getElementById(focus) : null;
    if (target) target.scrollIntoView({ behavior: reduced.matches ? "auto" : "smooth", block: "start" });
    else if (changed) scrollTo({ top: 0 });
  };
  type VT = { updateCallbackDone: Promise<void> };
  const doc = document as Document & { startViewTransition?: (cb: () => void) => VT };
  if (changed && !reduced.matches && doc.startViewTransition) {
    doc.startViewTransition(apply).updateCallbackDone.then(after, after);
  } else {
    apply();
    after();
  }
  if (changed && phone.matches) closeAssistant();
}

/** Make sure the map is on screen (assistant tools, "show on map" buttons). */
export function revealMap(): void {
  if (!STAGE_SLOT[current]) go("map");
  const stage = document.getElementById("stage");
  if (!stage) return;
  if (phone.matches) setSheet("peek");
  requestAnimationFrame(() => {
    const r = stage.getBoundingClientRect();
    if (r.top < 0 || r.bottom > innerHeight) stage.scrollIntoView({ behavior: reduced.matches ? "auto" : "smooth", block: "center" });
  });
}

function go(v: View, focus?: string) {
  const hash = focus ?? v;
  if (location.hash.slice(1) !== hash) history.pushState(null, "", `#${hash}`);
  showView(v, focus);
}

function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (!h) return showView("today");
  if ((VIEWS as string[]).includes(h)) return showView(h as View);
  const el = document.getElementById(h);
  const host = el?.closest<HTMLElement>(".view")?.dataset.view as View | undefined;
  showView(host ?? "today", el ? h : undefined);
}

// ---------------------------------------------------------------- assistant

type Sheet = "closed" | "open" | "peek";

function setSheet(s: Sheet) {
  const ask = document.getElementById("ask")!;
  const scrim = document.getElementById("ask-scrim")!;
  const docked = wide.matches && document.body.dataset.dock !== "off";
  if (docked) {
    ask.dataset.open = "true";
    scrim.hidden = true;
    return;
  }
  ask.dataset.open = s === "closed" ? "false" : s === "peek" ? "peek" : "true";
  scrim.hidden = s !== "open";
  document.body.classList.toggle("sheet-open", s === "open");
}

export function openAssistant(): void {
  if (wide.matches) {
    setDock(true);
  } else setSheet("open");
  const input = document.getElementById("ask-input") as HTMLTextAreaElement | null;
  // phones: don't pop the keyboard over the answer unless asked
  if (input && !phone.matches) setTimeout(() => input.focus({ preventScroll: true }), 250);
}

function closeAssistant() {
  if (wide.matches) return;
  setSheet("closed");
}

function setDock(on: boolean) {
  document.body.dataset.dock = on ? "on" : "off";
  document.getElementById("ask-dock")?.setAttribute("aria-pressed", String(on));
  store.set(KEY_DOCK, on ? "on" : "off");
  setSheet(on ? "open" : "closed");
  // the map stage changes width: keep the tab thumb aligned
  setTimeout(moveTabThumb, 320);
}

function initAssistantChrome() {
  document.body.dataset.dock = store.get(KEY_DOCK) === "off" ? "off" : "on";
  document.getElementById("ask-dock")?.setAttribute("aria-pressed", String(document.body.dataset.dock === "on"));
  setSheet("closed");
  for (const b of document.querySelectorAll<HTMLElement>("[data-open-ask]")) b.addEventListener("click", openAssistant);
  document.getElementById("ask-dock")?.addEventListener("click", () => setDock(document.body.dataset.dock === "off"));
  document.getElementById("ask-close")?.addEventListener("click", () => (wide.matches ? setDock(false) : setSheet("closed")));
  document.getElementById("ask-scrim")?.addEventListener("click", () => setSheet("closed"));
  addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !wide.matches && document.getElementById("ask")?.dataset.open !== "false") setSheet("closed");
  });
  wide.addEventListener("change", () => setSheet(wide.matches ? "open" : "closed"));

  // Preset questions anywhere on the page open the assistant (ask.ts sends them).
  document.addEventListener("click", (e) => {
    if ((e.target as Element).closest?.("[data-ask]")) openAssistant();
  });

  // Bottom sheet handle: tap toggles, drag down closes / drag up opens.
  const handle = document.getElementById("ask-handle");
  const ask = document.getElementById("ask")!;
  if (handle) {
    let y0 = 0;
    let dy = 0;
    let dragging = false;
    handle.addEventListener("pointerdown", (e) => {
      dragging = true;
      y0 = e.clientY;
      dy = 0;
      handle.setPointerCapture(e.pointerId);
      ask.classList.add("dragging");
    });
    handle.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      dy = e.clientY - y0;
      ask.style.setProperty("--drag", `${Math.max(-80, dy)}px`);
    });
    const end = () => {
      if (!dragging) return;
      dragging = false;
      ask.classList.remove("dragging");
      ask.style.removeProperty("--drag");
      const state = ask.dataset.open;
      if (Math.abs(dy) < 6) setSheet(state === "true" ? "peek" : "open");
      else if (dy > 60) setSheet(state === "true" ? "peek" : "closed");
      else if (dy < -40) setSheet("open");
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  }

  // Grow the question box with its text.
  const input = document.getElementById("ask-input") as HTMLTextAreaElement | null;
  if (input) {
    const fit = () => {
      input.style.height = "auto";
      input.style.height = `${Math.min(160, input.scrollHeight)}px`;
    };
    input.addEventListener("input", fit);
    document.getElementById("ask-form")?.addEventListener("submit", () => requestAnimationFrame(fit));
  }
}

// ---------------------------------------------------------------- page

export function initChrome(stage: HTMLElement): void {
  addEventListener("resize", moveTabThumb);
  document.fonts?.ready.then(moveTabThumb).catch(() => {});

  initAssistantChrome();

  // Router: links are plain #hash anchors, so back/forward and sharing work.
  addEventListener("hashchange", route);
  document.addEventListener("click", (e) => {
    const a = (e.target as Element).closest?.<HTMLAnchorElement>('a[href^="#"]');
    if (!a || e.defaultPrevented || e.metaKey || e.ctrlKey) return;
    e.preventDefault();
    const h = a.getAttribute("href")!.slice(1);
    if (h === "content") return document.getElementById("content")?.focus();
    if ((VIEWS as string[]).includes(h)) go(h as View);
    else {
      const host = document.getElementById(h)?.closest<HTMLElement>(".view")?.dataset.view as View | undefined;
      go(host ?? "today", h);
    }
  });
  route();

  // First-visit tips, dismissable and remembered.
  const welcome = document.getElementById("welcome");
  const help = () => {
    if (!welcome) return;
    welcome.hidden = false;
    if (current !== "today") go("today");
  };
  if (welcome && store.get(KEY_WELCOME) !== "1") welcome.hidden = false;
  document.getElementById("welcome-close")?.addEventListener("click", () => {
    welcome!.hidden = true;
    store.set(KEY_WELCOME, "1");
  });
  document.getElementById("help-open")?.addEventListener("click", help);

  // While the lake is being dragged or zoomed, overlays step back.
  let timer = 0;
  const calm = () => {
    clearTimeout(timer);
    timer = window.setTimeout(() => stage.classList.remove("interacting"), 900);
  };
  for (const id of ["canvas3d", "canvas"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener("pointerdown", () => {
      clearTimeout(timer);
      stage.classList.add("interacting");
    });
    el.addEventListener("pointerup", calm);
    el.addEventListener("pointercancel", calm);
    el.addEventListener("wheel", () => {
      stage.classList.add("interacting");
      calm();
    }, { passive: true });
  }
}

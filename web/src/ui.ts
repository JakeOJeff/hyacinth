// Page chrome behaviour: the sliding tab thumb, the collapsible info card,
// fading the overlays while the user handles the lake, and scroll reveals.

const STORE_KEY = "hyacinth.hud";

/** Slide the tab pill under the selected tab. */
export function moveTabThumb(): void {
  const tabs = document.querySelector<HTMLElement>(".tabs");
  const sel = tabs?.querySelector<HTMLElement>('button[aria-selected="true"]');
  if (!tabs || !sel) return;
  tabs.style.setProperty("--x", `${sel.offsetLeft}px`);
  tabs.style.setProperty("--w", `${sel.offsetWidth}px`);
}

export function initChrome(stage: HTMLElement): void {
  // Tab thumb follows layout changes (resize, web fonts arriving).
  addEventListener("resize", moveTabThumb);
  document.fonts?.ready.then(moveTabThumb).catch(() => {});

  // Info card: collapsible, remembered per viewer; starts collapsed on short screens.
  const hud = document.getElementById("hud");
  const toggle = document.getElementById("hud-toggle");
  if (hud && toggle) {
    let open = innerHeight > 700;
    try {
      const saved = localStorage.getItem(STORE_KEY);
      if (saved) open = saved === "open";
    } catch {
      /* storage unavailable */
    }
    const set = (v: boolean) => {
      hud.dataset.open = String(v);
      toggle.setAttribute("aria-expanded", String(v));
    };
    set(open);
    toggle.addEventListener("click", () => {
      const v = hud.dataset.open !== "true";
      set(v);
      try {
        localStorage.setItem(STORE_KEY, v ? "open" : "closed");
      } catch {
        /* storage unavailable */
      }
    });
  }

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
    el.addEventListener(
      "wheel",
      () => {
        stage.classList.add("interacting");
        calm();
      },
      { passive: true },
    );
  }

  // Dashboard cards rise in as they scroll into view, staggered per batch.
  const cards = [...document.querySelectorAll<HTMLElement>("[data-reveal]")];
  if (!("IntersectionObserver" in window)) {
    for (const c of cards) c.classList.add("in");
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      let k = 0;
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const el = e.target as HTMLElement;
        el.style.setProperty("--d", `${k++ * 80}ms`);
        el.classList.add("in");
        io.unobserve(el);
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );
  for (const c of cards) io.observe(c);
}

"""
test_ui_layout.py — measure the rendered page instead of eyeballing the CSS.

    python tests/test_ui_layout.py

===============================================================================
⚠️ WHAT THIS CATCHES, AND WHAT IT CANNOT
===============================================================================
Almost everything called "ugly" or "overlapping" is MEASURABLE without seeing it:

  horizontal overflow   documentElement.scrollWidth > viewport
  content off-screen    a bounding box past the right or left edge
  clipped content       scrollWidth > clientWidth on a non-scrolling element
  overlapping boxes     two siblings in a row intersecting where they should not
  cramped targets       a click target under 24px tall

⚠️⚠️ BUT IT HAS A FLOOR, AND THE FLOOR CAUGHT ME OUT.

Chrome will not render a window narrower than about 500px. The first version of this
audit asked for 360 and 480, silently got 500, measured a layout that fits, and
reported "no problems" — ⚠️ a FALSE PASS on exactly the widths that were broken.

So every width is now ASSERTED against the viewport actually reported by the page,
and a clamped width is reported as NOT TESTED rather than as a pass.

⚠️ AND THE BUG THAT PROMPTED THIS WAS FOUND BY ARITHMETIC, NOT BY THIS FILE.
`input[type=text]{min-width:340px}` needed 340 + 24 + 48 = 436px, so it overflowed
by 76px at 360px — inside the range Chrome refuses to render. Reading the CSS and
doing the sum found what the browser could not reach. **Both checks are needed.**
"""

from __future__ import annotations

import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ⚠️ 500 is Chrome's floor, included so the clamp is visible rather than a surprise.
WIDTHS = [1280, 1024, 768, 500]

DRIVER = r"""
(async () => {
  const t = (ms) => new Promise((r) => setTimeout(r, ms));
  const out = [];
  // ⚠️ populate first: an empty page has no table to overflow
  try {
    await ask("how do I bake sourdough bread"); await t(250);
    await ask("why does hybrid retrieval exist"); await t(250);
    document.querySelector("#idemBtn").click(); await t(2200);
    document.querySelector("#recBtn").click(); await t(1000);
  } catch (e) { out.push("setup-exception:" + e.message); }

  const vw = window.innerWidth, de = document.documentElement;
  if (de.scrollWidth > vw + 1) out.push("PAGE-OVERFLOW scrollWidth=" + de.scrollWidth + ">" + vw);

  const bad = [];
  for (const el of document.querySelectorAll("*")) {
    const r = el.getBoundingClientRect();
    if (!r.width && !r.height) continue;
    const id = el.tagName.toLowerCase() + (el.id ? "#" + el.id : "");
    if (r.right > vw + 1) bad.push(id + " right=" + Math.round(r.right));
    else if (r.left < -1) bad.push(id + " left=" + Math.round(r.left));
    const ox = getComputedStyle(el).overflowX;
    if (el.scrollWidth > el.clientWidth + 2 && ox !== "auto" && ox !== "scroll")
      bad.push(id + " clipped " + el.scrollWidth + ">" + el.clientWidth);
  }
  out.push("overflow:" + (bad.length ? bad.slice(0, 5).join(" ;; ") : "none"));

  const ov = [];
  for (const row of document.querySelectorAll(".row, .stat")) {
    const k = [...row.children].filter((c) => c.getBoundingClientRect().width > 0);
    for (let i = 0; i < k.length; i++) for (let j = i + 1; j < k.length; j++) {
      const a = k[i].getBoundingClientRect(), b = k[j].getBoundingClientRect();
      if (Math.min(a.right,b.right) - Math.max(a.left,b.left) > 2 &&
          Math.min(a.bottom,b.bottom) - Math.max(a.top,b.top) > 2)
        ov.push(".row child" + i + "/" + j);
    }
  }
  out.push("overlap:" + (ov.length ? ov.slice(0, 3).join(",") : "none"));

  const sm = [];
  for (const b of document.querySelectorAll("button, select, input[type=text]")) {
    const r = b.getBoundingClientRect();
    if (r.width > 0 && r.height < 24) sm.push((b.id || b.tagName) + "=" + Math.round(r.height));
  }
  out.push("small:" + (sm.length ? sm.join(",") : "none"));
  out.push("viewport:" + vw);
  document.title = "LAYOUT::" + out.join("|");
})();
"""


def main() -> int:
    print("test_ui_layout.py — the rendered layout, measured")

    if not pathlib.Path(CHROME).exists():
        print("\n  ⚠️  Chrome not found — SKIPPING, not failing.\n")
        return 0
    try:
        import fastapi  # noqa: F401
    except ImportError:                                                 # pragma: no cover
        print("\n  ⚠️  fastapi is not installed — SKIPPING.\n")
        return 0

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    page = ROOT / "ui" / "index.html"
    original = page.read_text(encoding="utf-8")
    page.write_text(original.replace("</body>", f"<script>{DRIVER}</script>\n</body>", 1), encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--app-dir", "src",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env={**os.environ, "PYTHONWARNINGS": "ignore"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    failures: list[str] = []
    try:
        for _ in range(40):
            time.sleep(0.25)
            try:
                with socket.create_connection(("127.0.0.1", port), 0.3):
                    break
            except OSError:
                continue

        for w in WIDTHS:
            dom = subprocess.run(
                [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                 f"--window-size={w},2000", "--virtual-time-budget=20000", "--dump-dom",
                 f"http://127.0.0.1:{port}/"],
                capture_output=True, text=True, timeout=90).stdout
            m = re.search(r"<title>LAYOUT::(.*?)</title>", dom, re.S)
            if not m:
                print(f"  {w:>5}px  ⚠️ audit did not finish")
                failures.append(f"{w}:did-not-finish")
                continue

            parts = dict(x.split(":", 1) for x in m.group(1).split("|") if ":" in x)
            real = parts.get("viewport", "?")
            clamped = str(w) != real
            note = f"  ⚠️ CHROME CLAMPED {w}→{real} — NOT TESTED" if clamped else ""
            print(f"  {w:>5}px  viewport={real}{note}")
            if clamped:
                continue
            for k in ("overflow", "overlap", "small"):
                v = parts.get(k, "none")
                if v != "none":
                    print(f"          ⚠️ {k}: {v}")
                    failures.append(f"{w}:{k}")
            if "PAGE-OVERFLOW" in m.group(1):
                print("          ⚠️ the page scrolls horizontally")
                failures.append(f"{w}:page-overflow")
    finally:
        page.write_text(original, encoding="utf-8")
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)

    print()
    if failures:
        print(f"  ❌ {len(failures)} layout problem(s): {', '.join(failures)}")
        return len(failures)
    print("  ✅ no layout problem measured at any width Chrome will render")
    print("  ⚠️ widths below ~500px are NOT covered — check those by arithmetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

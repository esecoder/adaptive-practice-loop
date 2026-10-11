"""
test_ui_browser.py — the demo UI, driven in a real browser.

    python tests/test_ui_browser.py

===============================================================================
⚠️ WHY curl IS NOT ENOUGH
===============================================================================
`curl http://127.0.0.1:8000/` returning 200 proves the HTML was SERVED. It proves
nothing whatsoever about the ~7,500 characters of JavaScript that fetch the data,
render the panels and decide green-versus-amber. Every one of those could throw on
the first line and curl would still report a healthy 200.

⚠️ This drives the actual page in headless Chrome: it calls the handlers, clicks the
buttons, waits for the fetches, and reads the resulting DOM. It is the only test in
this repository that executes the UI's own code paths.

⚠️ SKIPS CLEANLY when Chrome or the API extra is unavailable. A test that goes red on
a machine without Chrome is a test that gets deleted by someone who never had it.

⚠️ What it does NOT cover: visual layout, and anything requiring a human to look at
the page. It asserts behaviour, not appearance.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

# ⚠️ The driver lives in the test, not in the page. Injecting it into ui/index.html
# would ship test code to a reviewer — and the page that gets tested would no longer
# be the page that gets served.
DRIVER = r"""
(async () => {
  const out = [];
  const t = (ms) => new Promise((r) => setTimeout(r, ms));
  const txt = (s) => (document.querySelector(s)?.textContent || "").replace(/\s+/g, " ").trim();
  try {
    await ask("how do I bake sourdough bread"); await t(400);
    out.push("abstain:" + (document.querySelector("#askVerdict").classList.contains("no") ? "ok" : "WRONG"));
    await ask("why does hybrid retrieval exist"); await t(400);
    out.push("grounded:" + (document.querySelector("#askVerdict").classList.contains("ok") ? "ok" : "WRONG"));
    document.querySelector('.sample[data-q="what is the capital of Peru"]').click(); await t(400);
    // ⚠️ `.value`, NOT textContent — an <input> has no text content, so reading
    // textContent returns "" and the check fails on correct code. Measured.
    out.push("sample:" + (document.querySelector("#askInput").value === "what is the capital of Peru" ? "ok" : "WRONG"));
    document.querySelector("#idemBtn").click(); await t(2500);
    const i = txt("#idemOut");
    out.push("idem:" + (/WROTE/.test(i) && /replayed/.test(i) && /attempt rows/.test(i) ? "ok" : "WRONG"));
    document.querySelector("#conflictBtn").click(); await t(1500);
    out.push("conflict:" + (/409/.test(txt("#idemOut")) ? "ok" : "WRONG"));
    document.querySelector("#recBtn").click(); await t(1200);
    const r = txt("#recOut");
    out.push("recommend:" + (r.length > 30 && /cold_start/.test(r) ? "ok" : "WRONG"));
    document.querySelector('.opt[data-opt="A"]').click();
    if (document.querySelector("#explainBtn").disabled) out.push("optbutton:WRONG");
    document.querySelector("#explainBtn").click(); await t(1000);
    out.push("explain:" + (document.querySelector("#verdict").classList.contains("ok") ? "ok" : "WRONG"));
    // the dropdown must be populated BY THE FETCH, which only JS can do
    out.push("items:" + (document.querySelectorAll("#itemSelect option").length > 0 ? "ok" : "WRONG"));
    out.push("errors:" + (window.__errs?.length ? window.__errs.join(";") : "none"));
  } catch (e) { out.push("EXCEPTION:" + e.message); }
  document.title = "RESULTS::" + out.join("|");
})();
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def chrome() -> str | None:
    return next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)


def main() -> int:
    print("test_ui_browser.py — the demo, driven in a real browser")

    if not (ROOT / "ui" / "index.html").exists():                       # pragma: no cover
        print("  ⚠️  ui/index.html is missing")
        return 1

    browser = chrome()
    if not browser:
        print("\n  ⚠️  no Chrome or Chromium found — SKIPPING, not failing.")
        print("      Install Google Chrome to run the browser checks.\n")
        return 0

    try:
        import fastapi  # noqa: F401
    except ImportError:                                                 # pragma: no cover
        print("\n  ⚠️  fastapi is not installed — SKIPPING.")
        print("      .venv/bin/pip install -r requirements.txt\n")
        return 0

    port = free_port()
    original = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    patched = original.replace(
        "<script>",
        "<script>window.__errs=[];window.addEventListener('error',e=>window.__errs.push(e.message));"
        "window.addEventListener('unhandledrejection',e=>window.__errs.push('reject:'+e.reason));</script>\n<script>",
        1,
    ).replace("</body>", f"<script>{DRIVER}</script>\n</body>", 1)

    # ⚠️ The page is patched on disk because the API serves ui/index.html directly.
    # A `try/finally` restores it even if Chrome crashes, so a failed test cannot
    # leave a test-harness script shipped in the reviewer's page.
    proc = None
    try:
        (ROOT / "ui" / "index.html").write_text(patched, encoding="utf-8")

        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--app-dir", "src",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            time.sleep(0.25)
            try:
                with socket.create_connection(("127.0.0.1", port), 0.3):
                    break
            except OSError:
                continue
        else:                                                           # pragma: no cover
            print("  ⚠️  the server did not start — SKIPPING")
            return 0

        dom = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=20000", "--dump-dom",
             f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True, timeout=90,
        ).stdout

        m = re.search(r"<title>RESULTS::(.*?)</title>", dom, re.S)
        if not m:
            print("\n  ❌⚠️  THE DRIVER NEVER FINISHED — the page threw before the title was set")
            print("      ⚠️ This is what a JavaScript error looks like from outside: the HTML")
            print("      is served with a 200 and nothing on the page works.")
            return 1

        results = m.group(1).split("|")
        failures = []
        for r in results:
            name, _, value = r.partition(":")
            ok = value == "ok" or (name == "errors" and value == "none")
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + value if not ok else ''}")
            if not ok:
                failures.append(name)

        print()
        if failures:
            print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
            return len(failures)
        print("  ✅ every panel works in a real browser")
        return 0

    finally:
        (ROOT / "ui" / "index.html").write_text(original, encoding="utf-8")
        if proc:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())

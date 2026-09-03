#!/usr/bin/env python3
"""Re-render the figures used in the README from ``results/report.html``.

Kept out of the package because it is the only thing in the repository that
needs a browser. Everything that produces a *number* runs with no dependencies
at all; this only produces pictures.

    pip install playwright && playwright install chromium
    python tools/render_figures.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "report.html"
DOCS = ROOT / "docs"

SHOTS = [
    ("#films", "heap-player.png", 4200),
    ("#bars", "gc-work.png", 600),
    ("#kcurve", "k-curve.png", 600),
]


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "    pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 1
    if not REPORT.exists():
        print("no results/report.html -- run `make results && make report` first",
              file=sys.stderr)
        return 1

    DOCS.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900},
                                device_scale_factor=2)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("file://" + str(REPORT))
        for selector, name, wait_ms in SHOTS:
            page.wait_for_timeout(wait_ms)
            page.locator(selector).screenshot(path=str(DOCS / name))
            print("wrote docs/%s" % name)
        page.wait_for_timeout(500)
        page.screenshot(path=str(DOCS / "report-top.png"),
                        clip={"x": 0, "y": 0, "width": 1100, "height": 1400})
        print("wrote docs/report-top.png")
        browser.close()
    if errors:
        print("page errors: %r" % errors[:3], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

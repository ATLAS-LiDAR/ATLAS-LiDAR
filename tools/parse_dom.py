#!/usr/bin/env python3
"""Extract the test report and completion marker from a --dump-dom capture.

Chrome's --dump-dom emits <script> source verbatim, so a naive grep for the
completion marker matches the fallback string literal inside tests.html rather
than the element the harness actually wrote. Anchor on the elements by id.

stdout: the human-readable report.
Exit code: 0 all passing, 1 failures, 2 suite never finished.
"""
import html
import re
import sys


def element(dom, tag, ident):
    m = re.search(
        r'<%s[^>]*\bid="%s"[^>]*>(.*?)</%s>' % (tag, ident, tag), dom, re.S
    )
    return html.unescape(m.group(1)) if m else None


def main():
    dom = sys.stdin.read()

    report = element(dom, "pre", "atlas-results")
    if report is None:
        # Fall back to any <pre> so load errors still surface.
        pres = re.findall(r"<pre[^>]*>(.*?)</pre>", dom, re.S)
        report = html.unescape("\n".join(pres)) if pres else ""

    marker = element(dom, "div", "atlas-done")
    if marker is None:
        sys.stdout.write(report)
        sys.stderr.write(
            "\nrun-tests: suite did not finish - no #atlas-done element in DOM\n"
        )
        return 2

    m = re.search(r"pass=(\d+)\s+fail=(\d+)", marker)
    if not m:
        sys.stdout.write(report)
        sys.stderr.write("\nrun-tests: unparseable marker %r\n" % marker)
        return 2

    npass, nfail = int(m.group(1)), int(m.group(2))
    sys.stdout.write(report.rstrip("\n") + "\n")
    if nfail:
        sys.stderr.write("run-tests: %d failing, %d passing\n" % (nfail, npass))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

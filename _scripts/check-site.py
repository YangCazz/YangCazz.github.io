"""Check generated local links/assets (including CSS and manifest), without network access."""
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


def css_references(content):
    """Yield CSS url() and quoted @import targets with their line offsets."""
    patterns = (
        r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)",
        r"@import\s+['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            yield match[1].strip(), content.count("\n", 0, match.start())


class References(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references = []
        self.ids = []
        self.named_anchors = []
        self.in_style = False
        self.top_navigation = []
        self.nav_stack = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "nav":
            self.nav_stack.append("top-nav" in attrs.get("class", "").split() or any(self.nav_stack))
        if tag == "a" and any(self.nav_stack) and attrs.get("href"):
            self.top_navigation.append((attrs["href"], self.getpos()[0]))
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.named_anchors.append(attrs["name"])
        if tag == "style":
            self.in_style = True
        if attrs.get("style"):
            self.references.extend((url, self.getpos()[0] + offset)
                                   for url, offset in css_references(attrs["style"]))
        for attribute in ("href", "src", "poster"):
            if attrs.get(attribute):
                self.references.append((attrs[attribute], self.getpos()[0]))
        if attrs.get("srcset"):
            for item in attrs["srcset"].split(","):
                if item.strip():
                    self.references.append((item.strip().split()[0], self.getpos()[0]))
        if tag == "meta" and attrs.get("property", attrs.get("name")) in ("og:image", "twitter:image"):
            if attrs.get("content"):
                self.references.append((attrs["content"], self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag == "nav" and self.nav_stack:
            self.nav_stack.pop()
        if tag == "style":
            self.in_style = False

    def handle_data(self, content):
        if self.in_style:
            self.references.extend((url, self.getpos()[0] + offset)
                                   for url, offset in css_references(content))


def local_url(source, reference, site_url):
    reference = reference.strip()
    if not reference or reference == "#" or reference.startswith(("data:", "mailto:", "tel:", "javascript:")):
        return None
    site_url = site_url.rstrip("/") + "/"
    parsed = urlsplit(urljoin(urljoin(site_url, source), reference))
    if parsed.scheme not in ("http", "https") or parsed.netloc != urlsplit(site_url).netloc:
        return None
    return parsed


def local_target(site, source, reference, site_url):
    """Resolve same-origin URLs exactly as a static Jekyll output directory does."""
    parsed = local_url(source, reference, site_url)
    if parsed is None:
        return None
    relative = unquote(parsed.path).lstrip("/")
    target = site / relative
    if target.is_dir():
        target = target / "index.html"
    elif not target.suffix and target.with_suffix(".html").is_file():
        target = target.with_suffix(".html")
    return target


def check_site(site, site_url="https://yangcazz.github.io"):
    site = Path(site)
    if not (site / "index.html").is_file():
        return {"checked": 0, "errors": ["Build the site first: missing index.html"]}
    references = []
    errors = []
    html_count = 0
    fragment_targets = {}
    for file in sorted(site.rglob("*.html")):
        html_count += 1
        source = file.relative_to(site).as_posix()
        parser = References()
        parser.feed(file.read_text(encoding="utf-8"))
        fragment_targets[file.resolve()] = set(parser.ids + parser.named_anchors)
        references.extend((source, url, line) for url, line in parser.references)
        # Standalone apps do not inherit the shared layout. Keep their top bars
        # consistent without removing the resume page or its footer links.
        for reference, line in parser.top_navigation:
            parsed = local_url(source, reference, site_url)
            if parsed and re.search(r"/resume(?:\.html)?/?$", parsed.path):
                errors.append(f"{source}:{line}: resume must not appear in top navigation")
        # The blog's live navigation depends on these IDs and a single JSON dataset.
        if source == "blog/index.html":
            for identifier in ("tag-network", "blog-list", "blog-footer", "calendarData"):
                if parser.ids.count(identifier) != 1:
                    errors.append(f"{source}: expected exactly one #{identifier}")
    for file in sorted(site.rglob("*.css")):
        source = file.relative_to(site).as_posix()
        content = file.read_text(encoding="utf-8")
        references.extend((source, url, offset + 1) for url, offset in css_references(content))
    manifest_path = site / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            references.extend(("manifest.json", icon["src"], 0) for icon in manifest.get("icons", []))
            references.append(("manifest.json", manifest.get("start_url", "/"), 0))
        except (ValueError, KeyError) as error:
            errors.append(f"manifest.json: {error}")
    checked = 0
    for source, reference, line in references:
        target = local_target(site, source, reference, site_url)
        if target is None:
            continue
        checked += 1
        if not target.is_file():
            errors.append(f"{source}:{line}: missing {reference}")
            continue
        # Text fragments are browser search directives, not element IDs. Binary
        # media fragments (e.g. a PDF page or font #iefix) cannot be checked as HTML.
        fragment = unquote(local_url(source, reference, site_url).fragment).split(":~:", 1)[0]
        if fragment and target.suffix.lower() in (".html", ".htm", ".svg"):
            resolved = target.resolve()
            if resolved not in fragment_targets:
                parser = References()
                parser.feed(target.read_text(encoding="utf-8"))
                fragment_targets[resolved] = set(parser.ids + parser.named_anchors)
            if fragment not in fragment_targets[resolved]:
                errors.append(f"{source}:{line}: missing fragment {reference}")
    return {"html_files": html_count, "checked": checked, "errors": sorted(set(errors))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", default="_site")
    parser.add_argument("--site-url", default="https://yangcazz.github.io")
    arguments = parser.parse_args()
    report = check_site(arguments.site, arguments.site_url)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(report["errors"]))

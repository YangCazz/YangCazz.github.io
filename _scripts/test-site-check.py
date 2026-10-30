import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("site_check", Path(__file__).with_name("check-site.py"))
site_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_check)


class SiteCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.site = Path(self.directory.name)
        (self.site / "index.html").write_text('<a href="/post/">Post</a><img src="/asset.png">')
        (self.site / "post").mkdir()
        (self.site / "post/index.html").write_text('<a href="../">Home</a>')
        (self.site / "asset.png").write_bytes(b"fixture")

    def test_valid_local_routes_and_external_links(self):
        with (self.site / "index.html").open("a") as file:
            file.write('<a href="https://example.org/unverified">External</a><a href="#section">Anchor</a><h2 id="section">Section</h2><a href="#">Placeholder</a>')
        self.assertEqual(site_check.check_site(self.site)["errors"], [])

    def test_missing_article_route_fails(self):
        (self.site / "index.html").write_text('<a href="/2025/02/01/missing/">Wrong date</a>')
        self.assertIn("missing /2025/02/01/missing/", site_check.check_site(self.site)["errors"][0])

    def test_manifest_nested_icon_is_checked(self):
        (self.site / "manifest.json").write_text('{"icons":[{"src":"/absent.svg"}],"start_url":"/"}')
        self.assertTrue(any("absent.svg" in error for error in site_check.check_site(self.site)["errors"]))

    def test_css_relative_resource_is_checked(self):
        (self.site / "styles.css").write_text("body { background: url('absent.png'); }")
        self.assertTrue(any("absent.png" in error for error in site_check.check_site(self.site)["errors"]))

    def test_inline_style_attributes_and_blocks_are_checked(self):
        (self.site / "index.html").write_text('<div style="background-image: url(/missing-cover.png)"></div><style>body { background: url("missing-style.png"); }</style>')
        errors = site_check.check_site(self.site)["errors"]
        self.assertTrue(any("missing-cover.png" in error for error in errors))
        self.assertTrue(any("missing-style.png" in error for error in errors))

    def test_css_quoted_imports_and_url_imports_are_checked(self):
        (self.site / "styles.css").write_text('@import "missing-quoted.css" screen; @import url("missing-url.css");')
        errors = site_check.check_site(self.site)["errors"]
        self.assertTrue(any("missing-quoted.css" in error for error in errors))
        self.assertTrue(any("missing-url.css" in error for error in errors))

    def test_same_page_and_cross_page_missing_fragments_fail(self):
        (self.site / "index.html").write_text('<a href="#missing-local">Local</a><a href="/post/#missing-remote">Cross-page</a>')
        errors = site_check.check_site(self.site)["errors"]
        self.assertTrue(any("missing fragment #missing-local" in error for error in errors))
        self.assertTrue(any("missing fragment /post/#missing-remote" in error for error in errors))

    def test_valid_encoded_fragments_and_named_anchors_pass(self):
        (self.site / "post/index.html").write_text('<h2 id="中文标题">Heading</h2><a name="legacy"></a>')
        (self.site / "index.html").write_text('<a href="/post/#%E4%B8%AD%E6%96%87%E6%A0%87%E9%A2%98">Unicode</a><a href="/post/#legacy">Legacy</a><a href="#:~:text=Example">Text fragment</a>')
        self.assertEqual(site_check.check_site(self.site)["errors"], [])

    def test_svg_fragment_is_checked_without_treating_font_fragments_as_ids(self):
        (self.site / "icons.svg").write_text('<svg><symbol id="home"></symbol></svg>')
        (self.site / "font.eot").write_bytes(b"font fixture")
        (self.site / "index.html").write_text('<svg><use href="/icons.svg#home"></use></svg><style>@font-face { src: url("font.eot?#iefix"); }</style>')
        self.assertEqual(site_check.check_site(self.site)["errors"], [])
        (self.site / "index.html").write_text('<svg><use href="/icons.svg#absent"></use></svg>')
        self.assertTrue(any("missing fragment /icons.svg#absent" in error for error in site_check.check_site(self.site)["errors"]))

    def test_blog_required_targets_and_duplicate_ids_are_checked(self):
        (self.site / "blog").mkdir()
        (self.site / "blog/index.html").write_text('<script id="calendarData"></script><script id="calendarData"></script>')
        errors = site_check.check_site(self.site)["errors"]
        self.assertTrue(any("#tag-network" in error for error in errors))
        self.assertTrue(any("#calendarData" in error for error in errors))

    def test_encoded_file_path(self):
        (self.site / "测试 图片.png").write_bytes(b"fixture")
        (self.site / "index.html").write_text('<img src="/%E6%B5%8B%E8%AF%95%20%E5%9B%BE%E7%89%87.png">')
        self.assertEqual(site_check.check_site(self.site)["errors"], [])


if __name__ == "__main__":
    unittest.main()

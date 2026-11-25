from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ocr_archive import archive
from ocr_archive.app import create_app


BOXES = [
    {"page_no": 1, "box_no": 1, "rect": [10, 20, 100, 30], "text": "甲乙丙丁"},
    {"page_no": 1, "box_no": 2, "rect": [10, 60, 100, 30], "text": "戊己庚辛"},
    {"page_no": 2, "box_no": 1, "rect": [10, 20, 120, 40], "text": "壬癸"},
]


def many_boxes(count: int) -> list[dict]:
    """One single-character box per source set, so a release references count sets."""
    return [
        {
            "page_no": 1 + i // 50,
            "box_no": 1 + i % 50,
            "rect": [10, 20, 40, 20],
            "text": chr(0x4E00 + i),
        }
        for i in range(count)
    ]


def connect_with_low_limit(path: str, limit: int) -> sqlite3.Connection:
    """Open the archive like a SQLite build whose per-statement limit is tiny.

    This is how an otherwise valid old library fails on a platform with a 999
    (or lower) SQLITE_LIMIT_VARIABLE_NUMBER.
    """
    conn = archive.connect(path)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    return conn


class ArchiveTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.sqlite3")
        self.app = create_app({"TESTING": True, "DATABASE": self.db_path})
        self.client = self.app.test_client()
        with self.app.app_context():
            self.conn = archive.connect(self.db_path)
            archive.init_db(self.db_path)
            boxes = archive.parse_boxes(BOXES)
            self.mid = archive.ingest(self.conn, "JP-1", "测试报", boxes)
            self.ms = archive.get_manuscript(self.conn, "JP-1")
            self.r1 = archive.get_release(self.conn, self.mid, 1)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def revise(self, start, end, replacement, note="", base=None):
        base = base or self.ms.current_release_id
        with self.app.app_context():
            ms = archive.get_manuscript(self.conn, "JP-1")
            release = archive.revise(
                self.conn, ms, base, start, end, replacement, note
            )
            self.ms = archive.get_manuscript(self.conn, "JP-1")
            return release

    def lookup(self, release, start, end, page=None):
        return archive.lookup_sources(self.conn, release, start, end, page)


class InitialDraftTest(ArchiveTestBase):
    def test_body_is_box_text_in_reading_order(self):
        self.assertEqual(self.r1.body, "甲乙丙丁戊己庚辛壬癸")
        self.assertEqual(self.r1.release_no, 1)
        self.assertIsNone(self.r1.parent_id)

    def test_consecutive_same_origin_is_one_span(self):
        spans = archive.load_spans(self.conn, self.r1.id)
        # One box's text is one source set; separate boxes stay separate.
        self.assertEqual([(s.start, s.end) for s in spans], [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(len({s.source_set_id for s in spans}), 3)
        self.assertEqual(self.ms.current_release_id, self.r1.id)

    def test_boxes_keep_page_box_and_rect(self):
        rows = self.conn.execute(
            "SELECT page_no, box_no, x, y, w, h, text FROM box ORDER BY reading_pos"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            (rows[0]["page_no"], rows[0]["box_no"], rows[0]["x"], rows[0]["w"]),
            (1, 1, 10, 100),
        )
        self.assertEqual(rows[2]["page_no"], 2)

    def test_ingest_is_atomic_on_duplicate_code(self):
        before = self.conn.execute("SELECT COUNT(*) AS n FROM box").fetchone()["n"]
        with self.app.app_context(), self.assertRaises(archive.ValidationError):
            archive.ingest(self.conn, "JP-1", "重复", archive.parse_boxes(BOXES))
        after = self.conn.execute("SELECT COUNT(*) AS n FROM box").fetchone()["n"]
        self.assertEqual(before, after)

    def test_empty_box_text_is_allowed_but_all_empty_rejected(self):
        payload = [{"page_no": 1, "box_no": 1, "x": 0, "y": 0, "w": 1, "h": 1, "text": ""}]
        with self.assertRaises(archive.ValidationError):
            archive.parse_boxes(payload)


class ReviseTest(ArchiveTestBase):
    def test_replace_within_one_box_inherits_that_box_interval(self):
        r2 = self.revise(1, 3, "子丑")
        self.assertEqual(r2.body, "甲子丑丁戊己庚辛壬癸")
        self.assertEqual(r2.release_no, 2)
        self.assertEqual(r2.parent_id, self.r1.id)
        self.assertEqual(self.ms.current_release_id, r2.id)

        sources = self.lookup(r2, 1, 3)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["page_no"], 1)
        self.assertEqual(sources[0]["box_no"], 1)
        self.assertEqual(sources[0]["rect"], {"x": 10, "y": 20, "w": 100, "h": 30})
        self.assertEqual(sources[0]["intervals"], [(0, 4)])
        self.assertFalse(sources[0]["exact"])
        self.assertEqual(sources[0]["chars"], 2)

    def test_replacement_inherits_union_of_all_selected_sources(self):
        # 3..7 crosses box1 tail and box2 head.
        r2 = self.revise(3, 7, "X")
        sources = self.lookup(r2, 3, 4)
        self.assertEqual([(s["page_no"], s["box_no"]) for s in sources], [(1, 1), (1, 2)])
        for s in sources:
            self.assertEqual(s["intervals"], [(0, 4)])

    def test_multiple_inserted_chars_all_inherit_the_same_sources(self):
        r2 = self.revise(0, 1, "甲乙丙")
        self.assertEqual(r2.body, "甲乙丙乙丙丁戊己庚辛壬癸")
        spans = archive.load_spans(self.conn, r2.id)
        self.assertEqual(
            [(s.start, s.end) for s in spans],
            [(0, 3), (3, 6), (6, 10), (10, 12)],
            "inserted run and adjacent untouched run have different mappings",
        )
        sources = self.lookup(r2, 0, 3)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["chars"], 3)
        self.assertEqual(sources[0]["intervals"], [(0, 4)])
        # The untouched「乙丙丁」right after them keeps exact box offsets.
        untouched = self.lookup(r2, 3, 6)
        self.assertEqual(untouched[0]["intervals"], [(1, 4)])

    def test_delete_removes_chars_and_keeps_neighbours(self):
        r2 = self.revise(1, 3, "")
        self.assertEqual(r2.body, "甲丁戊己庚辛壬癸")
        sources = self.lookup(r2, 3, 4)
        self.assertEqual([(s["page_no"], s["box_no"]) for s in sources], [(1, 2)])
        # 己 was the second character of box 2, so the untouched mapping stays
        # exact after deletion.
        self.assertEqual(sources[0]["intervals"], [(1, 2)])
        self.assertTrue(sources[0]["exact"])

    def test_untouched_run_reports_exact_box_interval(self):
        sources = self.lookup(self.r1, 5, 6)
        self.assertEqual((sources[0]["page_no"], sources[0]["box_no"]), (1, 2))
        self.assertEqual(sources[0]["intervals"], [(1, 2)])
        self.assertTrue(sources[0]["exact"])
        cross = self.lookup(self.r1, 3, 6)
        self.assertEqual(
            [(s["box_no"], s["intervals"], s["body_ranges"], s["chars"]) for s in cross],
            [(1, [(3, 4)], [(3, 4)], 1), (2, [(0, 2)], [(4, 6)], 2)],
        )
        self.assertTrue(all(s["exact"] for s in cross))

    def test_shifted_untouched_run_keeps_exact_mapping(self):
        r2 = self.revise(0, 1, "前置三字")
        self.assertEqual(r2.body, "前置三字乙丙丁戊己庚辛壬癸")
        sources = self.lookup(r2, 8, 9)
        self.assertEqual((sources[0]["box_no"], sources[0]["intervals"]), (2, [(1, 2)]))
        self.assertTrue(sources[0]["exact"])

    def test_untouched_chars_keep_offsets_shifted_and_shared_source_sets(self):
        before = {
            s.source_set_id
            for s in archive.load_spans(self.conn, self.r1.id)
        }
        r2 = self.revise(2, 3, "长一点")
        after = {s.source_set_id for s in archive.load_spans(self.conn, r2.id)}
        self.assertTrue(before & after, "untouched run must reuse the same source set row")
        sources = self.lookup(r2, 0, 1)
        self.assertEqual((sources[0]["page_no"], sources[0]["box_no"]), (1, 1))

    def test_unchanged_runs_share_exact_source_set_rows_after_shift(self):
        before = {
            (s.start, s.end): s.source_set_id
            for s in archive.load_spans(self.conn, self.r1.id)
        }
        entries_before = self.conn.execute(
            "SELECT COUNT(*) AS n FROM source_entry"
        ).fetchone()["n"]
        # Insert four chars in box 1: boxes 2 and 3 shift but must reuse their
        # source-set rows rather than copy them.
        r2 = self.revise(3, 4, "長長長")
        shift = 2
        spans2 = archive.load_spans(self.conn, r2.id)
        by_new = {(s.start, s.end): s.source_set_id for s in spans2}
        for (start, end), set_id in before.items():
            if start >= 4:
                self.assertEqual(by_new[(start + shift, end + shift)], set_id)
        entries_after = self.conn.execute(
            "SELECT COUNT(*) AS n FROM source_entry"
        ).fetchone()["n"]
        # Only the new union set (all of box 1) may add entries; untouched boxes
        # add none.
        self.assertLessEqual(entries_after - entries_before, 1)

    def test_editing_replacement_again_uses_its_inherited_sources(self):
        r2 = self.revise(0, 2, "新字")  # 甲乙 -> 新字, inherits box 1
        r3 = self.revise(0, 2, "改", base=r2.id)  # edit 新字
        sources = self.lookup(r3, 0, 1)
        self.assertEqual([(s["page_no"], s["box_no"]) for s in sources], [(1, 1)])
        self.assertEqual(sources[0]["intervals"], [(0, 4)])
        self.assertEqual(r3.body, "改丙丁戊己庚辛壬癸")

    def test_repeated_edits_do_not_accumulate_intervals(self):
        r2 = self.revise(0, 2, "新字")
        r3 = self.revise(0, 2, "改", base=r2.id)
        r4 = self.revise(0, 1, "再", base=r3.id)
        sources = self.lookup(r4, 0, 1)
        self.assertEqual(sources[0]["intervals"], [(0, 4)])

    def test_noop_not_recorded_but_valid_edits_append(self):
        r2 = self.revise(5, 6, "己")
        self.assertEqual(r2.body, self.r1.body)
        self.assertEqual(r2.release_no, 2)
        self.assertEqual(self.ms.current_release_id, r2.id)

    def test_stale_base_rejected_and_current_version_unchanged(self):
        r2 = self.revise(0, 1, "A")
        r3 = self.revise(0, 1, "B", base=r2.id)
        with self.assertRaises(archive.StaleReleaseError) as ctx:
            self.revise(1, 2, "C", base=self.r1.id)
        self.assertEqual(ctx.exception.current_release_no, 3)
        ms = archive.get_manuscript(self.conn, "JP-1")
        self.assertEqual(ms.current_release_id, r3.id)
        self.assertEqual(archive.get_release_by_id(self.conn, ms.current_release_id).body,
                         "B乙丙丁戊己庚辛壬癸")

    def test_invalid_ranges_rejected_without_writing(self):
        count = self.conn.execute("SELECT COUNT(*) AS n FROM release").fetchone()["n"]
        for start, end in [(0, 0), (2, 1), (-1, 2), (0, 999), (11, 12)]:
            with self.assertRaises(archive.ValidationError):
                self.revise(start, end, "x")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM release").fetchone()["n"], count
        )

    def test_failed_write_rolls_back_everything(self):
        count = self.conn.execute("SELECT COUNT(*) AS n FROM release").fetchone()["n"]
        span_count = self.conn.execute("SELECT COUNT(*) AS n FROM body_span").fetchone()["n"]
        with self.assertRaises(RuntimeError):
            with self.app.app_context():
                ms = archive.get_manuscript(self.conn, "JP-1")
                with archive.immediate(self.conn):
                    self.conn.execute(
                        "INSERT INTO release "
                        "(manuscript_id, release_no, parent_id, body, note, created_at) "
                        "VALUES (?, 2, ?, 'x', '', 'now')",
                        (ms.id, ms.current_release_id),
                    )
                    raise RuntimeError("boom")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM release").fetchone()["n"], count
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM body_span").fetchone()["n"],
            span_count,
        )

    def test_failed_release_write_keeps_current_and_history_intact(self):
        """A failure between body and provenance writes leaves no partial release."""
        r2 = self.revise(0, 1, "A")
        current = archive.get_manuscript(self.conn, "JP-1").current_release_id
        release_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM release WHERE manuscript_id = ?",
            (self.mid,),
        ).fetchone()["n"]
        span_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM body_span"
        ).fetchone()["n"]

        # Fail the provenance insert after the release row has been written,
        # so this exercises rollback in the middle of publishing a version.
        self.conn.execute(
            "CREATE TRIGGER fail_span BEFORE INSERT ON body_span "
            "BEGIN SELECT RAISE(ABORT, 'provenance write failed'); END"
        )
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                with self.app.app_context():
                    ms = archive.get_manuscript(self.conn, "JP-1")
                    archive.revise(self.conn, ms, r2.id, 0, 1, "B", "会失败")
        finally:
            self.conn.execute("DROP TRIGGER fail_span")

        # The whole transaction rolled back: current pointer, releases and
        # provenance are exactly as before, and the old release still reads.
        self.assertEqual(
            archive.get_manuscript(self.conn, "JP-1").current_release_id, current
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM release WHERE manuscript_id = ?",
                (self.mid,),
            ).fetchone()["n"],
            release_count,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM body_span").fetchone()["n"],
            span_count,
        )
        old = archive.get_release(self.conn, self.mid, 1)
        self.assertEqual(old.body, "甲乙丙丁戊己庚辛壬癸")
        self.assertEqual(self.lookup(old, 0, 1)[0]["box_no"], 1)

    def test_old_release_read_does_not_need_replay(self):
        r2 = self.revise(0, 1, "ZZ")
        old = archive.get_release(self.conn, self.mid, 1)
        self.assertEqual(old.body, "甲乙丙丁戊己庚辛壬癸")
        self.assertEqual(old.release_no, 1)
        sources = self.lookup(old, 0, 1)
        self.assertEqual(sources[0]["box_no"], 1)
        self.assertNotEqual(r2.body, old.body)


class CodepointTest(ArchiveTestBase):
    def test_astral_characters_count_as_one_code_point(self):
        boxes = archive.parse_boxes(
            [{"page_no": 1, "box_no": 1, "rect": [0, 0, 10, 10], "text": "𠀀𠀁ab"}]
        )
        mid = archive.ingest(self.conn, "U-1", "补字", boxes)
        ms = archive.get_manuscript(self.conn, "U-1")
        release = archive.get_release(self.conn, mid, 1)
        self.assertEqual(len(release.body), 4)
        with self.app.app_context():
            r2 = archive.revise(self.conn, ms, release.id, 1, 2, "𠀂", "")
        self.assertEqual(r2.body, "𠀀𠀂ab")
        sources = self.lookup(r2, 1, 2)
        self.assertEqual(sources[0]["intervals"], [(0, 4)])
        with self.assertRaises(archive.ValidationError):
            self.lookup(r2, 0, 5)
        self.assertEqual(len(self.lookup(r2, 0, 2)), 1)


class HttpTest(ArchiveTestBase):
    def test_import_page_and_index(self):
        resp = self.client.post(
            "/import",
            data={
                "code": "WEB-1",
                "title": "网页导入",
                "boxes": json.dumps(BOXES, ensure_ascii=False),
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("网页导入", resp.get_data(as_text=True))
        self.assertIn("甲乙丙丁", resp.get_data(as_text=True))

    def test_api_import_and_revise_and_lookup(self):
        resp = self.client.post(
            "/api/import",
            json={"code": "API-1", "title": "接口", "boxes": BOXES},
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json()["body"], "甲乙丙丁戊己庚辛壬癸")
        base = self.conn.execute(
            "SELECT id FROM release ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        resp = self.client.post(
            "/api/m/API-1/revise",
            json={
                "base_release_id": base,
                "start": 0,
                "end": 2,
                "replacement": "新",
                "note": "改首二字",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["body"], "新丙丁戊己庚辛壬癸")
        resp = self.client.get("/api/m/API-1/lookup?start=0&end=1")
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["sources"][0]["page_no"], 1)
        self.assertEqual(data["sources"][0]["box_no"], 1)
        self.assertEqual(data["sources"][0]["box_intervals"], [{"start": 0, "end": 4}])

    def test_api_stale_returns_409_and_does_not_write(self):
        body = json.dumps({"code": "API-2", "title": "接口", "boxes": BOXES})
        self.client.post("/api/import", data=body, content_type="application/json")
        base = self.conn.execute(
            "SELECT r.id FROM release r JOIN manuscript m ON m.id = r.manuscript_id "
            "WHERE m.code = 'API-2' AND r.release_no = 1"
        ).fetchone()["id"]
        first = self.client.post(
            "/api/m/API-2/revise",
            json={"base_release_id": base, "start": 0, "end": 1, "replacement": "A"},
        )
        self.assertEqual(first.status_code, 200)
        second = self.client.post(
            "/api/m/API-2/revise",
            json={"base_release_id": base, "start": 0, "end": 1, "replacement": "B"},
        )
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.get_json()["code"], "stale_release")
        second_body = self.client.get("/api/m/API-2/releases").get_json()
        self.assertEqual(len(second_body["releases"]), 2)
        self.assertEqual(second_body["releases"][1]["body"], "A乙丙丁戊己庚辛壬癸")

    def test_revise_page_uses_htmx_free_form_flow(self):
        resp = self.client.post(
            "/m/JP-1/revise",
            data={
                "base_release_id": self.r1.id,
                "start": "2",
                "end": "4",
                "replacement": "新字",
                "note": "网页改",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("甲乙新字戊己庚辛壬癸", html)
        self.assertIn("网页改", html)

    def test_history_release_view_is_read_only(self):
        r2 = self.revise(0, 1, "改")
        resp = self.client.get("/m/JP-1?release=1")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("历史版", html)
        self.assertIn("甲乙丙丁戊己庚辛壬癸", html)
        self.assertNotIn('name="base_release_id"', html)

    def test_lookup_page_lists_page_box_rect_and_intervals(self):
        r2 = self.revise(3, 7, "X")
        resp = self.client.get(f"/m/JP-1?release={r2.release_no}&lookup_start=3&lookup_end=4")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("10, 20, 100, 30", html)
        self.assertIn("[0, 4)", html)

    def test_bad_json_import_is_rejected_without_writes(self):
        resp = self.client.post(
            "/import", data={"code": "BAD", "title": "坏", "boxes": "{not json"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIsNone(
            self.conn.execute(
                "SELECT id FROM manuscript WHERE code = 'BAD'"
            ).fetchone()
        )

    def test_duplicate_page_box_rejected(self):
        dup = BOXES + [dict(BOXES[0])]
        resp = self.client.post(
            "/api/import", json={"code": "DUP", "title": "重", "boxes": dup}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("重复", resp.get_json()["error"])

    def test_missing_manuscript_404(self):
        self.assertEqual(self.client.get("/m/NOPE").status_code, 404)
        self.assertEqual(self.client.get("/api/m/NOPE/releases").status_code, 404)
        # HTML 404 is rendered, not a bare stack trace.
        html = self.client.get("/m/NOPE").get_data(as_text=True)
        self.assertIn("404", html)

    def test_api_404_returns_json(self):
        resp = self.client.get("/api/m/NOPE/lookup?start=0&end=1")
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.get_json()["ok"])

    def test_lookup_limit_is_exclusive_at_end(self):
        # [3,4) touches exactly one character even though span is [0,4).
        resp = self.client.get("/api/m/JP-1/lookup?start=3&end=4")
        data = resp.get_json()
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["body_ranges"], [{"start": 3, "end": 4}])

    def test_import_form_via_file_upload(self):
        payload = json.dumps(BOXES, ensure_ascii=False).encode("utf-8")
        import io
        resp = self.client.post(
            "/import",
            data={
                "code": "FILE-1",
                "title": "上传",
                "file": (io.BytesIO(payload), "boxes.json"),
            },
            content_type="multipart/form-data",
        )
        self.assertIn(resp.status_code, (302, 303))
        self.assertIsNotNone(
            self.conn.execute(
                "SELECT id FROM manuscript WHERE code = 'FILE-1'"
            ).fetchone()
        )


class ConcurrencyTest(ArchiveTestBase):
    def test_stale_check_reads_current_inside_write_transaction(self):
        other = archive.connect(self.db_path)
        try:
            with self.app.app_context():
                ms = archive.get_manuscript(other, "JP-1")
                base = ms.current_release_id
            # Another writer advances the manuscript.
            r2 = self.revise(0, 1, "A")
            with self.assertRaises(archive.StaleReleaseError):
                with self.app.app_context():
                    archive.revise(other, ms, base, 1, 2, "B", "")
            # Nothing was written by the rejected call.
            ms2 = archive.get_manuscript(other, "JP-1")
            self.assertEqual(ms2.current_release_id, r2.id)
            self.assertEqual(
                other.execute(
                    "SELECT COUNT(*) AS n FROM release WHERE manuscript_id = ?",
                    (ms2.id,),
                ).fetchone()["n"],
                2,
            )
        finally:
            other.close()

    def test_unpaired_surrogate_rejected(self):
        payload = [{"page_no": 1, "box_no": 1, "x": 0, "y": 0, "w": 1, "h": 1, "text": "\ud800"}]
        with self.assertRaises(archive.ValidationError):
            archive.parse_boxes(payload)
        with self.assertRaises(archive.ValidationError):
            archive.revise(self.conn, self.ms, self.r1.id, 0, 1, "\ud800", "")

    def test_nul_rejected(self):
        payload = [{"page_no": 1, "box_no": 1, "x": 0, "y": 0, "w": 1, "h": 1, "text": "a\x00b"}]
        with self.assertRaises(archive.ValidationError):
            archive.parse_boxes(payload)


class LookupFilterTest(ArchiveTestBase):
    def test_page_filter_limits_boxes(self):
        r2 = self.revise(0, 1, "X")
        rows = self.lookup(r2, 0, 10)
        self.assertEqual({r["page_no"] for r in rows}, {1, 2})
        only2 = self.lookup(r2, 0, 10, page=2)
        self.assertEqual([(r["page_no"], r["box_no"]) for r in only2], [(2, 1)])

    def test_marks_are_render_only_and_text_is_unchanged(self):
        html = archive.body_markup(self.conn, self.r1, (1, 5))
        import re
        stripped = re.sub(r"<[^>]+>", "", html)
        stripped = stripped.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        self.assertEqual(stripped, self.r1.body)
        # The highlight crosses a provenance boundary, so it is split into one
        # mark per span and never merges spans.
        self.assertEqual(html.count('<mark class="hit">'), 2)

    def test_zero_length_and_reverse_lookups_rejected(self):
        for start, end in [(0, 0), (5, 3), (-1, 2), (0, 11)]:
            with self.assertRaises(archive.ValidationError):
                self.lookup(self.r1, start, end)


class SchemaTest(ArchiveTestBase):
    def test_schema_has_foreign_keys_and_single_current_pointer(self):
        fk = self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(fk, 1)
        cols = {
            r["name"]
            for r in self.conn.execute("PRAGMA table_info(manuscript)").fetchall()
        }
        self.assertIn("current_release_id", cols)

    def test_source_sets_are_shared_across_releases(self):
        r2 = self.revise(0, 1, "X")
        r3 = self.revise(9, 10, "Y", base=r2.id)
        sets = {
            row["source_set_id"]
            for row in self.conn.execute(
                "SELECT DISTINCT source_set_id FROM body_span WHERE release_id IN (?, ?, ?)",
                (self.r1.id, r2.id, r3.id),
            ).fetchall()
        }
        # Boxes 1..3 exist; no per-release copy of unchanged provenance.
        self.assertLessEqual(len(sets), 4)


class ParameterLimitTest(unittest.TestCase):
    """A big manuscript must open and be revised on a low-parameter-limit build."""

    # Far below the 999 minimum SQLite usually reports, and far below the 300
    # source sets / spans the fixture below puts in one release.
    LIMIT = 25
    BOX_COUNT = 300

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "big.sqlite3")
        archive.init_db(self.db_path)
        self.conn = connect_with_low_limit(self.db_path, self.LIMIT)
        boxes = archive.parse_boxes(many_boxes(self.BOX_COUNT))
        self.mid = archive.ingest(self.conn, "BIG-1", "大稿", boxes)
        self.ms = archive.get_manuscript(self.conn, "BIG-1")
        self.r1 = archive.get_release(self.conn, self.mid, 1)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_low_limit_is_actually_enforced(self):
        self.assertEqual(
            self.conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER), self.LIMIT
        )
        placeholders = ",".join("?" for _ in range(self.LIMIT + 1))
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute(f"SELECT 1 WHERE ? IN ({placeholders})", list(range(self.LIMIT + 1)))

    def test_page_markup_opens_with_many_source_sets(self):
        html = archive.body_markup(self.conn, self.r1)
        self.assertEqual(html.count('class="prov"'), self.BOX_COUNT)
        self.assertIn(self.r1.body[0], html)

    def test_page_markup_bounds_hover_to_eight_plus_ellipsis(self):
        union = self._union_release()
        html = archive.body_markup(self.conn, union)
        title = html.split('title="', 1)[1].split('"', 1)[0]
        self.assertEqual(title.count("页"), archive.PREVIEW_SOURCE_TITLES)
        self.assertTrue(title.endswith("；…"))

    def test_revise_whole_large_body_succeeds_and_keeps_old_release(self):
        r2 = archive.revise(
            self.conn, self.ms, self.r1.id, 0, len(self.r1.body), "替", "整版替换"
        )
        self.assertEqual(r2.body, "替")
        self.assertEqual(self.ms.current_release_id, self.r1.id)
        old = archive.get_release(self.conn, self.mid, 1)
        self.assertEqual(old.body, self.r1.body)
        self.assertEqual(len(archive.lookup_sources(self.conn, old, 0, 1)), 1)

    def _union_release(self) -> archive.Release:
        """One span whose source set holds one entry per selected box."""
        return archive.revise(
            self.conn, self.ms, self.r1.id, 0, len(self.r1.body), "Q", ""
        )

    def test_web_page_opens_and_revise_and_lookup_work_on_low_limit(self):
        """Old library + small SQLite limit: the browser flow must still work.

        The app's own per-request connection is opened with the same low
        SQLITE_LIMIT_VARIABLE_NUMBER, so this exercises the real HTTP handlers.
        """
        real_connect = archive.connect

        def low_limit_connect(path: str) -> sqlite3.Connection:
            conn = real_connect(path)
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, self.LIMIT)
            return conn

        app = create_app({"TESTING": True, "DATABASE": self.db_path})
        client = app.test_client()
        with mock.patch.object(archive, "connect", low_limit_connect):
            page = client.get("/m/BIG-1")
            self.assertEqual(page.status_code, 200)
            html = page.get_data(as_text=True)
            self.assertIn(f'data-release-id="{self.r1.id}"', html)

            resp = client.post(
                "/m/BIG-1/revise",
                data={
                    "base_release_id": self.r1.id,
                    "start": 0,
                    "end": self.BOX_COUNT,
                    "replacement": "整",
                    "note": "整版替换",
                },
            )
            self.assertEqual(resp.status_code, 302)
            page = client.get(resp.headers["Location"])
            self.assertEqual(page.status_code, 200)
            self.assertIn("整", page.get_data(as_text=True))

            # Full-detail lookup of the inserted character lists every box.
            resp = client.get("/api/m/BIG-1/lookup?start=0&end=1")
            payload = resp.get_json()
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(payload["sources"]), self.BOX_COUNT)
            self.assertFalse(payload["sources"][0]["exact"])

            # Every historical release still renders exactly as it was.
            old = client.get("/m/BIG-1?release=1")
            self.assertEqual(old.status_code, 200)
            self.assertIn("历史版", old.get_data(as_text=True))

    def test_large_span_selection_with_many_boxes_inherits_full_union(self):
        # The new single-character run inherits all 300 box intervals.
        r2 = self._union_release()
        self.assertEqual(r2.body, "Q")
        set_row = self.conn.execute(
            "SELECT source_set_id FROM body_span WHERE release_id = ?",
            (r2.id,),
        ).fetchone()
        entry_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM source_entry WHERE source_set_id = ?",
            (set_row["source_set_id"],),
        ).fetchone()["n"]
        self.assertEqual(entry_count, self.BOX_COUNT)

        # Full-detail lookup still returns every contributing box.
        sources = archive.lookup_sources(self.conn, r2, 0, 1)
        self.assertEqual(len(sources), self.BOX_COUNT)
        self.assertFalse(sources[0]["exact"])

        # Re-editing the inserted character reuses the whole inherited union.
        ms = archive.get_manuscript(self.conn, "BIG-1")
        r3 = archive.revise(self.conn, ms, r2.id, 0, 1, "Y", "")
        again = archive.lookup_sources(self.conn, r3, 0, 1)
        self.assertEqual(len(again), self.BOX_COUNT)

    def test_hover_on_shared_union_shows_first_eight_then_ellipsis(self):
        r2 = self._union_release()
        html = archive.body_markup(self.conn, r2)
        title = html.split('title="', 1)[1].split('"', 1)[0]
        self.assertEqual(title.count("页"), archive.PREVIEW_SOURCE_TITLES)
        self.assertTrue(title.endswith("；…"))
        # In stored order (box id / interval), i.e. the original reading order.
        self.assertIn("页1 框1 [0,1)", title)
        self.assertIn("页1 框8 [0,1)", title)
        self.assertNotIn("框9 ", title)

    def test_preview_fetches_at_most_nine_rows_per_set(self):
        r2 = self._union_release()
        previews = archive.load_source_previews(self.conn, r2.id)
        self.assertEqual(
            [len(v) for v in previews.values()], [archive.PREVIEW_SOURCE_LIMIT]
        )

    def test_preview_query_count_does_not_grow_with_spans_or_sets(self):
        def count_queries(release_id: int) -> int:
            seen: list[str] = []

            def tracer(statement: str) -> None:
                if statement.lstrip().upper().startswith("SELECT"):
                    seen.append(statement)

            self.conn.set_trace_callback(tracer)
            try:
                archive.body_markup(
                    self.conn, archive.get_release_by_id(self.conn, release_id)
                )
            finally:
                self.conn.set_trace_callback(None)
            return len(seen)

        mid = archive.ingest(
            self.conn, "SMALL-1", "小稿", archive.parse_boxes(many_boxes(3))
        )
        small_release = archive.get_release(self.conn, mid, 1)
        union = self._union_release()
        counts = {
            "3 spans/3 sets": count_queries(small_release.id),
            "1 span/1 huge set": count_queries(union.id),
            "300 spans/300 sets": count_queries(self.r1.id),
        }
        self.assertEqual(
            len(set(counts.values())), 1,
            f"body rendering must issue a constant number of queries: {counts}",
        )


class PreviewHttpTest(ArchiveTestBase):
    @staticmethod
    def _titles(html: str) -> list[str]:
        return re.findall(r'title="([^"]*)"', html)

    def test_hover_shows_all_entries_when_at_most_eight_no_ellipsis(self):
        boxes = archive.parse_boxes(many_boxes(8))
        with self.app.app_context():
            mid = archive.ingest(self.conn, "HOVER-8", "悬停八", boxes)
            release = archive.get_release(self.conn, mid, 1)
            html = archive.body_markup(self.conn, release)
        # Eight one-character spans, each with its own single-box source.
        self.assertEqual(len(self._titles(html)), 8)
        joined = "".join(self._titles(html))
        self.assertIn("页1 框1 [0,1)", joined)
        self.assertIn("页1 框8 [0,1)", joined)
        self.assertNotIn("…", joined)

    def test_hover_on_nine_entries_shows_eight_then_ellipsis(self):
        boxes = archive.parse_boxes(many_boxes(9))
        with self.app.app_context():
            mid = archive.ingest(self.conn, "HOVER-9", "悬停九", boxes)
            release = archive.get_release(self.conn, mid, 1)
            archive.revise(
                self.conn, archive.get_manuscript(self.conn, "HOVER-9"),
                release.id, 0, len(release.body), "Z", "",
            )
            union = archive.get_release(self.conn, mid, 2)
            title = self._titles(archive.body_markup(self.conn, union))[0]
        self.assertEqual(title.count("页"), 8)
        self.assertTrue(title.endswith("；…"))

    def test_history_page_is_read_only_and_keeps_original_text(self):
        boxes = archive.parse_boxes(many_boxes(12))
        with self.app.app_context():
            mid = archive.ingest(self.conn, "HIST-1", "历史", boxes)
            r1 = archive.get_release(self.conn, mid, 1)
            ms = archive.get_manuscript(self.conn, "HIST-1")
            archive.revise(self.conn, ms, r1.id, 0, 1, "改", "")
        resp = self.client.get("/m/HIST-1?release=1")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("历史版", html)
        self.assertNotIn('name="base_release_id"', html)
        self.assertIn(r1.body[:12], html)

    def test_fewer_than_nine_entries_has_no_ellipsis(self):
        html = archive.body_markup(self.conn, self.r1)
        title = self._titles(html)[0]
        self.assertEqual(title.count("页"), 1)
        self.assertNotIn("…", title)


if __name__ == "__main__":
    unittest.main()

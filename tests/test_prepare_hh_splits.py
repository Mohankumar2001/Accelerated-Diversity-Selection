import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_hh_splits import make_splits, parse_pair, prompt_id


def pair(question, a=" preferred", b=" rejected"):
    p = "\n\nHuman: " + question + "\n\nAssistant:"
    return {"chosen": p + a, "rejected": p + b}


class SplitTests(unittest.TestCase):
    def test_multiturn_reconstruction(self):
        row = pair("First\n\nAssistant: earlier\n\nHuman: Next")
        result, error = parse_pair(row)
        self.assertIsNone(error)
        self.assertIn("earlier", result["prompt"])
        self.assertEqual(result["prompt"] + result["chosen_response"], row["chosen"])
        self.assertEqual(result["prompt"] + result["rejected_response"], row["rejected"])

    def test_format_rejections(self):
        for row, expected in [
            (pair("a", a="  "), "empty_final_response"),
            (pair("a", a=" same", b="same "), "identical_final_responses"),
            ({"chosen": "a", "rejected": "b"}, "missing_assistant_marker"),
            ({"chosen": None, "rejected": "b"}, "missing_or_nonstring_text"),
            ({"chosen": pair("a")["chosen"], "rejected": pair("b")["rejected"]}, "different_last_turn_contexts"),
        ]:
            self.assertEqual(parse_pair(row)[1], expected)

    def test_grouping_dedup_and_original_test_reservation(self):
        train = [pair(str(i)) for i in range(30)]
        train += [pair("0"), pair("0", a=" another"), pair("held"), pair("bad-test")]
        raw = {"train": train, "test": [pair("held"), pair("held"), pair("bad-test", a="")]}
        before = copy.deepcopy(raw)
        splits, excluded, report = make_splits(raw)
        self.assertEqual(raw, before)
        self.assertEqual(report["output_rows"]["test"], 1)
        self.assertEqual(report["excluded_rows_by_reason"]["train"]["exact_duplicate_pair"], 1)
        self.assertEqual(report["excluded_rows_by_reason"]["train"]["prompt_reserved_for_official_test"], 2)
        self.assertEqual(report["excluded_rows_by_reason"]["test"]["exact_duplicate_pair"], 1)
        self.assertEqual(report["excluded_rows_by_reason"]["test"]["empty_final_response"], 1)
        self.assertEqual(sum(map(len, splits.values())) + len(excluded), 37)
        zero_group = parse_pair(pair("0"))[0]["prompt_id"]
        locations = [s for s, rows in splits.items() if any(r["prompt_id"] == zero_group for r in rows)]
        self.assertEqual(len(locations), 1)
        self.assertTrue(all(v == 0 for v in report["prompt_overlap"].values()))
        self.assertEqual(make_splits(raw), (splits, excluded, report))
        self.assertNotEqual(make_splits(raw, seed=43)[2]["assignment_sha256"], report["assignment_sha256"])

    def test_normalized_prompt_grouping(self):
        self.assertEqual(prompt_id("\r\n Cafe\u0301\r\n"), prompt_id("Caf\u00e9"))
        # Internal whitespace is preserved, important for code in prompts.
        self.assertNotEqual(prompt_id("a  b"), prompt_id("a b"))


if __name__ == "__main__":
    unittest.main()

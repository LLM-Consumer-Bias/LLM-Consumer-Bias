"""Tests for response parser."""

import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "04_parse"))

from parse_responses import parse_ab, parse_rep, parse_cmp, parse_response


class TestAB:
    def test_simple_a(self):
        assert parse_ab("A") == "A"

    def test_simple_b(self):
        assert parse_ab("B") == "B"

    def test_lowercase(self):
        assert parse_ab("a") == "A"
        assert parse_ab("b") == "B"

    def test_with_explanation(self):
        assert parse_ab("A. I would choose option A because...") == "A"

    def test_whitespace(self):
        assert parse_ab("  B  ") == "B"

    def test_gibberish(self):
        assert parse_ab("I can't decide") == "invalid"

    def test_empty(self):
        assert parse_ab("") == "invalid"


class TestREP:
    OPTION_A = "Price: $399; Camera: 48 MP; Battery: 4,500 mAh; Warranty: 2 years."
    OPTION_B = "Price: $749; Camera: 108 MP; Battery: 5,500 mAh; Warranty: 2 years."

    def test_exact_match_a(self):
        assert parse_rep(self.OPTION_A, self.OPTION_A, self.OPTION_B) == "A"

    def test_exact_match_b(self):
        assert parse_rep(self.OPTION_B, self.OPTION_A, self.OPTION_B) == "B"

    def test_partial_match(self):
        # Close enough to option A
        partial = "Price: $399; Camera: 48 MP; Battery: 4,500 mAh"
        assert parse_rep(partial, self.OPTION_A, self.OPTION_B) == "A"

    def test_below_threshold(self):
        assert parse_rep("hello world", self.OPTION_A, self.OPTION_B) == "invalid"

    def test_empty(self):
        assert parse_rep("", self.OPTION_A, self.OPTION_B) == "invalid"


class TestCMP:
    def test_yes_aoverb(self):
        assert parse_cmp("yes", "CMP_AoverB") == "A"

    def test_no_aoverb(self):
        assert parse_cmp("no", "CMP_AoverB") == "B"

    def test_yes_bovera(self):
        assert parse_cmp("yes", "CMP_BoverA") == "B"

    def test_no_bovera(self):
        assert parse_cmp("no", "CMP_BoverA") == "A"

    def test_Yes_capitalized(self):
        assert parse_cmp("Yes", "CMP_AoverB") == "A"

    def test_with_explanation(self):
        assert parse_cmp("Yes, I would prefer option A.", "CMP_AoverB") == "A"

    def test_y_shorthand(self):
        assert parse_cmp("y", "CMP_AoverB") == "A"

    def test_n_shorthand(self):
        assert parse_cmp("n", "CMP_BoverA") == "A"

    def test_gibberish(self):
        assert parse_cmp("maybe", "CMP_AoverB") == "invalid"


class TestRouter:
    def test_routes_ab(self):
        assert parse_response("A", "AB_Afirst") == "A"

    def test_routes_cmp(self):
        assert parse_response("yes", "CMP_AoverB") == "A"

    def test_routes_rep(self):
        opt_a = "Price: $399; Camera: 48 MP"
        opt_b = "Price: $749; Camera: 108 MP"
        assert parse_response(opt_a, "REP_Afirst", opt_a, opt_b) == "A"

    def test_empty_response(self):
        assert parse_response("", "AB_Afirst") == "invalid"

    def test_none_response(self):
        assert parse_response(None, "AB_Afirst") == "invalid"

"""Language semantics, executed on a heap large enough that the GC never runs."""

import pytest

from halflife.gc.marksweep import MarkSweep
from halflife.vm.vm import MiteError
from halflife.values import mk_int, as_int, is_int, mk_ref, as_ref, is_ref, NIL, TRUE, FALSE

from helpers import run


def out(src):
    return run(src, MarkSweep(heap_bytes=8 << 20)).output


# --- tagged values ---------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, -1, 7, -7, 2 ** 40, -(2 ** 40)])
def test_int_tagging_roundtrip(n):
    v = mk_int(n)
    assert is_int(v) and not is_ref(v)
    assert as_int(v) == n


def test_int_tags_preserve_order():
    assert mk_int(-5) < mk_int(0) < mk_int(5)


def test_ref_tagging_roundtrip():
    v = mk_ref(1234)
    assert is_ref(v) and not is_int(v)
    assert as_ref(v) == 1234


def test_constants_are_not_refs():
    for c in (NIL, TRUE, FALSE):
        assert not is_ref(c) and not is_int(c)


def test_address_zero_is_reserved():
    with pytest.raises(ValueError):
        mk_ref(0)


# --- language --------------------------------------------------------------
def test_arithmetic_and_precedence():
    assert out("print(1 + 2 * 3);") == ["7"]
    assert out("print((1 + 2) * 3);") == ["9"]


def test_truncating_division_and_remainder():
    # C semantics, not Python's: -7/2 is -3 and -7%2 is -1.
    assert out("print(0 - 7 / 2);") == ["-3"]
    assert out("print((0 - 7) % 2);") == ["-1"]
    assert out("print(7 / 2);") == ["3"]


def test_division_by_zero():
    with pytest.raises(MiteError):
        out("print(1 / 0);")


def test_comparison_and_logic():
    assert out("print(1 < 2 and 2 < 3);") == ["true"]
    assert out("print(1 > 2 or 3 > 2);") == ["true"]
    assert out("print(not nil);") == ["true"]


def test_short_circuit_does_not_evaluate_right_side():
    src = """
    fun boom() { let x = 1 / 0; return x; }
    print(false and boom());
    """
    assert out(src) == ["false"]


def test_strings_compare_by_value():
    assert out('print("ab" == "a" + "b");') == ["true"]


def test_records_and_fields():
    src = """
    let r = record { a: 1, b: 2 };
    r.a = 10;
    print(r.a + r.b);
    """
    assert out(src) == ["12"]


def test_missing_field_is_an_error():
    with pytest.raises(MiteError):
        out("let r = record { a: 1 }; print(r.b);")


def test_lists_index_push_len():
    src = """
    let xs = [1, 2, 3];
    push(xs, 4);
    xs[0] = 9;
    print(len(xs));
    print(xs[0] + xs[3]);
    """
    assert out(src) == ["4", "13"]


def test_list_growth_beyond_initial_capacity():
    src = """
    let xs = [];
    let i = 0;
    while (i < 100) { push(xs, i); i = i + 1; }
    let total = 0;
    let j = 0;
    while (j < len(xs)) { total = total + xs[j]; j = j + 1; }
    print(total);
    """
    assert out(src) == ["4950"]


def test_index_out_of_range():
    with pytest.raises(MiteError):
        out("let xs = [1]; print(xs[5]);")


def test_recursion():
    src = """
    fun fib(n) { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); }
    print(fib(18));
    """
    assert out(src) == ["2584"]


def test_while_break_continue():
    src = """
    let total = 0;
    let i = 0;
    while (true) {
      i = i + 1;
      if (i > 10) { break; }
      if (i % 2 == 0) { continue; }
      total = total + i;
    }
    print(total);
    """
    assert out(src) == ["25"]


def test_globals_are_writable_from_functions():
    src = """
    let counter = 0;
    fun bump() { counter = counter + 1; return counter; }
    bump(); bump();
    print(counter);
    """
    assert out(src) == ["2"]


def test_hash_is_stable_across_processes():
    # FNV-1a, not Python's randomised str hash: the same source must give the
    # same number in every process, or cached profiles would be worthless.
    assert out('print(hash("halflife"));') == out('print(hash("halflife"));')
    assert out('print(hash("halflife"));') == ["178523662"]


def test_assert_native():
    with pytest.raises(MiteError):
        out('assert(1 == 2, "nope");')


def test_instruction_budget_stops_infinite_loops():
    with pytest.raises(MiteError):
        run("while (true) { }", MarkSweep(heap_bytes=1 << 20), max_instructions=10000)

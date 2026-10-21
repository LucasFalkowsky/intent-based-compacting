"""The narrative sections of a block, and the instruction that produces them (T6.2, #27).

Three things are under test here, and the second is the reason for the first:

- **The instruction is a data file.** It is loaded from
  `src/idc/compact/narrative_instruction.toml`, it declares the version this
  build reads, and its bytes have a digest the run manifest (#39) can record —
  so the study can say which instruction produced a given run. The tests pin
  that the text really comes off the disk (a changed file changes the text) and
  that the digest is stable for the same bytes and different for changed ones.
- **Nothing a model returns is trusted.** The client answers with text, and
  every malformed shape gets a test of its own: a missing header, a repeated
  one, an injected deterministic header, a second chain heading, an empty
  answer, a misspelled header, prose before the first header, a code fence that
  would enclose the sections below it. The same shapes are refused **with any
  indentation in front of them**, down to a single space and up past the four
  that would make CommonMark call the line a code block, and so are a Setext
  underline and an HTML comment left open — the consumer of a block is a parser
  keyed on the header literals, so an indented copy of one is a copy. A line
  that opens with exactly two `#` goes with them whatever follows the hashes:
  `##` is the chain-heading marker, and the parser refuses a `##` line that is
  not a heading, so `##NOTE` was the one shape the writer called prose and the
  reader called structure (T8.4, #97 — the agreement itself is asserted in
  `tests/test_compact_parse_agreement.py`). A legitimately *empty* narrative
  section is not malformed and renders `EMPTY_SECTION`.
- **The model is called once per chain, and nothing else it returns can move.**
  The call count is read off the fake's record, and the two deterministic
  sections are compared byte for byte against what `render_block` produces
  without any model at all — under a hostile response as well.

Every chain and resolution comes from the shared corpus through the pipeline,
as in `tests/test_compact_template.py`; no test here calls a model.
"""

import hashlib
import tomllib
from pathlib import Path

import pytest

from fixture_corpus import CORPUS
from idc.chunk import chunk_session
from idc.classify import FakeAdapter, classify, gold_script
from idc.compact import (
    DECISIONS_HEADER,
    EMPTY_SECTION,
    INSTRUCTION_PATH,
    INSTRUCTION_VERSION,
    REJECTED_HEADER,
    SECTION_HEADERS,
    SOLUTION_HEADER,
    VERIFICATION_HEADER,
    CompactError,
    FakeNarrativeClient,
    Instruction,
    InstructionError,
    Narrative,
    NarrativeClient,
    NarrativeContractError,
    NarrativeRequest,
    NarrativeScriptExhaustedError,
    as_narrative,
    chain_heading,
    compact_chain,
    decisions_section,
    instruction,
    instruction_digest,
    load_instruction,
    rejected_section,
    render_block,
    render_narrative_block,
    solution_section,
    verification_section,
)
from idc.resolve import Resolution, resolve
from idc.schema import Chain

ONE_CHAIN = "single_intent"
"""The one-chunk, one-edit, one-run fixture the single-case tests read."""

MANY_CHAINS = "interleaved_intents"
"""A fixture whose transcript holds more than one chain."""

WELL_FORMED = (
    "## Intent: Stamp exported timestamps in UTC\n"
    f"{DECISIONS_HEADER}\n"
    "- The export has to stay timezone-aware.\n"
    "- `booked_at` is converted, never reformatted.\n"
    f"{REJECTED_HEADER}\n"
    "- Storing the offset in a second column.\n"
)
"""A response of the shape the instruction asks for, with both sections filled."""


def resolved(name: str) -> tuple[tuple[Chain, Resolution], ...]:
    """Return a fixture's chains with what each of them resolved to."""
    session = CORPUS.session(name)
    chunks = chunk_session(session)
    return tuple(
        (chain, resolve(session, chunks, chain))
        for chain in classify(session, chunks, FakeAdapter(gold_script(CORPUS.gold(name))))
    )


def only(name: str) -> tuple[Chain, Resolution]:
    """Return a single-chain fixture's chain and resolution."""
    pairs = resolved(name)
    assert len(pairs) == 1, f"{name} is used as a single-chain fixture"
    return pairs[0]


def copied_instruction(tmp_path: Path, document: str, name: str = "instruction") -> Path:
    """Write an instruction data file of its own and return its path."""
    path = tmp_path / f"{name}.toml"
    path.write_text(document, encoding="utf-8")
    return path


def packaged_document() -> dict[str, object]:
    """Return the packaged instruction file parsed, as an independent reading."""
    return tomllib.loads(INSTRUCTION_PATH.read_bytes().decode("utf-8"))


# --- the instruction data file -------------------------------------------------


def test_the_instruction_lives_inside_the_package_as_a_data_file():
    assert INSTRUCTION_PATH.is_file()
    assert INSTRUCTION_PATH.suffix == ".toml"
    assert INSTRUCTION_PATH.parent.name == "compact"
    assert INSTRUCTION_PATH.parent.parent.name == "idc"


def test_the_instruction_text_is_the_one_the_data_file_holds():
    # `load_instruction` strips the blank lines around the value; the text itself
    # is the file's, character for character.
    assert instruction().text == str(packaged_document()["instruction"]).strip()
    assert instruction().version == packaged_document()["version"]


def test_the_instruction_declares_the_version_this_build_reads():
    assert instruction().version == INSTRUCTION_VERSION


def test_the_instruction_text_holds_a_word():
    assert instruction().text.strip()


def test_the_instruction_is_read_from_disk_and_not_embedded_in_the_code(tmp_path):
    other = copied_instruction(
        tmp_path, f'version = {INSTRUCTION_VERSION}\ninstruction = """write it my way"""\n'
    )
    assert load_instruction(other).text == "write it my way"
    assert load_instruction(other).text != instruction().text


def test_the_instruction_names_the_two_narrative_headers():
    text = instruction().text
    assert DECISIONS_HEADER in text
    assert REJECTED_HEADER in text


def test_the_instruction_never_asks_for_the_deterministic_headers():
    text = instruction().text
    assert SOLUTION_HEADER not in text
    assert VERIFICATION_HEADER not in text


def test_an_instruction_of_another_version_is_refused(tmp_path):
    other = copied_instruction(
        tmp_path, f'version = {INSTRUCTION_VERSION + 1}\ninstruction = """x"""\n'
    )
    with pytest.raises(InstructionError, match="version"):
        load_instruction(other)


def test_an_instruction_without_a_version_is_refused(tmp_path):
    other = copied_instruction(tmp_path, 'instruction = """x"""\n')
    with pytest.raises(InstructionError, match="version"):
        load_instruction(other)


def test_an_instruction_whose_text_holds_no_word_is_refused(tmp_path):
    other = copied_instruction(
        tmp_path, f'version = {INSTRUCTION_VERSION}\ninstruction = """   """\n'
    )
    with pytest.raises(InstructionError, match="instruction"):
        load_instruction(other)


def test_an_instruction_without_a_text_is_refused(tmp_path):
    other = copied_instruction(tmp_path, f"version = {INSTRUCTION_VERSION}\n")
    with pytest.raises(InstructionError, match="instruction"):
        load_instruction(other)


def test_an_instruction_file_that_is_not_toml_is_refused(tmp_path):
    other = copied_instruction(tmp_path, "version = \ninstruction =\n")
    with pytest.raises(InstructionError, match="TOML"):
        load_instruction(other)


def test_a_missing_instruction_file_is_refused(tmp_path):
    with pytest.raises(InstructionError, match="cannot read"):
        load_instruction(tmp_path / "absent.toml")


def test_the_instruction_error_is_a_failure_of_the_stage(tmp_path):
    assert issubclass(InstructionError, CompactError)
    with pytest.raises(CompactError):
        load_instruction(tmp_path / "absent.toml")


# --- the instruction digest ----------------------------------------------------


def test_the_digest_is_the_sha256_of_the_files_bytes():
    assert instruction_digest() == hashlib.sha256(INSTRUCTION_PATH.read_bytes()).hexdigest()


def test_the_digest_is_stable_across_reads():
    assert instruction_digest() == instruction_digest()
    assert instruction().digest == instruction_digest()


def test_the_digest_changes_when_the_bytes_change(tmp_path):
    document = INSTRUCTION_PATH.read_bytes().decode("utf-8")
    same = copied_instruction(tmp_path, document, name="same")
    changed = copied_instruction(tmp_path, document + "\n# one more comment\n", name="changed")
    assert instruction_digest(same) == instruction_digest()
    assert instruction_digest(changed) != instruction_digest()


def test_the_digest_is_a_hex_digest():
    digest = instruction_digest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


# --- reading a response --------------------------------------------------------


def test_a_well_formed_response_is_read_into_its_three_parts():
    narrative = as_narrative(WELL_FORMED)
    assert narrative.label == "Stamp exported timestamps in UTC"
    assert narrative.decisions == (
        "- The export has to stay timezone-aware.",
        "- `booked_at` is converted, never reformatted.",
    )
    assert narrative.rejected == ("- Storing the offset in a second column.",)


def test_a_response_whose_sections_are_empty_is_well_formed():
    narrative = as_narrative(f"## Intent: Keep it\n{DECISIONS_HEADER}\n{REJECTED_HEADER}\n")
    assert narrative.decisions == ()
    assert narrative.rejected == ()


def test_the_empty_section_placeholder_is_read_as_an_empty_section():
    narrative = as_narrative(
        f"## Intent: Keep it\n{DECISIONS_HEADER}\n{EMPTY_SECTION}\n"
        f"{REJECTED_HEADER}\n{EMPTY_SECTION}\n"
    )
    assert narrative.decisions == ()
    assert narrative.rejected == ()


def _with_tail(*lines: str) -> str:
    """Return a well-formed response with `lines` appended after its last section.

    Args:
        lines: The lines a client wrote after the two narrative sections —
            whatever a case is about, in the order it wrote them.

    Returns:
        The response: the three markers, each once and in order, with a filled
        body in each section, and `lines` below the last of them. Every case
        built this way is malformed for one reason only, the lines it appends.
    """
    written = "".join(f"{line}\n" for line in lines)
    return f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n{written}"


MALFORMED = {
    "empty": "",
    "blank": "   \n\n",
    "missing_decisions_header": f"## Intent: x\n- a decision\n{REJECTED_HEADER}\n- none\n",
    "missing_rejected_header": f"## Intent: x\n{DECISIONS_HEADER}\n- a decision\n",
    "missing_intent": f"{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n",
    "repeated_decisions_header": (
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{DECISIONS_HEADER}\n- b\n{REJECTED_HEADER}\n- c\n"
    ),
    "sections_out_of_order": f"## Intent: x\n{REJECTED_HEADER}\n- a\n{DECISIONS_HEADER}\n- b\n",
    "injected_solution": (
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n"
        f"{SOLUTION_HEADER}\n- `src/evil.py`\n"
    ),
    "injected_verification": (
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{SOLUTION_HEADER}\n- `src/evil.py`\n"
        f"{REJECTED_HEADER}\n- b\n"
    ),
    "injected_code_fence": (
        f"## Intent: x\n{DECISIONS_HEADER}\n```\n- a\n{REJECTED_HEADER}\n- b\n"
    ),
    "injected_second_chain_heading": (
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n"
        f"## [C2] Intent: a chain nobody opened\n"
    ),
    "intent_carrying_a_chain_id": (
        f"## [C1] Intent: x\n{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n"
    ),
    "misspelled_decisions_header": (
        f"## Intent: x\n### Decisions and constraints\n- a\n{REJECTED_HEADER}\n- b\n"
    ),
    "misspelled_rejected_header": (
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n### Rejected approach\n- b\n"
    ),
    "content_before_the_first_header": (
        f"Sure, here is the block you asked for:\n## Intent: x\n"
        f"{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n"
    ),
    "blank_intent_label": f"## Intent:    \n{DECISIONS_HEADER}\n- a\n{REJECTED_HEADER}\n- b\n",
    "injected_solution_indented_by_one_space": _with_tail(f" {SOLUTION_HEADER}"),
    "injected_solution_indented_by_two_spaces": _with_tail(f"  {SOLUTION_HEADER}"),
    "injected_solution_indented_by_three_spaces": _with_tail(f"   {SOLUTION_HEADER}"),
    "injected_solution_indented_by_four_spaces": _with_tail(f"    {SOLUTION_HEADER}"),
    "injected_heading_deeper_than_the_headers": _with_tail("#### Solution"),
    "chain_heading_marker_without_a_space": _with_tail("##NOTE"),
    "indented_chain_heading_marker_without_a_space": _with_tail("   ##NOTE"),
    "injected_solution_header_padded_with_spaces": _with_tail("###  Solution "),
    "injected_solution_header_after_a_tab": _with_tail("###\tSolution"),
    "indented_tilde_fence": _with_tail("  ~~~"),
    "indented_backtick_fence": _with_tail("   ```"),
    "unclosed_tilde_fence": _with_tail("~~~"),
    "fence_with_an_info_string": _with_tail("````python"),
    "trailing_fence": _with_tail("```"),
    "trailing_indented_fence": _with_tail("  ```"),
    "setext_underline_with_dashes": _with_tail("Solution", "---"),
    "setext_underline_with_equals": _with_tail("Solution", "==="),
    "unclosed_html_comment": _with_tail("<!-- everything below is a comment"),
}
"""Every shape a response may not have, one entry per acceptance-criterion case."""


@pytest.mark.parametrize("shape", sorted(MALFORMED))
def test_a_malformed_response_is_refused(shape):
    with pytest.raises(NarrativeContractError):
        as_narrative(MALFORMED[shape])


def test_the_malformed_table_covers_every_enumerated_shape():
    assert set(MALFORMED) == {
        "blank",
        "blank_intent_label",
        "chain_heading_marker_without_a_space",
        "content_before_the_first_header",
        "empty",
        "fence_with_an_info_string",
        "indented_backtick_fence",
        "indented_chain_heading_marker_without_a_space",
        "indented_tilde_fence",
        "injected_code_fence",
        "injected_heading_deeper_than_the_headers",
        "injected_second_chain_heading",
        "injected_solution",
        "injected_solution_header_after_a_tab",
        "injected_solution_header_padded_with_spaces",
        "injected_solution_indented_by_four_spaces",
        "injected_solution_indented_by_one_space",
        "injected_solution_indented_by_three_spaces",
        "injected_solution_indented_by_two_spaces",
        "injected_verification",
        "intent_carrying_a_chain_id",
        "misspelled_decisions_header",
        "misspelled_rejected_header",
        "missing_decisions_header",
        "missing_intent",
        "missing_rejected_header",
        "repeated_decisions_header",
        "sections_out_of_order",
        "setext_underline_with_dashes",
        "setext_underline_with_equals",
        "trailing_fence",
        "trailing_indented_fence",
        "unclosed_html_comment",
        "unclosed_tilde_fence",
    }


def test_a_response_that_is_not_text_is_refused():
    with pytest.raises(NarrativeContractError):
        as_narrative(["## Intent: x"])


def test_the_contract_error_is_a_failure_of_the_stage_and_a_bad_value():
    assert issubclass(NarrativeContractError, CompactError)
    assert issubclass(NarrativeContractError, ValueError)


def test_a_narrative_built_with_a_blank_label_is_the_callers_own_mistake():
    with pytest.raises(ValueError, match="label") as caught:
        Narrative(label="  ", decisions=(), rejected=())
    assert not isinstance(caught.value, NarrativeContractError)


def test_a_narrative_built_with_a_header_line_in_its_body_is_refused():
    with pytest.raises(ValueError, match="header"):
        Narrative(label="x", decisions=(SOLUTION_HEADER,), rejected=())


UNWRITABLE_BODY_LINES = {
    "a_header": SOLUTION_HEADER,
    "a_header_indented_by_one_space": f" {SOLUTION_HEADER}",
    "a_header_indented_by_four_spaces": f"    {SOLUTION_HEADER}",
    "a_chain_heading_marker_without_a_space": "##NOTE",
    "an_indented_chain_heading_marker_without_a_space": "   ##NOTE",
    "a_backtick_fence": "```",
    "an_indented_backtick_fence": "   ```",
    "an_indented_tilde_fence": "  ~~~",
    "a_setext_underline_with_dashes": "---",
    "a_setext_underline_with_equals": "===",
    "an_unclosed_html_comment": "<!-- everything below is a comment",
    "a_setext_underline_with_trailing_spaces": "---   ",
    "a_setext_underline_with_a_trailing_tab": "===\t",
    "an_indented_setext_underline_with_trailing_spaces": "   ---  ",
}
"""Body lines a caller may not put into a `Narrative`, one per refused shape.

The last three are not right-stripped on purpose. `as_narrative` rstrips every
line before it judges it, a caller constructing a `Narrative` does not, and a
guard that only held for the first of those two would be no guard at all —
CommonMark allows trailing whitespace on a Setext underline and on a thematic
break, so `---   ` under a line of text is the same header as `---`.
"""


@pytest.mark.parametrize("shape", sorted(UNWRITABLE_BODY_LINES))
def test_a_narrative_built_with_a_structural_body_line_is_refused(shape):
    line = UNWRITABLE_BODY_LINES[shape]
    with pytest.raises(ValueError, match="body line") as caught:
        Narrative(label="x", decisions=(line,), rejected=())
    assert not isinstance(caught.value, NarrativeContractError)
    with pytest.raises(ValueError, match="body line"):
        Narrative(label="x", decisions=(), rejected=(line,))


def test_the_unwritable_body_lines_cover_every_enumerated_shape():
    assert set(UNWRITABLE_BODY_LINES) == {
        "a_backtick_fence",
        "a_chain_heading_marker_without_a_space",
        "a_header",
        "a_header_indented_by_four_spaces",
        "a_header_indented_by_one_space",
        "a_setext_underline_with_dashes",
        "a_setext_underline_with_equals",
        "a_setext_underline_with_a_trailing_tab",
        "a_setext_underline_with_trailing_spaces",
        "an_indented_backtick_fence",
        "an_indented_chain_heading_marker_without_a_space",
        "an_indented_setext_underline_with_trailing_spaces",
        "an_indented_tilde_fence",
        "an_unclosed_html_comment",
    }


def test_a_body_line_is_judged_whether_or_not_its_caller_stripped_it():
    """Both guards judge the same line, however its caller spaced it.

    `as_narrative` rstrips every line of a response before judging it and a
    caller constructing a `Narrative` does not, so a guard that relied on that
    normalization held for one of its two entry points only. The shape it let
    through was the Setext underline, the one shape whose meaning sits at the
    right end of the line.
    """
    for line in ("---   ", "===\t", "   ---  "):
        with pytest.raises(NarrativeContractError, match="body line"):
            as_narrative(f"## Intent: x\n{DECISIONS_HEADER}\n- a\n{line}\n{REJECTED_HEADER}\n")
        with pytest.raises(ValueError, match="body line"):
            Narrative(label="x", decisions=("- a", line), rejected=())


def test_an_indented_bullet_is_not_structure():
    narrative = as_narrative(
        f"## Intent: x\n{DECISIONS_HEADER}\n- a\n  - under a\n{REJECTED_HEADER}\n- b\n"
    )
    assert narrative.decisions == ("- a", "  - under a")
    assert Narrative(label="x", decisions=("  - under a",)).decisions == ("  - under a",)


# --- rendering the narrative sections ------------------------------------------


def test_the_two_narrative_sections_carry_the_template_headers():
    narrative = as_narrative(WELL_FORMED)
    assert decisions_section(narrative).startswith(f"{DECISIONS_HEADER}\n")
    assert rejected_section(narrative).startswith(f"{REJECTED_HEADER}\n")
    assert decisions_section(narrative).endswith("\n")
    assert rejected_section(narrative).endswith("\n")


def test_a_filled_section_renders_the_lines_the_model_wrote():
    narrative = as_narrative(WELL_FORMED)
    assert decisions_section(narrative) == (
        f"{DECISIONS_HEADER}\n"
        "- The export has to stay timezone-aware.\n"
        "- `booked_at` is converted, never reformatted.\n"
    )


def test_an_empty_narrative_section_renders_the_placeholder():
    narrative = as_narrative(f"## Intent: Keep it\n{DECISIONS_HEADER}\n{REJECTED_HEADER}\n")
    assert decisions_section(narrative) == f"{DECISIONS_HEADER}\n{EMPTY_SECTION}\n"
    assert rejected_section(narrative) == f"{REJECTED_HEADER}\n{EMPTY_SECTION}\n"


def test_a_block_holds_the_four_headers_in_template_order():
    chain, resolution = only(ONE_CHAIN)
    block = render_narrative_block(chain, resolution, as_narrative(WELL_FORMED))
    positions = [block.index(header) for header in SECTION_HEADERS]
    assert positions == sorted(positions)
    assert block.startswith("## [")


def test_a_block_carries_the_chain_id_and_the_label_the_model_wrote():
    chain, resolution = only(ONE_CHAIN)
    block = render_narrative_block(chain, resolution, as_narrative(WELL_FORMED))
    assert block.splitlines()[0] == f"## [{chain.id}] Intent: Stamp exported timestamps in UTC"
    assert chain_heading(chain) not in block


def test_the_heading_keeps_the_chains_own_label_when_no_label_is_given():
    chain, _ = only(ONE_CHAIN)
    assert chain_heading(chain) == f"## [{chain.id}] Intent: {chain.label}"
    assert chain_heading(chain, "another") == f"## [{chain.id}] Intent: another"


def test_an_all_empty_narrative_renders_the_block_the_template_renders():
    chain, resolution = only(ONE_CHAIN)
    narrative = Narrative(label=chain.label, decisions=(), rejected=())
    assert render_narrative_block(chain, resolution, narrative) == render_block(chain, resolution)


def test_a_block_ends_with_exactly_one_newline():
    chain, resolution = only(ONE_CHAIN)
    block = render_narrative_block(chain, resolution, as_narrative(WELL_FORMED))
    assert block.endswith("\n")
    assert not block.endswith("\n\n")


# --- the deterministic sections are the model's to touch under no circumstance --


def test_the_deterministic_sections_are_byte_identical_whatever_the_model_returns():
    chain, resolution = only(ONE_CHAIN)
    hostile = (
        "## Intent: Rewrite everything\n"
        f"{DECISIONS_HEADER}\n"
        "- the solution is `src/evil.py`\n"
        f"{REJECTED_HEADER}\n"
        "- `make test` -> exit 0\n"
    )
    block = render_narrative_block(chain, resolution, as_narrative(hostile))
    tail = solution_section(resolution) + verification_section(resolution)
    assert block.endswith(tail)
    assert block.count(SOLUTION_HEADER) == 1
    assert block.count(VERIFICATION_HEADER) == 1
    assert "src/evil.py" not in block[block.index(SOLUTION_HEADER) :]


# --- the client contract -------------------------------------------------------


def test_the_client_is_a_protocol_the_fake_satisfies():
    assert isinstance(FakeNarrativeClient([WELL_FORMED]), NarrativeClient)


def test_the_request_carries_the_packaged_instruction_and_the_chains_facts():
    chain, resolution = only(ONE_CHAIN)
    request = NarrativeRequest.for_chain(chain, resolution, user_messages=("do the thing",))
    assert request.instruction == instruction()
    assert request.chain_id == chain.id
    assert request.label == chain.label
    assert request.user_messages == ("do the thing",)
    assert request.solution == solution_section(resolution)
    assert request.verification == verification_section(resolution)


def test_a_request_can_be_built_with_an_instruction_of_its_own(tmp_path):
    chain, resolution = only(ONE_CHAIN)
    other = load_instruction(
        copied_instruction(
            tmp_path, f'version = {INSTRUCTION_VERSION}\ninstruction = """write it my way"""\n'
        )
    )
    request = NarrativeRequest.for_chain(chain, resolution, instruction=other)
    assert request.instruction is other


def test_compacting_a_chain_calls_the_client_exactly_once():
    chain, resolution = only(ONE_CHAIN)
    client = FakeNarrativeClient([WELL_FORMED])
    compact_chain(chain, resolution, client)
    assert len(client.calls) == 1
    assert client.remaining == 0
    assert client.calls[0].request.chain_id == chain.id
    assert client.calls[0].response == WELL_FORMED


def test_every_chain_of_a_transcript_costs_exactly_one_call():
    pairs = resolved(MANY_CHAINS)
    assert len(pairs) > 1, f"{MANY_CHAINS} is used as a multi-chain fixture"
    client = FakeNarrativeClient([WELL_FORMED] * len(pairs))
    for chain, resolution in pairs:
        compact_chain(chain, resolution, client)
    assert len(client.calls) == len(pairs)
    assert [call.request.chain_id for call in client.calls] == [chain.id for chain, _ in pairs]


def test_compacting_a_chain_renders_what_the_client_returned():
    chain, resolution = only(ONE_CHAIN)
    client = FakeNarrativeClient([WELL_FORMED])
    assert compact_chain(chain, resolution, client) == render_narrative_block(
        chain, resolution, as_narrative(WELL_FORMED)
    )


def test_a_malformed_response_fails_the_chain_it_was_returned_for():
    chain, resolution = only(ONE_CHAIN)
    client = FakeNarrativeClient([MALFORMED["injected_solution"]])
    with pytest.raises(NarrativeContractError):
        compact_chain(chain, resolution, client)
    assert len(client.calls) == 1


def test_the_fake_raises_when_its_script_runs_out():
    chain, resolution = only(ONE_CHAIN)
    client = FakeNarrativeClient([])
    with pytest.raises(NarrativeScriptExhaustedError, match=chain.id):
        compact_chain(chain, resolution, client)
    assert issubclass(NarrativeScriptExhaustedError, CompactError)


def test_the_fake_plays_its_script_in_order():
    chain, resolution = only(ONE_CHAIN)
    second = WELL_FORMED.replace("Stamp exported timestamps in UTC", "Something else")
    client = FakeNarrativeClient([WELL_FORMED, second])
    assert "Stamp exported timestamps in UTC" in compact_chain(chain, resolution, client)
    assert "Something else" in compact_chain(chain, resolution, client)
    assert client.remaining == 0


# --- determinism ---------------------------------------------------------------


def test_the_same_chain_and_the_same_response_render_the_same_bytes():
    chain, resolution = only(ONE_CHAIN)
    first = compact_chain(chain, resolution, FakeNarrativeClient([WELL_FORMED]))
    second = compact_chain(chain, resolution, FakeNarrativeClient([WELL_FORMED]))
    assert first == second


def test_reading_the_same_response_twice_gives_the_same_narrative():
    assert as_narrative(WELL_FORMED) == as_narrative(WELL_FORMED)


# --- the stage's surface -------------------------------------------------------


def test_the_instruction_value_object_carries_its_source_and_its_digest():
    loaded = instruction()
    assert isinstance(loaded, Instruction)
    assert loaded.path == INSTRUCTION_PATH
    assert loaded.digest == instruction_digest()

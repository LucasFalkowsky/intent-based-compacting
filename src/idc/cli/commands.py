"""What each command does: one function per pipeline stage, plus the whole run.

Every function here takes a validated `Config` and a `Runtime` and returns an
exit status. None of them parses an argument — `idc.cli.main` does that and has
already refused an invocation that left out a path — and none of them calls an
agent, runs git or reaches the network: a command reads transcript files and
writes transcript files.

The commands pass the stage's own entry point its arguments and nothing more.
Three decisions are visible in how they are wired together, and they are
decisions rather than consequences:

- **`chunk` reduces by default.** `README.md` describes reduction as part of the
  chunk stage ("split the transcript at user-message boundaries, reduce the
  harness noise inside a chunk"), so `idc.chunk.reduce_session` is what the
  command runs and `--no-reduce` falls back to `idc.chunk.chunk_session`. A
  reduced session is a *different* session — the chunks' ranges index into it —
  which is why the artifact carries both halves together.
- **`resolve` prints and writes no file.** Resolution is pure computation from
  the transcript, so `compact` derives it again rather than reading it back; the
  command exists to make what the resolver decided readable, and there is no
  canonical document for a `Resolution` to be written as.
- **`compact` names the target agent, `render` does not.** `CompactedBlock`
  carries `rendered_for`, and `idc.render.write_transcript` takes the target
  from the block, so `--target` belongs to the command that builds the block.

`run` is the end-to-end command the study uses. It is the only one that writes a
run manifest, and it writes one on every successful pass: that was T9's own
acceptance criterion, left to this ticket because `idc.measure.build_manifest`
had no production caller until now.
"""

import sys
from dataclasses import replace
from pathlib import Path

from idc.chunk import chunk_session, reduce_session
from idc.classify import classify_chunks
from idc.cli.artifacts import Artifact, dump_artifact, load_artifact
from idc.cli.config import Config
from idc.cli.exit_codes import EXIT_OK
from idc.cli.runtime import Runtime
from idc.compact import incremental_blocks
from idc.importers import INTENT_SUMMARY_MARKER, ImportNote, import_transcript
from idc.measure import RunTimer, build_manifest, dump_manifest, load_manifest, measure
from idc.parse import parse_chains
from idc.render import write_transcript
from idc.resolve import Resolution, resolve
from idc.schema import Agent, Chain, Chunk, CompactedBlock, Session, SessionSource, dump

__all__ = [
    "MANIFEST_SUFFIX",
    "candidate_chains",
    "chunk_command",
    "classify_command",
    "compact_command",
    "compacted_block",
    "import_command",
    "manifest_path",
    "measure_command",
    "render_command",
    "resolve_command",
    "resolve_report",
    "run_command",
    "summary_block_text",
]

MANIFEST_SUFFIX = ".manifest.json"
"""What `run` appends to its output path when no manifest path was given.

A derived default and not a required flag, because "a manifest is written for
every compaction run" is the criterion; an invocation that forgot `--manifest`
would otherwise produce a run with no provenance.
"""


def _report(notes: tuple[ImportNote, ...]) -> None:
    """Print an import's notes on stderr, one per line, or nothing when there are none.

    Args:
        notes: The notes the importer produced, already grouped and sorted by
            `idc.importers`.
    """
    for note in notes:
        print(f"idc: {note.kind} {note.detail} ({note.count}x)", file=sys.stderr)


def _chunked(session: Session, *, reduce: bool) -> tuple[Session, tuple[Chunk, ...]]:
    """Chunk a session, reducing the harness noise inside each chunk when asked.

    Args:
        session: The canonical session to chunk.
        reduce: Whether to run `idc.chunk.reduce_session` rather than
            `idc.chunk.chunk_session`.

    Returns:
        The session the chunks range over and the chunks themselves. With
        reduction on, the session is the reduced one — the chunks' ranges are
        indices into it — and the reduction notes are printed on stderr.
    """
    if not reduce:
        return session, tuple(chunk_session(session))
    reduced = reduce_session(session)
    for note in reduced.notes:
        print(f"idc: reduced {note.reason.value} in {note.message_id}", file=sys.stderr)
    return reduced.session, tuple(chunked.chunk for chunked in reduced.chunks)


def summary_block_text(session: Session) -> str | None:
    """Return the Markdown block sequence an earlier run of this tool left behind.

    Args:
        session: An imported session, compacted or not.

    Returns:
        The summary message's text with a leading `INTENT_SUMMARY_MARKER` and
        the blank lines after it removed — which is what `idc.parse` reads —
        or `None` when the session carries no summary of this tool's. A summary
        the *agent* wrote (`native-compaction`) returns `None` too: it is prose
        in the agent's own shape and holds no chain to recover.

    The marker is stripped from the front only, because that is where every
    writer in `idc.render` puts it; a marker anywhere else is left in place and
    `idc.parse` will refuse the block, which is the safe direction — a block
    whose text was quietly edited is not one this tool wrote.
    """
    if session.source is not SessionSource.INTENT_COMPACTION:
        return None
    if session.native_summary_id is None:
        return None
    for message in session.messages:
        if message.id == session.native_summary_id:
            lines = message.content.splitlines()
            while lines and lines[0].strip() in ("", INTENT_SUMMARY_MARKER):
                lines.pop(0)
            return "\n".join(lines)
    return None


def candidate_chains(session: Session) -> tuple[Chain, ...]:
    """Return the chains of an earlier run, to be offered to the classifier.

    This is the seam that makes a second run on this tool's own output possible:
    the chains the previous run wrote are parsed back out of the summary block
    and offered as candidates, so a chunk that continues an earlier intent is
    assigned to that chain instead of opening a new one.

    Args:
        session: The imported session this run is about to classify.

    Returns:
        The parsed chains, in the order their blocks had, and the empty tuple
        for a transcript nothing compacted or one the agent compacted itself.

    Raises:
        ParseError: If the session carries a summary of this tool's that cannot
            be read back as blocks. Refused rather than ignored: silently
            dropping it would re-open every chain of the previous run under a
            new id, and the summary would grow instead of being extended.
    """
    text = summary_block_text(session)
    if text is None:
        return ()
    return tuple(parse_chains(text))


def _user_messages(session: Session, chunks: tuple[Chunk, ...], chain: Chain) -> tuple[str, ...]:
    """Return the user turns of a chain's chunks, for the narrative request.

    Args:
        session: The transcript the chunks range over.
        chunks: The session's chunks.
        chain: The chain whose turns to collect.

    Returns:
        One entry per chunk of the chain that holds a real user turn, in the
        order the chain lists them. A chunk with no user turn — a subagent's
        chunk — contributes nothing.
    """
    # Deferred import: `idc.classify` defines what a chunk's user turn is, and
    # this module must not disagree with the chunker or the classifier about it.
    from idc.classify import chunk_user_message

    by_id = {chunk.id: chunk for chunk in chunks}
    messages = []
    for chunk_id in chain.chunk_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        text = chunk_user_message(session, chunk)
        if text is not None:
            messages.append(text)
    return tuple(messages)


def compacted_block(
    session: Session,
    chunks: tuple[Chunk, ...],
    chains: tuple[Chain, ...],
    runtime: Runtime,
    target: Agent,
) -> CompactedBlock:
    """Compact every new or extended chain and assemble the run's block.

    Args:
        session: The transcript the chunks range over.
        chunks: The session's chunks, in transcript order.
        chains: This run's chains, in the order they are to be written.
        runtime: Where the narrative backend comes from.
        target: The agent the block is rendered for.

    Returns:
        The block: one chain per entry of `chains`, each carrying the Markdown
        `idc.compact` produced for it, with the last chunk named as the one
        that is appended verbatim instead of compacted.

    Raises:
        BackendUnavailableError: If no narrative backend was injected.
        CompactError: If the backend's answer to a chain is not a narrative.
        ResolveError: If a chain cannot be resolved against this transcript.
    """
    client = runtime.require_narrative_client()
    blocks = incremental_blocks(
        chains,
        resolution_of=lambda chain: resolve(session, chunks, chain),
        client=client,
        user_messages_of=lambda chain: _user_messages(session, chunks, chain),
    )
    written = [
        Chain(
            id=chain.id,
            label=chain.label,
            chunk_ids=list(chain.chunk_ids),
            compacted=True,
            compacted_text=text,
        )
        for chain, text in zip(chains, blocks, strict=True)
    ]
    return CompactedBlock(chains=written, latest_chunk_id=chunks[-1].id, rendered_for=target)


def resolve_report(session: Session, chunks: tuple[Chunk, ...], chains: tuple[Chain, ...]) -> str:
    """Return what the resolver decided for every chain, as readable lines.

    Args:
        session: The transcript the chunks range over.
        chunks: The session's chunks.
        chains: The chains to resolve.

    Returns:
        One paragraph per chain: its id and label, the file it ended up
        writing with how many edits followed, the verification run that stands
        for it, and how many of its calls are path rather than solution.
    """
    lines = []
    for chain in chains:
        resolution = resolve(session, chunks, chain)
        lines.append(f"{chain.id} {chain.label!r}")
        lines.extend(_solution_lines(resolution))
        lines.append(f"  path: {len(resolution.path_call_ids)} call(s) not in the solution")
    return "\n".join(lines)


def _solution_lines(resolution: Resolution) -> list[str]:
    """Return the solution and verification lines of one resolved chain.

    Args:
        resolution: What the resolver decided.

    Returns:
        One line per solution file and one for the verification, or a line
        saying there is none.
    """
    lines = [
        f"  solution: {entry.path} ({len(entry.edits)} edit(s) after the write)"
        if entry.write is not None
        else f"  solution: {entry.path} ({len(entry.edits)} edit(s), no full write)"
        for entry in resolution.solution_files
    ]
    if not lines:
        lines.append("  solution: no file")
    verification = resolution.verification
    if verification is None:
        lines.append("  verification: none")
    else:
        lines.append(f"  verification: {verification.run.command!r} -> {verification.status.value}")
    return lines


def manifest_path(config: Config, output: Path) -> Path:
    """Return where a run writes its manifest.

    Args:
        config: The run's configuration.
        output: The transcript the run wrote.

    Returns:
        `paths.manifest` when the configuration names one, and
        `<output>.manifest.json` otherwise.
    """
    if config.paths.manifest is not None:
        return Path(config.paths.manifest)
    return Path(str(output) + MANIFEST_SUFFIX)


def import_command(config: Config, runtime: Runtime) -> int:
    """Read an agent transcript into a canonical session document.

    Args:
        config: Needs `agent`, `paths.transcript` and `paths.output`.
        runtime: Unused; the importers compute everything from the file.

    Returns:
        `EXIT_OK`.
    """
    del runtime
    imported = import_transcript(config.agent, config.paths.transcript)
    _report(imported.notes)
    dump(imported.session, config.paths.output)
    print(
        f"{imported.session.id}: {len(imported.session.messages)} message(s) "
        f"-> {config.paths.output}"
    )
    return EXIT_OK


def chunk_command(config: Config, runtime: Runtime) -> int:
    """Split a canonical transcript into chunks at the user-message boundaries.

    Args:
        config: Needs `paths.transcript` and `paths.output`; honours `reduce`.
        runtime: Unused; chunking is pure computation.

    Returns:
        `EXIT_OK`.
    """
    del runtime
    artifact = load_artifact(config.paths.transcript)
    session, chunks = _chunked(artifact.session, reduce=config.reduce)
    dump_artifact(Artifact(session=session, chunks=chunks), config.paths.output)
    print(f"{len(chunks)} chunk(s) -> {config.paths.output}")
    return EXIT_OK


def classify_command(config: Config, runtime: Runtime) -> int:
    """Assign every chunk to an intent chain.

    Args:
        config: Needs `paths.transcript` and `paths.output`; honours `adapter`.
        runtime: Supplies the classifier adapter.

    Returns:
        `EXIT_OK`.
    """
    artifact = load_artifact(config.paths.transcript)
    chunks = artifact.require_chunks()
    adapter = runtime.require_adapter(config.adapter)
    classification = classify_chunks(
        artifact.session, chunks, adapter, candidates=candidate_chains(artifact.session)
    )
    dump_artifact(replace(artifact, chains=classification.chains), config.paths.output)
    print(f"{len(classification.chains)} chain(s) -> {config.paths.output}")
    return EXIT_OK


def resolve_command(config: Config, runtime: Runtime) -> int:
    """Report the accepted solution of every chain.

    Args:
        config: Needs `paths.transcript`.
        runtime: Unused; resolution is pure computation.

    Returns:
        `EXIT_OK`.
    """
    del runtime
    artifact = load_artifact(config.paths.transcript)
    print(resolve_report(artifact.session, artifact.require_chunks(), artifact.require_chains()))
    return EXIT_OK


def compact_command(config: Config, runtime: Runtime) -> int:
    """Render every new or extended chain into a Markdown summary block.

    Args:
        config: Needs `paths.transcript`, `paths.output` and a target agent.
        runtime: Supplies the narrative backend.

    Returns:
        `EXIT_OK`.
    """
    artifact = load_artifact(config.paths.transcript)
    block = compacted_block(
        artifact.session,
        artifact.require_chunks(),
        artifact.require_chains(),
        runtime,
        config.target_agent,
    )
    dump_artifact(replace(artifact, block=block), config.paths.output)
    print(f"{len(block.chains)} block(s) -> {config.paths.output}")
    return EXIT_OK


def render_command(config: Config, runtime: Runtime) -> int:
    """Write the summary and the latest chunk in the target agent's native format.

    The target is the block's own `rendered_for`, which is why this command
    takes no agent of its own.

    Args:
        config: Needs `paths.transcript` and `paths.output`.
        runtime: Unused; rendering is pure computation.

    Returns:
        `EXIT_OK`.
    """
    del runtime
    artifact = load_artifact(config.paths.transcript)
    block = artifact.require_block()
    written = write_transcript(
        block,
        session=artifact.session,
        chunks=artifact.require_chunks(),
        path=config.paths.output,
    )
    print(f"{block.rendered_for.value} transcript -> {written}")
    return EXIT_OK


def measure_command(config: Config, runtime: Runtime) -> int:
    """Count tokens before and after compaction, and report the run's tool statistics.

    Args:
        config: Needs `agent`, `paths.transcript` and `paths.compacted`. Both
            files are read as the agent's native transcripts, so this command
            measures what the agent would actually load.
        runtime: Supplies the tokenizer registry.

    Returns:
        `EXIT_OK`.
    """
    before = import_transcript(config.agent, config.paths.transcript)
    after = import_transcript(config.agent, config.paths.compacted)
    tokenizer = runtime.tokenizer_registry().tokenizer_for(before.session)
    print(measure(before, after, tokenizer))
    return EXIT_OK


def run_command(config: Config, runtime: Runtime) -> int:
    """Import, chunk, classify, compact, render and measure in one pass.

    The study's command. The four injected dependencies are resolved before any
    work starts and the tokenizer is picked before anything is written, so a run
    that cannot be completed leaves no half-written transcript behind.

    Args:
        config: Needs `agent`, `paths.transcript` and `paths.output`; honours
            `target`, `adapter`, `reduce`, `paths.manifest` and
            `paths.previous_manifest`.
        runtime: Supplies all four dependencies.

    Returns:
        `EXIT_OK`.

    Raises:
        BackendUnavailableError: If the adapter or the narrative backend was
            not injected.
    """
    adapter = runtime.require_adapter(config.adapter)
    runtime.require_narrative_client()
    previous = (
        None
        if config.paths.previous_manifest is None
        else load_manifest(config.paths.previous_manifest)
    )
    timer = RunTimer(runtime.run_clock())
    timer.start()
    imported = import_transcript(config.agent, config.paths.transcript)
    _report(imported.notes)
    tokenizer = runtime.tokenizer_registry().tokenizer_for(imported.session)
    session, chunks = _chunked(imported.session, reduce=config.reduce)
    classification = classify_chunks(
        session, chunks, adapter, candidates=candidate_chains(imported.session)
    )
    block = compacted_block(session, chunks, classification.chains, runtime, config.target_agent)
    output = write_transcript(block, session=session, chunks=chunks, path=config.paths.output)
    metrics = measure(imported, import_transcript(block.rendered_for, output), tokenizer)
    window = timer.finish()
    destination = dump_manifest(
        build_manifest(
            window=window,
            input_path=config.paths.transcript,
            output_path=output,
            config=config.run_config(type(adapter).__name__),
            tokenizer_name=type(tokenizer).__name__,
            metrics=metrics,
            previous=previous,
        ),
        manifest_path(config, output),
    )
    print(f"{len(block.chains)} chain(s) -> {output}")
    print(metrics)
    print(f"manifest -> {destination}")
    return EXIT_OK

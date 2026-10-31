"""Derive and hash the local vocabulary denylist.

``--derive <source>``
    Collect vocabulary candidates from an external source tree: path
    components, Python identifiers, docstrings, and short string constants.
    The §4 naming contract and a generic-programming allowlist are
    subtracted, and so are the tokens of the source's project-agnostic layer
    (the §2 role split), because those are programming vocabulary by
    construction. The
    result is written to the gitignored ``.scrub/denylist.candidates`` for
    human review.

``--hashes``
    Normalize the reviewed ``.scrub/denylist.txt`` and write sorted sha256
    digests to ``tests/vocabulary_denylist.sha256``. Only the digests are
    committed; the reviewed terms stay untracked.

The source path supplied to ``--derive`` is used in-process only. It is
never written to a file, a commit message, or a test.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scrub_repo import (  # noqa: E402
    LOCAL_TERMS_PATH,
    canonicalize,
    digest,
    is_tracked,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = Path(".scrub/denylist.candidates")
HASHES_PATH = Path("tests/vocabulary_denylist.sha256")

SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".pixi",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".pi",
        "output",
        "node_modules",
    }
)
SOURCE_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".txt", ".json"}
)

# The §4 destination vocabulary: names the template is allowed to use.
DESTINATION_VOCABULARY = """
case case_id case_date key key_columns outcome outcome_1 outcome_2 outcome_3
outcome_4 feature feature_1 feature_2 feature_3 feature_4 feature_5 feature_6
feature_7 feature_8 value value_1 source_table_1 model model_00_signal_0
model_00_signal_1 model_00_noise_0 model_00_noise_1 model_01a model_01b
model_02_subset_a model_02_subset_b train valid test s01 s01b s02 s03 s04 s05
s06 outcome_analysis feature divide predict evaluate output stage stage_spec
pipeline_context pipeline_layout identity regression analysis
"""

# Generic programming vocabulary. This list never gates anything; it only
# keeps derivation candidates focused on project vocabulary.
GENERIC_VOCABULARY = """
about above accept accepted access according across act active actual add added
adding after again against aggregate all allow allowed allows along already also
always among amount an and another any anything appear apply applied approach
argument arguments around array as assert assign assigned at attribute avoid
back base based basic be because become been before begin behind being below
best better between beyond body bool boolean both bound boundary break build
built but by cache calculate call called caller can cannot case catch cause cell
change changed check checks choose chosen class clean clear close code collect
collection column columns come command common compare compared comparison
complete complete computed compute concrete config configuration consider
consist constant constants construct constructor contain containing contains
content contents context control convert copy core correct could count create
created current data dataframe default define defined definition delete depend
depends derive derived describe description detail details determine dict
dictionary did different direct directory discard do doc document does done
double down draw driver drop dry due during each early easily edge edit either
element elements else empty enable end enough ensure entire entry equal error
errors establish even event every exact example except exception exclude
expected explicit export express extend extract fact fail failed failure false
far fast few field fields file filed filename filenames filepath filepaths files
fill filter final find fine first fit five fixed flag flags float float64
floating follow following for force form format forward found four frame frames
from full function functions general generated get give given global go good
group groups guard had handle has have here hidden hold how however
identify if ignore import important in include included includes index indicate
individual infer info information initial input inputs inside instead integer
into invalid is issue item items its itself join json just keep key keys kind
known large last late later layout least leave left length less let level like
line lines list lists little load loaded local lock log long look machine made
main make makes making many map matching may mean means measure method methods
might mode model models module modules more most move much must name named names
need needed needs never new next no non none normal not note nothing now null
number object objects of off offset often on once one only open operation
operations optional or order ordered orders other others our out output outputs
over own package pair pairs parse part parts pass path paths pattern per
perform perhaps period place plan point policy possible prefer prefix preserve
previous print process produce produced program project proper provide provided
public put quality query raise random range rather raw reach read reader real
reason received record records refer reference reject rejected relation remain
remove render repeat replace report require required requires resolve resource
response rest result results return returned right root row rows rule rules run
running same save saved schema schemas scope second section see seed select self
separate sequence serial set sets several shape short should show side simple
since single size skip small so some something sort source sources special
split splits stage stages standard start state step steps still store stored
str strategy strategies string strings structure style such suffix support sure
take target targets task team tell temp temporary test tests text than
that the their them then there these they thing things this those three through
time times to together token tokens too top total toward tmp train trained
training transform try tuple two type types under unit units unknown up update
upon us use used user uses using valid validate validation value values
variable variables various via view want was way we well were what when where
whether which while who why will with within without word words work would write
writer yes yet you zero
"""

# Tokens from the destination vocabulary and the generic allowlist.
STOP_TERMS = frozenset(tokenize(DESTINATION_VOCABULARY)) | frozenset(
    tokenize(GENERIC_VOCABULARY)
)

# The source's project-agnostic layer (plan §2): the generic core modules.
# Their tokens are programming vocabulary by construction, so they are
# subtracted from the candidates. Tests are not subtracted: a shared fixture
# can legitimately carry project vocabulary that the gate must still deny.
GENERIC_DIRECTORIES = (("src", "core"),)


def iter_source_files(source: Path) -> Iterator[Path]:
    """Yield the text files of ``source`` that participate in derivation."""
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            yield path


def is_generic_path(relative: Path) -> bool:
    """Return whether ``relative`` belongs to the source's generic layer."""
    parts = relative.parts
    return any(parts[: len(prefix)] == prefix for prefix in GENERIC_DIRECTORIES)


def python_tokens(path: Path) -> tuple[set[str], set[str]]:
    """Return the tokens and phrases of a Python file's code and docstrings."""
    tokens: set[str] = set()
    phrases: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return tokens, phrases
    docstring = ast.get_docstring(tree)
    if docstring:
        tokens.update(tokenize(docstring))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) <= 64:
                names.append(node.value)
        for name in names:
            parts = tokenize(name)
            tokens.update(parts)
            if 2 <= len(parts) <= 3:
                phrases.add(" ".join(parts))
    return tokens, phrases


def path_tokens(relative: Path) -> tuple[set[str], set[str]]:
    """Return the tokens and phrases of a file's path components."""
    tokens: set[str] = set()
    phrases: set[str] = set()
    for component in relative.parts:
        parts = tokenize(component)
        tokens.update(parts)
        if 2 <= len(parts) <= 3:
            phrases.add(" ".join(parts))
    return tokens, phrases


def collect(source: Path) -> tuple[set[str], set[str]]:
    """Collect project-layer tokens and phrases from ``source``."""
    project_tokens: set[str] = set()
    project_phrases: set[str] = set()
    generic_tokens: set[str] = set()
    for path in iter_source_files(source):
        relative = path.relative_to(source)
        tokens, phrases = path_tokens(relative)
        if path.suffix.lower() == ".py":
            python, python_phrases = python_tokens(path)
            tokens |= python
            phrases |= python_phrases
        if is_generic_path(relative):
            generic_tokens |= tokens
        else:
            project_tokens |= tokens
            project_phrases |= phrases
    return project_tokens - generic_tokens, project_phrases


def keep_token(token: str) -> bool:
    """Return whether a unigram is a plausible project-vocabulary term."""
    return len(token) >= 3 and not token.isdigit() and token not in STOP_TERMS


def candidates_from(source: Path) -> list[str]:
    """Return the sorted, de-duplicated candidates for ``source``."""
    project_tokens, project_phrases = collect(source)
    candidates = {token for token in project_tokens if keep_token(token)}
    for phrase in project_phrases:
        parts = phrase.split()
        if all(part not in STOP_TERMS for part in parts) and any(
            keep_token(part) or len(part) >= 2 for part in parts
        ):
            candidates.add(phrase)
    return sorted(candidates)


def _write_atomically(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive(root: Path, source: Path) -> int:
    """Write the derivation candidates and return the exit status."""
    if not source.is_dir():
        print(f"source tree not found: {source}", file=sys.stderr)
        return 1
    if is_tracked(root, LOCAL_TERMS_PATH.as_posix()):
        print("the local terms file is tracked", file=sys.stderr)
        return 1
    candidates = candidates_from(source)
    if not candidates:
        print("derivation produced no candidates", file=sys.stderr)
        return 1
    _write_atomically(root / CANDIDATES_PATH, candidates)
    print(
        f"wrote {len(candidates)} candidates to {CANDIDATES_PATH.as_posix()}; "
        "review them into .scrub/denylist.txt, then run --hashes",
        file=sys.stderr,
    )
    return 0


def write_hashes(root: Path) -> int:
    """Hash the reviewed terms into the committed digest file."""
    terms_file = root / LOCAL_TERMS_PATH
    if is_tracked(root, LOCAL_TERMS_PATH.as_posix()):
        print("the local terms file is tracked", file=sys.stderr)
        return 1
    if not terms_file.is_file():
        print("denylist unavailable: the local terms file is missing", file=sys.stderr)
        return 1
    lines = [
        line.strip() for line in terms_file.read_text(encoding="utf-8").splitlines()
    ]
    lines = [line for line in lines if line]
    if not lines:
        print("denylist unavailable: the local terms file is empty", file=sys.stderr)
        return 1
    normalized: set[str] = set()
    for line in lines:
        if not tokenize(line):
            print(f"denylist entry has no tokens: {line!r}", file=sys.stderr)
            return 1
        normalized.add(canonicalize(line))
    digests = sorted({digest(term) for term in normalized})
    _write_atomically(root / HASHES_PATH, digests)
    print(
        f"wrote {len(digests)} digests to {HASHES_PATH.as_posix()}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Derive and hash the local vocabulary denylist.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--derive", metavar="SOURCE", help="derive candidates")
    modes.add_argument("--hashes", action="store_true", help="hash reviewed terms")
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="repository root (default: this checkout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return the process exit status."""
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    if args.hashes:
        return write_hashes(root)
    return derive(root, Path(args.derive).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())

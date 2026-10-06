#!/usr/bin/env python3
"""
contacts.py - contact store for the Kellogg magneto phone.

Three jobs:

  1. Keep names, numbers and spoken aliases in SQLite so they survive reboots.
  2. Resolve something the caller said into a contact, or honestly report
     that it could not.
  3. Hand asr.py the vocabulary it needs to build a Vosk grammar.

The design rule throughout is that a wrong match is worse than no match. The
operator saying "I don't have anyone by that name" is a minor annoyance.
The operator confidently dialing the wrong person is a real failure, so
lookup() refuses to guess when the evidence is thin and reports ambiguity
instead of silently picking the top score.

Standalone use:

    python src/contacts.py add "Gavin Bowden" 7572747943 --alias gavin
    python src/contacts.py list
    python src/contacts.py find "gavin bowden"
    python src/contacts.py grammar
    python src/contacts.py selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_DB = Path(__file__).resolve().parent.parent / "contacts.db"

# A match at or above this ratio is good enough to act on.
CONFIDENT = 0.84
# If the runner-up is this close to the winner, we do not have a winner.
AMBIGUITY_GAP = 0.07
# Below this, do not even offer it as a candidate.
FLOOR = 0.55

# Spoken digits, for "five five five one two one two".
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9",
}

YES_WORDS = {"yes", "yeah", "yep", "correct", "right", "that's right", "affirmative", "yup"}
NO_WORDS = {"no", "nope", "wrong", "incorrect", "negative"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    number     TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS aliases (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    alias      TEXT    NOT NULL,
    norm       TEXT    NOT NULL,
    auto       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alias_norm ON aliases(norm);
CREATE INDEX IF NOT EXISTS idx_contact_number ON contacts(number);
"""


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------
def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_number(raw: str) -> str:
    """Reduce a dialable number to digits, with US 11-digit trimmed to 10."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def spoken_to_digits(spoken: str) -> str:
    """Turn 'five five five one two one two' into '5551212'.

    Returns an empty string if the phrase is not purely digits, so callers
    can tell "they said a number" from "they said a name".
    """
    out = []
    for word in normalise(spoken).split():
        if word.isdigit():
            out.append(word)
        elif word in DIGIT_WORDS:
            out.append(DIGIT_WORDS[word])
        else:
            return ""
    return "".join(out)


def format_number(number: str) -> str:
    """Human form for readback. Ten digits become (757) 555-1212."""
    d = normalise_number(number)
    if len(d) == 10:
        return f"({d[0:3]}) {d[3:6]}-{d[6:]}"
    if len(d) == 7:
        return f"{d[0:3]}-{d[3:]}"
    return d


def speak_digits(number: str) -> str:
    """Digit-by-digit form."""
    return " ".join(normalise_number(number))


# --------------------------------------------------------------------------
# data types
# --------------------------------------------------------------------------
@dataclass
class Contact:
    id: int
    name: str
    number: str
    aliases: list[str] = field(default_factory=list)

    @property
    def pretty_number(self) -> str:
        return format_number(self.number)

    def __str__(self) -> str:
        return f"{self.name} <{self.pretty_number}>"


@dataclass
class Match:
    """Outcome of a lookup.

    status is one of:
        exact      - the spoken text matched a name or alias outright
        confident  - fuzzy score cleared the threshold with a clear winner
        ambiguous  - two or more contacts scored too closely to choose
        none       - nothing scored high enough to be worth offering
    """
    status: str
    contact: Optional[Contact] = None
    candidates: list[Contact] = field(default_factory=list)
    score: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("exact", "confident")


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
class ContactBook:
    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- writes ----------------

    def add(self, name: str, number: str,
            aliases: Optional[Sequence[str]] = None) -> Contact:
        """Add a contact. Auto-generates first-name and last-name aliases."""
        name = name.strip()
        digits = normalise_number(number)
        if not name:
            raise ValueError("name cannot be empty")
        if not digits:
            raise ValueError(f"no digits found in number: {number!r}")

        cur = self.conn.execute(
            "INSERT INTO contacts (name, number) VALUES (?, ?)", (name, digits))
        contact_id = int(cur.lastrowid)

        wanted: list[tuple[str, int]] = [(name, 1)]
        parts = name.split()
        if len(parts) > 1:
            wanted.append((parts[0], 1))
            wanted.append((parts[-1], 1))
        for a in (aliases or []):
            if a.strip():
                wanted.append((a.strip(), 0))

        seen: set[str] = set()
        for alias, auto in wanted:
            norm = normalise(alias)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            self.conn.execute(
                "INSERT INTO aliases (contact_id, alias, norm, auto) "
                "VALUES (?, ?, ?, ?)", (contact_id, alias, norm, auto))

        self.conn.commit()
        return self.get(contact_id)  # type: ignore[return-value]

    def add_alias(self, contact_id: int, alias: str) -> None:
        norm = normalise(alias)
        if not norm:
            return
        existing = self.conn.execute(
            "SELECT 1 FROM aliases WHERE contact_id=? AND norm=?",
            (contact_id, norm)).fetchone()
        if existing:
            return
        self.conn.execute(
            "INSERT INTO aliases (contact_id, alias, norm, auto) VALUES (?,?,?,0)",
            (contact_id, alias.strip(), norm))
        self.conn.commit()

    def set_number(self, contact_id: int, number: str) -> None:
        digits = normalise_number(number)
        if not digits:
            raise ValueError(f"no digits found in number: {number!r}")
        self.conn.execute("UPDATE contacts SET number=? WHERE id=?",
                          (digits, contact_id))
        self.conn.commit()

    def remove(self, contact_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM contacts WHERE id=?", (contact_id,))
        self.conn.execute("DELETE FROM aliases WHERE contact_id=?", (contact_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------------- reads ----------------

    def _hydrate(self, row: sqlite3.Row) -> Contact:
        aliases = [r["alias"] for r in self.conn.execute(
            "SELECT alias FROM aliases WHERE contact_id=? ORDER BY auto, id",
            (row["id"],))]
        return Contact(id=row["id"], name=row["name"],
                       number=row["number"], aliases=aliases)

    def get(self, contact_id: int) -> Optional[Contact]:
        row = self.conn.execute(
            "SELECT * FROM contacts WHERE id=?", (contact_id,)).fetchone()
        return self._hydrate(row) if row else None

    def all(self) -> list[Contact]:
        return [self._hydrate(r) for r in self.conn.execute(
            "SELECT * FROM contacts ORDER BY name COLLATE NOCASE")]

    def find_by_number(self, number: str) -> Optional[Contact]:
        digits = normalise_number(number)
        row = self.conn.execute(
            "SELECT * FROM contacts WHERE number=?", (digits,)).fetchone()
        return self._hydrate(row) if row else None

    def __len__(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) AS n FROM contacts").fetchone()["n"])

    # ---------------- matching ----------------

    @staticmethod
    def _score(spoken_norm: str, alias_norm: str) -> float:
        """Similarity in 0..1, with a bonus for whole-word overlap.

        Pure character similarity treats 'gavin' against 'gavin reed' as a
        weak match because of the trailing word. Word overlap fixes that,
        which matters because callers usually say a first name alone.
        """
        ratio = SequenceMatcher(None, spoken_norm, alias_norm).ratio()
        s_words = set(spoken_norm.split())
        a_words = set(alias_norm.split())
        if s_words and a_words:
            overlap = len(s_words & a_words) / min(len(s_words), len(a_words))
            ratio = max(ratio, 0.6 * ratio + 0.4 * overlap)
            if s_words <= a_words or a_words <= s_words:
                ratio = max(ratio, 0.90)
        return ratio

    def lookup(self, spoken: str) -> Match:
        """Resolve something the caller said into a contact."""
        spoken_norm = normalise(spoken)
        if not spoken_norm:
            return Match("none")

        # A spoken number short-circuits all the fuzzy work.
        digits = spoken_to_digits(spoken)
        if digits:
            hit = self.find_by_number(digits)
            if hit:
                return Match("exact", contact=hit, score=1.0)
            return Match("none")

        rows = self.conn.execute(
            "SELECT contact_id, norm FROM aliases").fetchall()
        if not rows:
            return Match("none")

        # An exact alias hit wins, but only if exactly one contact owns it.
        # Shared surnames are the common case here: "Reed" is the auto
        # last-name alias for every Reed in the book, and picking the first
        # row would dial a coin flip.
        exact_ids = {r["contact_id"] for r in rows if r["norm"] == spoken_norm}
        if len(exact_ids) == 1:
            contact = self.get(exact_ids.pop())
            if contact:
                return Match("exact", contact=contact, score=1.0)
        if len(exact_ids) > 1:
            cands = [c for c in (self.get(i) for i in sorted(exact_ids)) if c]
            return Match("ambiguous", candidates=cands, score=1.0)

        best: dict[int, float] = {}
        for r in rows:
            s = self._score(spoken_norm, r["norm"])
            cid = r["contact_id"]
            if s > best.get(cid, 0.0):
                best[cid] = s

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [(cid, s) for cid, s in ranked if s >= FLOOR]
        if not ranked:
            return Match("none")

        top_id, top_score = ranked[0]
        close = [cid for cid, s in ranked if top_score - s <= AMBIGUITY_GAP]

        if len(close) > 1:
            cands = [c for c in (self.get(cid) for cid in close) if c]
            return Match("ambiguous", candidates=cands, score=top_score)

        if top_score >= CONFIDENT:
            return Match("confident", contact=self.get(top_id), score=top_score)

        cands = [c for c in (self.get(cid) for cid, _ in ranked[:3]) if c]
        return Match("none", candidates=cands, score=top_score)

    # ---------------- grammar for asr.py ----------------

    def phrases(self) -> list[str]:
        """Every name and alias, normalised, deduplicated."""
        rows = self.conn.execute("SELECT DISTINCT norm FROM aliases").fetchall()
        return sorted({r["norm"] for r in rows if r["norm"]})

    def words(self) -> set[str]:
        """Individual words across all names and aliases."""
        out: set[str] = set()
        for p in self.phrases():
            out.update(p.split())
        return out

    def grammar(self, extra: Optional[Iterable[str]] = None) -> str:
        """JSON grammar string for KaldiRecognizer.

        Vosk takes a JSON list of allowed phrases. Constraining it to the
        contact list plus digits plus yes/no is what makes recognition on a
        Pi both fast and close to exact. '[unk]' is included so genuinely
        unknown speech comes back as unknown rather than being forced onto
        the nearest contact name.
        """
        items: set[str] = set(self.phrases())
        items.update(DIGIT_WORDS.keys())
        items.update(YES_WORDS)
        items.update(NO_WORDS)
        items.update({"operator", "hang up", "goodbye", "cancel", "repeat"})
        if extra:
            items.update(normalise(e) for e in extra if normalise(e))
        items.discard("")
        return json.dumps(sorted(items) + ["[unk]"])

    def unknown_words(self, lexicon: Iterable[str]) -> set[str]:
        """Words in the contact list that the Vosk model cannot produce.

        Vosk silently drops grammar words outside the model's lexicon, so an
        unusual surname can look like a recognition bug when it is really a
        vocabulary gap. Add a phonetic alias for anything this returns, for
        example 'shivawn' for 'Siobhan'.
        """
        known = {w.lower() for w in lexicon}
        return {w for w in self.words() if w not in known}

    # ---------------- lifecycle ----------------

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "ContactBook":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _print_contacts(book: ContactBook) -> None:
    rows = book.all()
    if not rows:
        print("no contacts yet")
        return
    width = max(len(c.name) for c in rows)
    for c in rows:
        extra = [a for a in c.aliases if normalise(a) != normalise(c.name)]
        tail = f"   aliases: {', '.join(extra)}" if extra else ""
        print(f"  [{c.id:>3}] {c.name:<{width}}  {c.pretty_number}{tail}")


def _cmd_find(book: ContactBook, spoken: str) -> None:
    m = book.lookup(spoken)
    print(f"  heard:  {spoken!r}")
    print(f"  status: {m.status}   score {m.score:.2f}")
    if m.contact:
        print(f"  match:  {m.contact}")
        print(f"  readback: \"{m.contact.name}, "
              f"{speak_digits(m.contact.number)}. Is that right?\"")
    for c in m.candidates:
        print(f"  candidate: {c}")
    if m.status == "none" and not m.candidates:
        print("  operator would say: I don't have anyone by that name.")


def _selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "selftest.db"
    with ContactBook(tmp) as book:
        book.add("Gavin Bowden", "757-274-7943", aliases=["gav"])
        book.add("Angie Bowden", "(757) 555-9090", aliases=["maggie", "mum"])
        book.add("Daniel Reed", "7575553434")
        book.add("Theodore Brooks", "17575557777", aliases=["teddy"])

        checks: list[tuple[str, str, Optional[str]]] = [
            ("Gavin Reed", "exact", "Gavin Reed"),
            ("gavin", "exact", "Gavin Reed"),
            ("gav", "exact", "Gavin Reed"),
            ("maggie", "exact", "Margaret Hale"),
            ("mum", "exact", "Margaret Hale"),
            ("teddy", "exact", "Theodore Brooks"),
            ("margret hale", "confident", "Margaret Hale"),
            ("theodor brooks", "confident", "Theodore Brooks"),
            ("seven five seven five five five one two one two",
             "exact", "Gavin Reed"),
            ("7575559090", "exact", "Margaret Hale"),
            ("winston churchill", "none", None),
            ("", "none", None),
        ]

        failures = 0
        for spoken, want_status, want_name in checks:
            m = book.lookup(spoken)
            ok = m.status == want_status and (
                want_name is None or (m.contact and m.contact.name == want_name))
            if not ok:
                failures += 1
                got = m.contact.name if m.contact else None
                print(f"  FAIL {spoken!r}: wanted {want_status}/{want_name}, "
                      f"got {m.status}/{got} ({m.score:.2f})")
            else:
                print(f"  ok   {spoken!r:<52} -> {m.status}")

        # "reed" is shared by two contacts, so it must not resolve silently.
        m = book.lookup("reed")
        if m.status != "ambiguous" or len(m.candidates) < 2:
            failures += 1
            print(f"  FAIL 'reed' should be ambiguous, got {m.status}")
        else:
            names = ", ".join(c.name for c in m.candidates)
            print(f"  ok   {'reed':<52} -> ambiguous ({names})")

        # Number formatting and readback.
        g = book.lookup("gavin").contact
        assert g is not None
        if g.pretty_number != "(757) 555-1212":
            failures += 1
            print(f"  FAIL number format: {g.pretty_number}")
        if speak_digits(g.number) != "7 5 7 5 5 5 1 2 1 2":
            failures += 1
            print(f"  FAIL digit readback: {speak_digits(g.number)}")

        # Leading US 1 should have been trimmed on insert.
        if book.find_by_number("7575557777") is None:
            failures += 1
            print("  FAIL 11-digit number was not normalised to 10")

        # Grammar must include aliases and control words.
        gram = json.loads(book.grammar())
        for needed in ("gavin reed", "maggie", "yes", "no", "seven", "[unk]"):
            if needed not in gram:
                failures += 1
                print(f"  FAIL grammar missing {needed!r}")

        # Deleting a contact removes its aliases from the grammar.
        book.remove(g.id)
        if "gav" in book.phrases():
            failures += 1
            print("  FAIL alias survived contact deletion")

        print()
        if failures:
            print(f"{failures} failure(s)")
        else:
            print(f"all checks passed ({len(book)} contacts remain)")
        return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kellogg phone contact book")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="path to contacts.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="add a contact")
    p.add_argument("name")
    p.add_argument("number")
    p.add_argument("--alias", action="append", default=[],
                   help="spoken alias; repeatable")

    sub.add_parser("list", help="list contacts")

    p = sub.add_parser("remove", help="remove a contact by id")
    p.add_argument("id", type=int)

    p = sub.add_parser("alias", help="add an alias to an existing contact")
    p.add_argument("id", type=int)
    p.add_argument("alias")

    p = sub.add_parser("find", help="resolve a spoken phrase")
    p.add_argument("spoken", nargs="+")

    sub.add_parser("grammar", help="print the Vosk grammar JSON")
    sub.add_parser("selftest", help="run built-in checks against a temp db")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        return _selftest()

    with ContactBook(args.db) as book:
        if args.cmd == "add":
            c = book.add(args.name, args.number, args.alias)
            print(f"added [{c.id}] {c}")
            if c.aliases:
                print(f"  aliases: {', '.join(c.aliases)}")
        elif args.cmd == "list":
            _print_contacts(book)
        elif args.cmd == "remove":
            print("removed" if book.remove(args.id) else f"no contact {args.id}")
        elif args.cmd == "alias":
            book.add_alias(args.id, args.alias)
            print(f"alias added to [{args.id}]")
        elif args.cmd == "find":
            _cmd_find(book, " ".join(args.spoken))
        elif args.cmd == "grammar":
            print(book.grammar())
    return 0


if __name__ == "__main__":
    sys.exit(main())
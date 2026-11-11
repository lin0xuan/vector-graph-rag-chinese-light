"""The local manifest defines which immutable vector records are published."""
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    location: str
    text: str

    def as_dict(self):
        return asdict(self)


class Index:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "index.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL,
                    location TEXT NOT NULL, text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS passages_source ON passages(source);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES ('namespace', ?)", (uuid4().hex,))
            self.namespace = db.execute("SELECT value FROM settings WHERE key='namespace'").fetchone()[0]

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            with db:
                if write:
                    db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    def passages(self):
        with self.connection() as db:
            return [Passage(*row) for row in db.execute(
                "SELECT id, source, location, text FROM passages ORDER BY source, rowid"
            )]

    def replace(self, db, source, passages):
        db.execute("DELETE FROM passages WHERE source=?", (source,))
        db.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?)",
            ((p.id, p.source, p.location, p.text) for p in passages),
        )

    def remove(self, source):
        with self.connection(write=True) as db:
            return db.execute("DELETE FROM passages WHERE source=?", (source,)).rowcount

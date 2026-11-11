"""The local manifest defines which immutable vector records are published."""
import json
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
    title: str | None = None
    chunk_metadata: dict | None = None

    def as_dict(self):
        result = asdict(self)
        result.pop("title")
        result.update(result.pop("chunk_metadata") or {})
        return result


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
                    location TEXT NOT NULL, text TEXT NOT NULL, title TEXT, chunk_metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS passages_source ON passages(source);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES ('namespace', ?)", (uuid4().hex,))
            self.namespace = db.execute("SELECT value FROM settings WHERE key='namespace'").fetchone()[0]

        with self.connection(write=True) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(passages)")}
            if "title" not in columns:
                db.execute("ALTER TABLE passages ADD COLUMN title TEXT")
            if "chunk_metadata" not in columns:
                db.execute("ALTER TABLE passages ADD COLUMN chunk_metadata TEXT")

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
            return [Passage(*row[:-1], json.loads(row[-1]) if row[-1] else None) for row in db.execute(
                "SELECT id, source, location, text, title, chunk_metadata FROM passages ORDER BY source, rowid"
            )]

    def replace(self, db, source, passages):
        db.execute("DELETE FROM passages WHERE source=?", (source,))
        db.executemany(
            "INSERT INTO passages (id, source, location, text, title, chunk_metadata) VALUES (?, ?, ?, ?, ?, ?)",
            ((p.id, p.source, p.location, p.text, p.title,
              json.dumps(p.chunk_metadata, ensure_ascii=False) if p.chunk_metadata else None) for p in passages),
        )

    def remove(self, source):
        with self.connection(write=True) as db:
            return db.execute("DELETE FROM passages WHERE source=?", (source,)).rowcount

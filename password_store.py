"""Utilities for securely storing hashed passwords in an SQLite database.

This module provides helper functions for creating a credential database that
stores only salted password hashes using PBKDF2-HMAC.  It also exposes a simple
command-line interface for registering users and validating credentials.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Tuple

# Default configuration values. The iteration count is high enough to resist
# brute-force attempts while remaining fast for legitimate usage.
PBKDF2_ITERATIONS = 390000
HASH_NAME = "sha256"
SALT_BYTES = 16


@contextmanager
def _connect(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection that enforces foreign keys."""
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
    finally:
        connection.close()


def initialize_database(db_path: Path) -> None:
    """Create the credential table if it does not exist."""
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def _generate_salt() -> bytes:
    return os.urandom(SALT_BYTES)


def _hash_password(password: str, salt: bytes, iterations: int) -> str:
    """Return a base64-encoded PBKDF2-HMAC hash."""
    derived = hashlib.pbkdf2_hmac(
        HASH_NAME,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return base64.b64encode(derived).decode("ascii")


def _encode_salt(salt: bytes) -> str:
    return base64.b64encode(salt).decode("ascii")


def _decode_salt(encoded: str) -> bytes:
    return base64.b64decode(encoded.encode("ascii"))


def create_user(db_path: Path, username: str, password: str) -> None:
    """Create a new user storing only the password hash and salt."""
    initialize_database(db_path)
    salt = _generate_salt()
    iterations = PBKDF2_ITERATIONS
    password_hash = _hash_password(password, salt, iterations)
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO users (username, password_hash, salt, iterations, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash=excluded.password_hash,
                salt=excluded.salt,
                iterations=excluded.iterations,
                updated_at=excluded.updated_at
            """,
            (
                username,
                password_hash,
                _encode_salt(salt),
                iterations,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()


def verify_user(db_path: Path, username: str, password: str) -> bool:
    """Validate a password against the stored hash."""
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "SELECT password_hash, salt, iterations FROM users WHERE username = ?",
            (username,),
        )
        row: Optional[Tuple[str, str, int]] = cursor.fetchone()

    if row is None:
        return False

    stored_hash, encoded_salt, iterations = row
    salt = _decode_salt(encoded_salt)
    computed_hash = _hash_password(password, salt, iterations)
    return hashlib.compare_digest(stored_hash, computed_hash)


def list_users(db_path: Path) -> list[Tuple[int, str, str, str]]:
    """Return a list of existing users and metadata (without password hashes)."""
    initialize_database(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "SELECT id, username, created_at, updated_at FROM users ORDER BY username"
        )
        return list(cursor.fetchall())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gerencia um banco seguro de senhas usando hashes PBKDF2-HMAC.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("credentials.db"),
        help="Caminho para o arquivo SQLite (padrão: credentials.db)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Cria ou atualiza um usuário")
    add_parser.add_argument("username", help="Nome de usuário")
    add_parser.add_argument("password", help="Senha em texto puro (não será armazenada)")

    verify_parser = subparsers.add_parser("verify", help="Verifica uma senha")
    verify_parser.add_argument("username", help="Nome de usuário")
    verify_parser.add_argument("password", help="Senha a validar")

    subparsers.add_parser("list", help="Lista usuários cadastrados")

    args = parser.parse_args()

    if args.command == "add":
        create_user(args.database, args.username, args.password)
        print(f"Usuário '{args.username}' foi criado/atualizado com sucesso.")
    elif args.command == "verify":
        if verify_user(args.database, args.username, args.password):
            print("Senha válida.")
        else:
            print("Senha inválida ou usuário inexistente.")
    elif args.command == "list":
        users = list_users(args.database)
        for user_id, username, created_at, updated_at in users:
            print(f"{user_id}: {username} (criado: {created_at}, atualizado: {updated_at})")


if __name__ == "__main__":
    main()

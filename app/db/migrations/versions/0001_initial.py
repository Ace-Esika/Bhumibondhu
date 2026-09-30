"""Initial schema: extensions, source/document/chunk tables, FTS and HNSW indexes.

Written as explicit, idempotent SQL (IF NOT EXISTS) so it is safe to re-run against a
database that was partially migrated.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-29
"""

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id              BIGSERIAL PRIMARY KEY,
            status          TEXT NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued','running','succeeded','partial','failed')),
            trigger         TEXT NOT NULL DEFAULT 'cli',
            source_types    TEXT[] NOT NULL DEFAULT '{}',
            options         JSONB NOT NULL DEFAULT '{}'::jsonb,
            requested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at      TIMESTAMPTZ,
            completed_at    TIMESTAMPTZ,
            records_fetched INTEGER NOT NULL DEFAULT 0,
            records_created INTEGER NOT NULL DEFAULT 0,
            records_updated INTEGER NOT NULL DEFAULT 0,
            records_deleted INTEGER NOT NULL DEFAULT 0,
            records_skipped INTEGER NOT NULL DEFAULT 0,
            errors          INTEGER NOT NULL DEFAULT 0,
            error_details   JSONB NOT NULL DEFAULT '[]'::jsonb,
            stats           JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_ingestion_runs_status ON ingestion_runs (status, requested_at)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id                BIGSERIAL PRIMARY KEY,
            source_type       TEXT NOT NULL,
            source_id         TEXT NOT NULL,
            raw               JSONB NOT NULL,
            content_hash      TEXT NOT NULL,
            source_created_at TIMESTAMPTZ,
            first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_run_id       BIGINT REFERENCES ingestion_runs(id) ON DELETE SET NULL,
            is_deleted        BOOLEAN NOT NULL DEFAULT false,
            deleted_at        TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_sources_type_id UNIQUE (source_type, source_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id                BIGSERIAL PRIMARY KEY,
            source_pk         BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            source_type       TEXT NOT NULL,
            source_id         TEXT NOT NULL,
            parent_source_id  TEXT,
            title             TEXT NOT NULL,
            url               TEXT,
            file_url          TEXT,
            doc_type          TEXT,
            category          TEXT,
            year              INTEGER,
            act_number        TEXT,
            publication_date  DATE,
            author            TEXT,
            authority         SMALLINT NOT NULL DEFAULT 0,
            metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
            content_hash      TEXT NOT NULL,
            is_active         BOOLEAN NOT NULL DEFAULT true,
            source_updated_at TIMESTAMPTZ,
            ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_documents_type_id UNIQUE (source_type, source_id)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_source_pk ON documents (source_pk)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_type_active ON documents (source_type, is_active)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_doc_type ON documents (doc_type) WHERE is_active")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id                   BIGSERIAL PRIMARY KEY,
            document_id          BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index          INTEGER NOT NULL,
            source_type          TEXT NOT NULL,
            source_id            TEXT NOT NULL,
            parent_source_id     TEXT,
            doc_source_type      TEXT NOT NULL,
            doc_type             TEXT,
            category             TEXT,
            year                 INTEGER,
            authority            SMALLINT NOT NULL DEFAULT 0,
            title                TEXT NOT NULL,
            section_number       TEXT,
            section_heading      TEXT,
            context              TEXT NOT NULL,
            content              TEXT NOT NULL,
            lexical_title        TEXT NOT NULL DEFAULT '',
            lexical_body         TEXT NOT NULL DEFAULT '',
            search_vector        TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('simple'::regconfig, coalesce(lexical_title, '')), 'A') ||
                setweight(to_tsvector('simple'::regconfig, coalesce(lexical_body, '')), 'B')
            ) STORED,
            token_count          INTEGER NOT NULL DEFAULT 0,
            metadata             JSONB NOT NULL DEFAULT '{}'::jsonb,
            content_hash         TEXT NOT NULL,
            embedding            vector(1024),
            embedding_model      TEXT,
            embedding_created_at TIMESTAMPTZ,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_chunks_doc_idx UNIQUE (document_id, chunk_index)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_document ON document_chunks (document_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_search_vector ON document_chunks USING GIN (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_filters ON document_chunks (doc_source_type, doc_type, year)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_category ON document_chunks (category)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_section ON document_chunks (section_number)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_content_hash ON document_chunks (content_hash)")
    # Work queue for the embedder: chunks without a vector.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_pending_embedding ON document_chunks (id) WHERE embedding IS NULL"
    )
    # HNSW created once here (not per ingestion). Cosine distance matches normalised BGE-M3 vectors.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_jobs (
            id              BIGSERIAL PRIMARY KEY,
            run_id          BIGINT REFERENCES ingestion_runs(id) ON DELETE SET NULL,
            model           TEXT NOT NULL,
            mode            TEXT NOT NULL DEFAULT 'pending',
            status          TEXT NOT NULL DEFAULT 'running',
            chunks_total    INTEGER NOT NULL DEFAULT 0,
            chunks_embedded INTEGER NOT NULL DEFAULT 0,
            chunks_failed   INTEGER NOT NULL DEFAULT 0,
            started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at    TIMESTAMPTZ,
            error           TEXT
        )
        """
    )

    # updated_at maintenance for rows touched outside the ORM.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
        BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql
        """
    )
    for table in ("sources", "documents", "document_chunks"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table}")
        op.execute(
            f"CREATE TRIGGER trg_{table}_updated_at BEFORE UPDATE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION set_updated_at()"
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_jobs")
    op.execute("DROP TABLE IF EXISTS document_chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS sources")
    op.execute("DROP TABLE IF EXISTS ingestion_runs")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

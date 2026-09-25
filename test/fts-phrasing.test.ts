/**
 * Phrasing-robust keyword retrieval (FTS5 tiered fallback).
 *
 * Regression coverage for the phrasing-sensitivity bug: adding an
 * interrogative frame ("where does X live", "which files use Y") used to turn
 * a strong keyword match into zero results because every token — including
 * "where"/"does"/"live" — was ANDed into the FTS5 MATCH. searchFTS now falls
 * back through relaxed tiers (drop stopwords, then OR of content terms with a
 * coverage floor) so question phrasings still retrieve, while genuinely-absent
 * queries still return nothing.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { createHash } from "crypto";
import type { Database } from "../src/db.js";

const tempDir = mkdtempSync(join(tmpdir(), "qmd-fts-phrasing-"));
process.env.INDEX_PATH = join(tempDir, "fts-phrasing.sqlite");

// Import AFTER INDEX_PATH is set so the store does not touch the global index.
const { createStore, searchFTS, insertDocument, insertContent } = await import("../src/store");

interface Doc {
  file: string;
  title: string;
  body: string;
}

const docs: Doc[] = [
  {
    file: "availability_detection_service.md",
    title: "app/domain/services/availability_detection_service.py",
    body:
      "AvailabilityDetectionService is the domain service that detects consultant " +
      "availability conflicts for a date range across calendar entries.",
  },
  {
    file: "project_creation_service.md",
    title: "app/domain/services/project_creation_service.py",
    body:
      "ProjectCreationService handles project creation: validates the create DTO " +
      "and persists a new project with its default phases.",
  },
  {
    file: "annotations_helper.md",
    title: "scripts/type_annotations_helper.py",
    body:
      "Utility for inspecting Python type annotations on functions and classes. " +
      "Nothing to do with clusters or networking.",
  },
  {
    file: "calendar_endpoint.md",
    title: "app/api/v1/calendar.py",
    body:
      "Calendar endpoint exposing GET and POST for calendar entries used by the " +
      "scheduling UI.",
  },
  {
    file: "agents.md",
    title: "docs/agents.md",
    body: "Notes on multi-agent orchestration and handoff between workers.",
  },
];

describe("FTS phrasing-robust tiers", () => {
  let store: ReturnType<typeof createStore>;
  let db: Database;

  beforeAll(() => {
    store = createStore();
    db = store.db;
    const now = new Date().toISOString();
    for (const d of docs) {
      const hash = createHash("sha256").update(d.file).digest("hex").slice(0, 12);
      insertContent(db, hash, `# ${d.title}\n\n${d.body}`, now);
      insertDocument(db, "test", d.file, d.title, hash, now, now);
    }
  });

  afterAll(() => {
    store.close();
    rmSync(tempDir, { recursive: true, force: true });
  });

  const top = (q: string) => searchFTS(db, q, 5).map(r => r.title);

  test("baseline: exact keyword query still matches (unchanged strict AND)", () => {
    expect(top("consultant availability")).toContain(
      "app/domain/services/availability_detection_service.py",
    );
  });

  test("interrogative frame no longer zeroes results ('where does X live')", () => {
    // Every term ANDed would fail: no doc contains "where"/"does"/"live".
    const results = top("where does consultant availability live");
    expect(results.length).toBeGreaterThan(0);
    expect(results).toContain(
      "app/domain/services/availability_detection_service.py",
    );
  });

  test("'which files use X endpoint' recovers the endpoint doc", () => {
    const results = top("which files use calendar endpoint");
    expect(results).toContain("app/api/v1/calendar.py");
  });

  test("'where does project creation live' recovers the creation service", () => {
    const results = top("where does project creation live");
    expect(results).toContain(
      "app/domain/services/project_creation_service.py",
    );
  });

  test("nonsense multi-term query still returns nothing (coverage floor)", () => {
    // Only "annotations" exists in the corpus; "kubernetes"/"ingress" do not.
    // The OR tier must NOT surface the annotations doc on a single-term overlap.
    expect(top("kubernetes ingress annotations")).toEqual([]);
  });

  test("query about an entirely absent topic returns nothing", () => {
    expect(top("redis pubsub replication lag")).toEqual([]);
  });

  test("OR tier applies negation to every alternative, not just the last", () => {
    // Strict AND fails (no doc has consultant + scheduling), so this reaches the
    // OR tier. `a OR b OR c NOT d` would bind as `a OR b OR (c NOT d)` and let
    // the calendar doc back in through "scheduling".
    const results = top("consultant scheduling entries -endpoint");
    expect(results).toContain("app/domain/services/availability_detection_service.py");
    expect(results).not.toContain("app/api/v1/calendar.py");
  });

  test("OR-tier coverage counts hyphenated compounds ('multi-agent')", () => {
    // "kubernetes" is absent, so only the OR tier can match; the doc covers
    // "multi agent" + "orchestration" only if the raw hyphen is normalized.
    expect(top("multi-agent orchestration kubernetes")).toContain("docs/agents.md");
  });

  test("single content word that is absent returns nothing", () => {
    expect(top("where is the mongodb sharding")).toEqual([]);
  });
});

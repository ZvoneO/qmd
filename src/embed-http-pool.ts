// =============================================================================
// HTTP Embedding Pool — multi-server work-stealing embedding backend
// =============================================================================
//
// Self-contained fork-only module. Lives outside src/llm.ts so future upstream
// merges of llm.ts have zero conflicts here. Speaks Ollama's /api/embed
// protocol today; can be widened to OpenAI /v1/embeddings later without
// touching the embed loop in src/cli/qmd.ts or src/store.ts.
//
// To re-wire into the CLI behind QMD_EMBED_URLS, see the TODO at the bottom.

import type { Database } from "./db.js";
import type { EmbeddingResult, EmbedOptions as LlmEmbedOptions } from "./llm.js";
import { formatDocForEmbedding } from "./llm.js";
import {
  type Store,
  type EmbedOptions,
  type EmbedResult,
  chunkDocumentByTokens,
  extractTitle,
  insertEmbedding,
  clearAllEmbeddings,
} from "./store.js";
import { BackendLifecycle, loadBackendConfig } from "./embed-lifecycle.js";

export type HttpEmbedConfig = {
  /** Backend server URLs (default: localhost:11434, override via QMD_EMBED_URLS or QMD_OLLAMA_URLS env) */
  urls?: string[];
  /** Model name on the backend (default: embeddinggemma:300m, override via QMD_OLLAMA_MODEL env) */
  model?: string;
  /** Request timeout in ms (default: 120000) */
  timeoutMs?: number;
};

const DEFAULT_URLS = ["http://localhost:11434"];
// Ollama expects a model name like "embeddinggemma:300m", not the llama.cpp HF URI
// used by node-llama-cpp. Override with QMD_OLLAMA_MODEL env var.
const DEFAULT_MODEL = "embeddinggemma:300m";

export type ServerStats = {
  url: string;
  textsProcessed: number;
  requestCount: number;
  totalMs: number;
};

/**
 * Embedding backend using Ollama HTTP API (multi-server).
 * Probes all configured URLs and uses all reachable servers in parallel.
 * Batches are split across active servers for maximum throughput.
 * If a server fails mid-run, its work is redistributed to survivors.
 *
 * Configure multiple servers via QMD_EMBED_URLS (preferred) or QMD_OLLAMA_URLS:
 *   QMD_EMBED_URLS=http://gpu1:11434,http://gpu2:11434,http://localhost:11434
 *
 * Configure model via QMD_OLLAMA_MODEL environment variable:
 *   QMD_OLLAMA_MODEL=qwen3-embedding:0.6b
 */
type Protocol = "llama" | "ollama";

export class HttpEmbedPool {
  private urls: string[];
  private model: string;
  private timeoutMs: number;
  private activeUrls: string[] = [];
  private probed = false;
  private stats: Map<string, ServerStats> = new Map();
  private protocols: Map<string, Protocol> = new Map();
  private dimensions: number = 0;

  constructor(config: HttpEmbedConfig = {}) {
    const envUrls = process.env.QMD_EMBED_URLS || process.env.QMD_OLLAMA_URLS;
    this.urls = config.urls
      ?? (envUrls
        ? envUrls.split(",").map((u) => u.trim())
        : DEFAULT_URLS);
    this.model = config.model ?? (process.env.QMD_OLLAMA_MODEL || DEFAULT_MODEL);
    this.timeoutMs = config.timeoutMs ?? 120_000;
  }

  /** Probe a single URL. Tries llama-server /health first, falls back to Ollama /api/embed. */
  private async probeOne(url: string): Promise<number> {
    const t0 = Date.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15_000);
    try {
      // llama-server: GET /health is cheap and definitive
      try {
        const res = await fetch(`${url}/health`, { signal: controller.signal });
        if (res.ok) {
          const body = await res.json().catch(() => ({})) as { status?: string };
          if (body.status === "ok") {
            this.protocols.set(url, "llama");
            return Date.now() - t0;
          }
        }
      } catch {
        // fall through to Ollama probe
      }
      // Ollama: POST /api/embed
      try {
        const res = await fetch(`${url}/api/embed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: this.model, input: ["probe"], keep_alive: -1 }),
          signal: controller.signal,
        });
        if (res.ok) {
          const data = await res.json() as { embeddings: number[][] };
          if (data.embeddings?.[0] && !this.dimensions) {
            this.dimensions = data.embeddings[0].length;
          }
          this.protocols.set(url, "ollama");
          return Date.now() - t0;
        }
      } catch {
        // unreachable
      }
    } finally {
      clearTimeout(timeout);
    }
    return -1;
  }

  /** Probe all URLs in parallel. Populates activeUrls with all reachable servers. */
  private async probeAll(): Promise<void> {
    const results = await Promise.all(
      this.urls.map(async (url) => {
        const ms = await this.probeOne(url);
        return { url, ok: ms >= 0, ms };
      })
    );
    this.activeUrls = results.filter((r) => r.ok).map((r) => r.url);
    for (const url of this.activeUrls) {
      this.stats.set(url, { url, textsProcessed: 0, requestCount: 0, totalMs: 0 });
    }
    if (this.activeUrls.length === 0) {
      throw new Error(
        `No reachable embedding server. Tried: ${this.urls.join(", ")}`
      );
    }
    const serverList = results.map((r) => {
      const name = r.url.replace("http://", "").replace(":11434", "").replace(":8081", "");
      const proto = this.protocols.get(r.url);
      return r.ok ? `${name} ${r.ms}ms (${proto})` : `${name} ✗`;
    }).join("  ");
    process.stderr.write(`Servers (ping): ${serverList}\n`);
  }

  /** Ensure we have at least one working URL. Probes all on first call. */
  private async ensureActive(): Promise<string[]> {
    if (!this.probed) {
      await this.probeAll();
      this.probed = true;
    }
    return this.activeUrls;
  }

  /** Send embed request to a specific URL — branches on detected protocol. */
  private async fetchEmbed(baseUrl: string, input: string[]): Promise<number[][]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    const proto = this.protocols.get(baseUrl) ?? "ollama";
    try {
      if (proto === "llama") {
        // llama-server speaks OpenAI /v1/embeddings.
        // model can be empty when only one model is loaded; we still pass-through
        // for clarity in logs. response: { data: [{ embedding, index }] }
        const res = await fetch(`${baseUrl}/v1/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: this.model, input }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.text().catch(() => "");
          throw new Error(`Embed backend ${res.status}: ${body}`);
        }
        const data = (await res.json()) as {
          data: { embedding: number[]; index?: number }[];
        };
        // Sort by index in case server reorders; default to input order otherwise.
        const ordered = [...data.data].sort(
          (a, b) => (a.index ?? 0) - (b.index ?? 0),
        );
        const embeddings = ordered.map((d) => d.embedding);
        if (!this.dimensions && embeddings[0]) {
          this.dimensions = embeddings[0].length;
        }
        return embeddings;
      }
      // Ollama path (existing).
      const res = await fetch(`${baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, input, keep_alive: -1 }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`Embed backend ${res.status}: ${body}`);
      }
      const data = (await res.json()) as { embeddings: number[][] };
      return data.embeddings;
    } finally {
      clearTimeout(timeout);
    }
  }

  async embed(text: string, _options: LlmEmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const urls = await this.ensureActive();
      const [embedding] = await this.fetchEmbed(urls[0]!, [text]);
      return { embedding: embedding!, model: this.model };
    } catch (error) {
      console.error("HttpEmbedPool embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const urls = await this.ensureActive();

    if (urls.length === 1) {
      try {
        const t0 = Date.now();
        const embeddings = await this.fetchEmbed(urls[0]!, texts);
        const elapsed = Date.now() - t0;
        const s = this.stats.get(urls[0]!);
        if (s) { s.textsProcessed += texts.length; s.requestCount++; s.totalMs += elapsed; }
        return embeddings.map((embedding) => ({ embedding, model: this.model }));
      } catch (error) {
        console.error("HttpEmbedPool batch embedding error:", error);
        return texts.map(() => null);
      }
    }

    // Multiple servers — work-queue pattern.
    const SUB_BATCH = 32;
    const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
    const queue: { startIdx: number; texts: string[] }[] = [];
    for (let i = 0; i < texts.length; i += SUB_BATCH) {
      queue.push({ startIdx: i, texts: texts.slice(i, i + SUB_BATCH) });
    }

    let queueIdx = 0;
    const worker = async (url: string) => {
      while (true) {
        const idx = queueIdx++;
        if (idx >= queue.length) break;
        const item = queue[idx]!;
        try {
          const t0 = Date.now();
          const embeddings = await this.fetchEmbed(url, item.texts);
          const elapsed = Date.now() - t0;
          const s = this.stats.get(url);
          if (s) { s.textsProcessed += item.texts.length; s.requestCount++; s.totalMs += elapsed; }
          for (let i = 0; i < embeddings.length; i++) {
            results[item.startIdx + i] = { embedding: embeddings[i]!, model: this.model };
          }
        } catch {
          process.stderr.write(`\nHttpEmbedPool: ${url} failed, removing from pool\n`);
          this.activeUrls = this.activeUrls.filter((u) => u !== url);
          queue.push(item);
          break;
        }
      }
    };
    await Promise.all(urls.map((url) => worker(url)));
    return results;
  }

  /**
   * Stream-process a large set of texts across all servers.
   * Each server runs independently, pulling sub-batches from a shared queue.
   * Calls onBatch after each sub-batch completes — use for DB writes + progress.
   * Pipelines fetch/write to keep GPUs busy.
   */
  async embedStream(
    texts: string[],
    onBatch: (startIdx: number, results: (EmbeddingResult | null)[]) => void,
    subBatchSize: number = 16
  ): Promise<void> {
    const urls = await this.ensureActive();
    const queue: { startIdx: number; texts: string[] }[] = [];
    for (let i = 0; i < texts.length; i += subBatchSize) {
      queue.push({ startIdx: i, texts: texts.slice(i, i + subBatchSize) });
    }

    let queueIdx = 0;
    const MAX_RETRIES = 3;

    const worker = async (url: string) => {
      const name = url.replace("http://", "").replace(":11434", "");
      let consecutiveFailures = 0;
      let pendingWrite: { startIdx: number; results: (EmbeddingResult | null)[] } | null = null;

      while (true) {
        const idx = queueIdx++;
        if (idx >= queue.length) break;
        const item = queue[idx]!;

        let success = false;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            const t0 = Date.now();
            const fetchPromise = this.fetchEmbed(url, item.texts);

            // While GPU is working, process the PREVIOUS batch's DB write
            if (pendingWrite) {
              onBatch(pendingWrite.startIdx, pendingWrite.results);
              pendingWrite = null;
            }

            const embeddings = await fetchPromise;
            const elapsed = Date.now() - t0;
            const s = this.stats.get(url);
            if (s) { s.textsProcessed += item.texts.length; s.requestCount++; s.totalMs += elapsed; }

            const results: (EmbeddingResult | null)[] = embeddings.map((e) => ({ embedding: e!, model: this.model }));
            pendingWrite = { startIdx: item.startIdx, results };
            consecutiveFailures = 0;
            success = true;
            break;
          } catch (err: any) {
            const msg = err?.message || err;
            if (attempt < MAX_RETRIES) {
              process.stderr.write(`\nHttpEmbedPool: ${name} attempt ${attempt}/${MAX_RETRIES} failed: ${msg} — retrying...\n`);
              await new Promise((r) => setTimeout(r, 1000 * attempt));
            } else {
              process.stderr.write(`\nHttpEmbedPool: ${name} failed after ${MAX_RETRIES} attempts: ${msg}\n`);
              consecutiveFailures++;
            }
          }
        }

        if (!success) {
          queue.push(item);
          if (consecutiveFailures >= 2) {
            process.stderr.write(`\nHttpEmbedPool: removing ${name} (${consecutiveFailures} consecutive failures)\n`);
            this.activeUrls = this.activeUrls.filter((u) => u !== url);
            break;
          }
        }
      }

      if (pendingWrite) {
        onBatch(pendingWrite.startIdx, pendingWrite.results);
      }
    };

    await Promise.all(urls.map((url) => worker(url)));
  }

  getActiveUrls(): string[] { return this.activeUrls; }

  /** Configured URLs, including unprobed ones. */
  getConfiguredUrls(): string[] { return [...this.urls]; }
  getDimensions(): number { return this.dimensions; }
  getStats(): ServerStats[] { return Array.from(this.stats.values()); }

  /** Print a summary of per-server work distribution to stderr */
  printStats(): void {
    const stats = this.getStats().filter((s) => s.textsProcessed > 0);
    if (stats.length === 0) return;
    const total = stats.reduce((sum, s) => sum + s.textsProcessed, 0);
    process.stderr.write(`\nServer work distribution:\n`);
    for (const s of stats) {
      const name = s.url.replace("http://", "").replace(":11434", "");
      const pct = ((s.textsProcessed / total) * 100).toFixed(0);
      const avg = s.requestCount > 0 ? (s.totalMs / s.requestCount).toFixed(0) : "0";
      const tps = s.totalMs > 0 ? ((s.textsProcessed / s.totalMs) * 1000).toFixed(0) : "0";
      process.stderr.write(`  ${name.padEnd(12)} ${String(s.textsProcessed).padStart(6)} texts (${pct.padStart(3)}%)  ${tps.padStart(4)} texts/s  avg ${avg}ms/batch\n`);
    }
  }

  async dispose(): Promise<void> {}
}

// =============================================================================
// Singleton for default HttpEmbedPool instance
// =============================================================================

let defaultPool: HttpEmbedPool | null = null;

/** Get the default HttpEmbedPool instance (creates one if needed). */
export function getDefaultEmbedPool(): HttpEmbedPool {
  if (!defaultPool) {
    defaultPool = new HttpEmbedPool();
  }
  return defaultPool;
}

export function setDefaultEmbedPool(pool: HttpEmbedPool | null): void {
  defaultPool = pool;
}

/** Print per-server stats from the default HttpEmbedPool instance. */
export function printPoolStats(): void {
  if (defaultPool) {
    defaultPool.printStats();
  }
}

// =============================================================================
// Sibling of store.ts:generateEmbeddings — uses HttpEmbedPool for the
// embedding step. Replicates the pending-docs query + chunking + insertion
// loop so we don't depend on un-exported store internals. All persistence
// goes through the vec0-safe insertEmbedding() exported from store.ts.
// =============================================================================

type PendingDoc = {
  hash: string;
  path: string;
  bytes: number;
  body: string;
};

function fetchPendingDocs(db: Database, collection?: string): PendingDoc[] {
  const collectionFilter = collection ? `AND d.collection = ?` : ``;
  const stmt = db.prepare(`
    SELECT
      d.hash AS hash,
      MIN(d.path) AS path,
      length(CAST(c.doc AS BLOB)) AS bytes,
      c.doc AS body
    FROM documents d
    JOIN content c ON d.hash = c.hash
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL ${collectionFilter}
    GROUP BY d.hash
    ORDER BY MIN(d.path)
  `);
  return (collection ? stmt.all(collection) : stmt.all()) as PendingDoc[];
}

type ChunkItem = {
  hash: string;
  title: string;
  text: string;
  seq: number;
  pos: number;
  bytes: number;
};

/**
 * Sibling of generateEmbeddings() in store.ts that uses the multi-server
 * HttpEmbedPool. Same input/output contract, so the CLI swap is one line.
 *
 * Flow:
 *   1. Optionally clear existing embeddings if force=true.
 *   2. Pull pending docs (those without seq=0 entry in content_vectors).
 *   3. Chunk each doc with upstream's chunkDocumentByTokens.
 *   4. Probe pool, ensure vec0 table at the discovered dimension.
 *   5. embedStream() fans batches across all servers; each callback writes
 *      its slice via insertEmbedding() (vec0-safe DELETE+INSERT).
 */
export async function generateEmbeddingsViaPool(
  store: Store,
  options?: EmbedOptions,
): Promise<EmbedResult> {
  const db = store.db;
  const startTime = Date.now();

  if (options?.force) {
    clearAllEmbeddings(db, options.collection);
  }

  const docs = fetchPendingDocs(db, options?.collection);
  if (docs.length === 0) {
    return { docsProcessed: 0, chunksEmbedded: 0, errors: 0, durationMs: 0 };
  }

  const totalDocs = docs.length;

  // Chunk every pending doc up front. With ~40 docs/sec chunking speed this
  // is fast even for tens of thousands; embedding is the real bottleneck.
  const encoder = new TextEncoder();
  const allChunks: ChunkItem[] = [];
  for (const doc of docs) {
    if (!doc.body || !doc.body.trim()) continue;
    const title = extractTitle(doc.body, doc.path);
    const chunks = await chunkDocumentByTokens(
      doc.body,
      undefined, undefined, undefined,
      doc.path,
      options?.chunkStrategy,
    );
    for (let seq = 0; seq < chunks.length; seq++) {
      const c = chunks[seq]!;
      allChunks.push({
        hash: doc.hash,
        title,
        text: c.text,
        seq,
        pos: c.pos,
        bytes: encoder.encode(c.text).length,
      });
    }
  }

  if (allChunks.length === 0) {
    return { docsProcessed: totalDocs, chunksEmbedded: 0, errors: 0, durationMs: Date.now() - startTime };
  }

  // Use total chunk bytes for progress denominator. Chunk bytes can exceed
  // doc bytes (each chunk repeats the title prefix), so doc-byte totals
  // produce percent>100 → renderProgressBar throws on negative repeat().
  const totalBytes = allChunks.reduce((sum, c) => sum + c.bytes, 0);

  // Lifecycle: bring up backends configured in ~/.config/qmd/embed-backends.yaml
  // that aren't already reachable. Adopt the ones that are. Cleanup on exit.
  const lifecycleConfig = loadBackendConfig();
  const lifecycle = new BackendLifecycle();
  const pool = getDefaultEmbedPool();
  if (lifecycleConfig) {
    const urls = pool.getConfiguredUrls();
    await lifecycle.prepare(urls, lifecycleConfig);
  }

  // Pool probe + dimension discovery + vec0 table init.
  await pool.embedBatch(["__probe__"]); // primes ensureActive() + dimensions
  const dims = pool.getDimensions();
  if (!dims) {
    throw new Error("Failed to detect embedding dimensions from pool probe");
  }
  store.ensureVecTable(dims);

  const embedModelUri = options?.model ?? "embeddinggemma";
  const texts = allChunks.map(c => formatDocForEmbedding(c.text, c.title, embedModelUri));
  const now = new Date().toISOString();
  const model = options?.model ?? embedModelUri;

  let chunksEmbedded = 0;
  let errors = 0;
  let bytesProcessed = 0;
  const totalChunks = allChunks.length;

  // SUB_BATCH balances two pressures: large = closer to /v1/embeddings
  // ceiling (per-request overhead amortized), small = better parallelism
  // (each server pulls work independently from the queue). On a tiny
  // corpus, large SUB_BATCH means one server gets the whole job — the
  // others sit idle. Auto-pick scales with corpus size and pool size,
  // capped at 128 to keep memory peaks bounded.
  const numServers = pool.getActiveUrls().length;
  const auto = Math.min(
    128,
    Math.max(16, Math.floor(totalChunks / Math.max(numServers, 1) / 4)),
  );
  const SUB_BATCH = process.env.QMD_EMBED_SUB_BATCH
    ? Number(process.env.QMD_EMBED_SUB_BATCH)
    : auto;

  await pool.embedStream(texts, (startIdx, results) => {
    // Wrap each callback's inserts in a single transaction.
    // better-sqlite3 transactions are synchronous and run faster than
    // per-row autocommit, which matters when the pool fires many small
    // sub-batches concurrently from different workers.
    const tx = db.transaction((slice: typeof results) => {
      for (let i = 0; i < slice.length; i++) {
        const chunkIdx = startIdx + i;
        const chunk = allChunks[chunkIdx]!;
        const result = slice[i];
        if (result) {
          insertEmbedding(db, chunk.hash, chunk.seq, chunk.pos, new Float32Array(result.embedding), model, now);
          chunksEmbedded++;
        } else {
          errors++;
        }
        bytesProcessed += chunk.bytes;
      }
    });
    tx(results);

    options?.onProgress?.({
      chunksEmbedded,
      totalChunks,
      bytesProcessed,
      totalBytes,
      errors,
    });
  }, SUB_BATCH);

  // Stop any backends we started (adopted ones left alone).
  await lifecycle.cleanup();

  return {
    docsProcessed: totalDocs,
    chunksEmbedded,
    errors,
    durationMs: Date.now() - startTime,
  };
}

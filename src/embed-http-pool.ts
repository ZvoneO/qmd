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

import type { EmbeddingResult, EmbedOptions } from "./llm.js";

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
export class HttpEmbedPool {
  private urls: string[];
  private model: string;
  private timeoutMs: number;
  private activeUrls: string[] = [];
  private probed = false;
  private stats: Map<string, ServerStats> = new Map();
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

  /** Probe a single URL. Returns probe time in ms, or -1 if unreachable. */
  private async probeOne(url: string): Promise<number> {
    const t0 = Date.now();
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15_000);
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
          return Date.now() - t0;
        }
      } finally {
        clearTimeout(timeout);
      }
    } catch {
      // unreachable
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
      const name = r.url.replace("http://", "").replace(":11434", "");
      return r.ok ? `${name} ${r.ms}ms` : `${name} ✗`;
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

  /** Send embed request to a specific URL. */
  private async fetchEmbed(baseUrl: string, input: string[]): Promise<number[][]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
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

  async embed(text: string, _options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
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
// TODO: re-wire into CLI behind QMD_EMBED_URLS
// =============================================================================
//
// To restore the 3× multi-server speedup, two paths:
//
// (A) Widen upstream's withLLMSessionForLlm to accept the LLM interface
//     instead of the concrete LlamaCpp class. Then HttpEmbedPool can
//     implement the LLM interface (stubbing rerank/generate/expandQuery)
//     and be injected via store.setLlm(). Touches src/llm.ts upstream code
//     — adds fork delta there. Smallest behavioral change.
//
// (B) Add a sibling generateEmbeddingsViaPool() in this file that owns
//     its own chunking + insertion loop, calling exported helpers from
//     store.ts (chunkDocumentByTokens, insertEmbedding, extractTitle).
//     Some store.ts internals are private (getPendingEmbeddingDocs,
//     resolveEmbedOptions, buildEmbeddingBatches) — would need to either
//     export them upstream (small PR) or reimplement them here.
//
// Recommend (A) — fewer total moving parts, one well-defined widening
// in upstream's signature that's easy to upstream as a PR.

// =============================================================================
// Embedding backend lifecycle — on-demand start/adopt/stop
// =============================================================================
//
// Reads ~/.config/qmd/embed-backends.yaml. For each pool URL:
//   - probes <url>+health_path. If 200 → "adopted" (leave it alone)
//   - else runs `start` over SSH (or locally), polls health up to ready_timeout_s
//   - on process exit / SIGINT / SIGTERM, runs `stop` ONLY for "started-by-us"
//
// Adoption is the safety net: a server that was already running stays running.
// Cold-start is the convenience: qmd embed brings up servers it needs and
// puts them down when done, so Ollama and llama-server don't fight over GPU.
//
// Free-form `start`/`stop` shell scripts give max flexibility. The lifecycle
// layer just shells them out — qmd doesn't try to be a process supervisor.

import { spawn } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parse as parseYaml } from "yaml";

export type BackendDef = {
  url: string;
  ssh?: string;              // e.g. "strives@gb10c". Omit for localhost.
  health_path?: string;      // default: "/health"
  start?: string;            // shell script run on cold-start
  stop?: string;             // shell script run on our-cleanup
  ready_timeout_s?: number;  // default: 30
};

export type BackendConfig = {
  backends: BackendDef[];
};

export type LifecycleState = "adopted" | "started-by-us" | "unavailable";

export type LifecycleRecord = {
  url: string;
  state: LifecycleState;
  def?: BackendDef;
};

const DEFAULT_CONFIG_PATH = join(homedir(), ".config", "qmd", "embed-backends.yaml");
const DEFAULT_HEALTH_PATH = "/health";
const DEFAULT_READY_TIMEOUT_S = 30;

export function loadBackendConfig(path?: string): BackendConfig | null {
  const file = path ?? process.env.QMD_EMBED_BACKENDS_CONFIG ?? DEFAULT_CONFIG_PATH;
  if (!existsSync(file)) return null;
  try {
    const raw = readFileSync(file, "utf8");
    const parsed = parseYaml(raw) as BackendConfig;
    if (!parsed || !Array.isArray(parsed.backends)) {
      throw new Error(`expected { backends: [...] } in ${file}`);
    }
    return parsed;
  } catch (err) {
    process.stderr.write(`[lifecycle] failed to load ${file}: ${(err as Error).message}\n`);
    return null;
  }
}

/** Run a shell command, optionally over SSH. Returns {code, stdout, stderr}. */
function runShell(cmd: string, ssh?: string, timeoutMs = 60_000): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const argv = ssh
      ? ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh, "bash -s"]
      : ["bash", "-s"];
    const proc = spawn(argv[0]!, argv.slice(1), { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => { stdout += d.toString(); });
    proc.stderr.on("data", (d) => { stderr += d.toString(); });
    const timer = setTimeout(() => proc.kill("SIGKILL"), timeoutMs);
    proc.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? -1, stdout, stderr });
    });
    proc.stdin.write(cmd);
    proc.stdin.end();
  });
}

async function probeHealth(url: string, healthPath: string, timeoutMs = 3_000): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}${healthPath}`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function waitHealthy(url: string, healthPath: string, timeoutS: number): Promise<boolean> {
  const deadline = Date.now() + timeoutS * 1000;
  while (Date.now() < deadline) {
    if (await probeHealth(url, healthPath)) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

export class BackendLifecycle {
  private records: Map<string, LifecycleRecord> = new Map();
  private exitHookRegistered = false;
  private cleanupPromise: Promise<void> | null = null;

  /**
   * Prepare each URL: probe → adopt OR cold-start.
   * Returns the subset of URLs that are now reachable.
   */
  async prepare(urls: string[], config: BackendConfig | null): Promise<string[]> {
    const defsByUrl = new Map<string, BackendDef>();
    if (config) {
      for (const d of config.backends) defsByUrl.set(d.url, d);
    }

    const ready: string[] = [];
    for (const url of urls) {
      const def = defsByUrl.get(url);
      const healthPath = def?.health_path ?? DEFAULT_HEALTH_PATH;

      // Try adoption first.
      if (await probeHealth(url, healthPath)) {
        this.records.set(url, { url, state: "adopted", def });
        ready.push(url);
        continue;
      }

      // Cold-start needs a config entry with `start`.
      if (!def || !def.start) {
        process.stderr.write(`[lifecycle] ${url}: not reachable and no start script configured\n`);
        this.records.set(url, { url, state: "unavailable", def });
        continue;
      }

      process.stderr.write(`[lifecycle] ${url}: cold-starting via ${def.ssh ?? "local"}\n`);
      const res = await runShell(def.start, def.ssh);
      if (res.code !== 0) {
        process.stderr.write(`[lifecycle] ${url}: start script exit ${res.code}: ${res.stderr.slice(0, 200)}\n`);
        this.records.set(url, { url, state: "unavailable", def });
        continue;
      }

      const okWait = await waitHealthy(url, healthPath, def.ready_timeout_s ?? DEFAULT_READY_TIMEOUT_S);
      if (!okWait) {
        process.stderr.write(`[lifecycle] ${url}: health-check timeout after start\n`);
        this.records.set(url, { url, state: "unavailable", def });
        continue;
      }

      process.stderr.write(`[lifecycle] ${url}: started, will stop on exit\n`);
      this.records.set(url, { url, state: "started-by-us", def });
      ready.push(url);
    }

    this.registerExitHooks();
    return ready;
  }

  /** Stop ONLY servers we started. Idempotent. */
  async cleanup(): Promise<void> {
    if (this.cleanupPromise) return this.cleanupPromise;
    this.cleanupPromise = this.doCleanup();
    return this.cleanupPromise;
  }

  private async doCleanup(): Promise<void> {
    for (const rec of this.records.values()) {
      if (rec.state !== "started-by-us") continue;
      if (!rec.def?.stop) {
        process.stderr.write(`[lifecycle] ${rec.url}: no stop script — leaving running\n`);
        continue;
      }
      process.stderr.write(`[lifecycle] ${rec.url}: stopping (we started it)\n`);
      const res = await runShell(rec.def.stop, rec.def.ssh);
      if (res.code !== 0) {
        process.stderr.write(`[lifecycle] ${rec.url}: stop script exit ${res.code}: ${res.stderr.slice(0, 200)}\n`);
      }
    }
  }

  private registerExitHooks(): void {
    if (this.exitHookRegistered) return;
    this.exitHookRegistered = true;

    const handler = (signal: NodeJS.Signals | "exit") => {
      // process.on('exit') is synchronous-only; we can't await there. But
      // SIGINT/SIGTERM handlers are async-OK — we'll do the work there and
      // then re-raise. If the user does `process.exit()` from elsewhere,
      // we'll miss cleanup, but the CLI calls cleanup() explicitly.
      if (signal === "exit") return;
      this.cleanup().finally(() => {
        process.exit(signal === "SIGINT" ? 130 : 143);
      });
    };
    process.on("SIGINT", () => handler("SIGINT"));
    process.on("SIGTERM", () => handler("SIGTERM"));
  }

  getRecords(): LifecycleRecord[] {
    return Array.from(this.records.values());
  }
}

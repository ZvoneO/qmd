import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url);
const pkg = JSON.parse(readFileSync(new URL("package.json", root), "utf8"));

describe("package grammar distribution", () => {
  test("installs AST grammar wasm packages as required runtime dependencies", () => {
    for (const dep of ["tree-sitter-typescript", "tree-sitter-python", "tree-sitter-go", "tree-sitter-rust"]) {
      expect(pkg.dependencies, `${dep} should be a required dependency`).toHaveProperty(dep);
      expect(pkg.optionalDependencies ?? {}, `${dep} should not be optional`).not.toHaveProperty(dep);
    }
  });

  test("documents a packaging smoke check for grammar wasm availability", () => {
    expect(pkg.scripts, "package.json scripts").toHaveProperty("smoke:package-grammars");
    expect(String(pkg.scripts["smoke:package-grammars"])).toContain("check-package-grammars");

    expect(pkg.files, "published package files").toContain("scripts/check-package-grammars.mjs");

    const scriptPath = join(root.pathname, "scripts", "check-package-grammars.mjs");
    const script = readFileSync(scriptPath, "utf8");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-typescript.wasm");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-tsx.wasm");
  });
});

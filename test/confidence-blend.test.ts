/**
 * Confidence blending + deterministic exact-match boost.
 *
 * Regression coverage for RRF rank-1 saturation: a single weak candidate used
 * to inherit the rank-1 positional floor (~0.75) and display ~88% even when the
 * reranker was uncertain (~0.50). blendConfidenceScore scales the positional
 * weight by evidence (reranker confidence OR cross-arm corroboration) so
 * garbage drops toward the reranker's neutral point while genuine hits and
 * multi-arm rescues stay high. extractIdentifierTerms/textMatchesIdentifier
 * drive the exact-identifier boost.
 */

import { describe, test, expect } from "vitest";
import {
  blendConfidenceScore,
  extractIdentifierTerms,
  textMatchesIdentifier,
  positionWeightForRank,
} from "../src/store";

describe("blendConfidenceScore — RRF rank-1 saturation fix", () => {
  test("lone uncertain candidate no longer floors near 0.88", () => {
    // Reproduces "kubernetes ingress annotations": rank 1, one arm, pool of 1,
    // reranker at its "no signal" floor. Old formula: 0.75*1 + 0.25*0.5 = 0.875.
    const { blendedScore } = blendConfidenceScore({
      rerankScore: 0.5001,
      rrfRank: 1,
      armCount: 1,
      poolSize: 1,
    });
    expect(blendedScore).toBeLessThan(0.5); // damped below neutral
  });

  test("genuine rank-1 hit with a confident reranker stays high (>0.9)", () => {
    const { blendedScore } = blendConfidenceScore({
      rerankScore: 0.7281,
      rrfRank: 1,
      armCount: 1,
      poolSize: 22,
    });
    expect(blendedScore).toBeGreaterThan(0.9);
    // Unchanged from the legacy formula for confident hits: 0.75*1 + 0.25*rerank.
    expect(blendedScore).toBeCloseTo(0.75 + 0.25 * 0.7281, 6);
  });

  test("multi-arm agreement preserves the positional rescue despite low rerank", () => {
    // Two arms independently surfaced this doc → position is trusted even though
    // the reranker underscores it. Should keep the full rank-1 protection.
    const { blendedScore, positionWeight } = blendConfidenceScore({
      rerankScore: 0.5,
      rrfRank: 1,
      armCount: 2,
      poolSize: 10,
    });
    expect(positionWeight).toBeCloseTo(0.75, 6);
    expect(blendedScore).toBeCloseTo(0.75 * 1 + 0.25 * 0.5, 6); // 0.875
  });

  test("isolation damp only fires for a thin pool, not a deep one", () => {
    const deep = blendConfidenceScore({
      rerankScore: 0.5, rrfRank: 1, armCount: 1, poolSize: 10,
    }).blendedScore;
    const thin = blendConfidenceScore({
      rerankScore: 0.5, rrfRank: 1, armCount: 1, poolSize: 1,
    }).blendedScore;
    expect(deep).toBeCloseTo(0.5, 6);   // pure reranker, no damp
    expect(thin).toBeLessThan(deep);    // lone candidate damped further
  });

  test("monotonic in rerankScore (fixed rank/arm/pool)", () => {
    const scores = [0.5, 0.6, 0.7, 0.8, 0.9].map(
      s => blendConfidenceScore({ rerankScore: s, rrfRank: 1, armCount: 1, poolSize: 20 }).blendedScore,
    );
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]!).toBeGreaterThan(scores[i - 1]!);
    }
  });

  test("garbage (50%) is clearly separated from a genuine hit (>90%)", () => {
    const garbage = blendConfidenceScore({
      rerankScore: 0.5001, rrfRank: 1, armCount: 1, poolSize: 1,
    }).blendedScore;
    const hit = blendConfidenceScore({
      rerankScore: 0.73, rrfRank: 1, armCount: 1, poolSize: 20,
    }).blendedScore;
    expect(hit - garbage).toBeGreaterThan(0.4);
  });

  test("positionWeightForRank tiers unchanged", () => {
    expect(positionWeightForRank(1)).toBe(0.75);
    expect(positionWeightForRank(3)).toBe(0.75);
    expect(positionWeightForRank(4)).toBe(0.60);
    expect(positionWeightForRank(10)).toBe(0.60);
    expect(positionWeightForRank(11)).toBe(0.40);
  });
});

describe("extractIdentifierTerms — conservative identifier detection", () => {
  test("recognizes snake_case, camelCase, and dotted paths", () => {
    expect(extractIdentifierTerms("preferred_consultant_ids")).toEqual(["preferred_consultant_ids"]);
    expect(extractIdentifierTerms("DecimalInput")).toEqual(["DecimalInput"]);
    expect(extractIdentifierTerms("app.module.thing")).toEqual(["app.module.thing"]);
  });

  test("ignores ordinary words and single Capitalized words", () => {
    expect(extractIdentifierTerms("where does consultant availability live")).toEqual([]);
    expect(extractIdentifierTerms("Consultant availability")).toEqual([]);
    expect(extractIdentifierTerms("csrf token validation")).toEqual([]);
  });

  test("picks identifiers out of a mixed natural-language query", () => {
    expect(extractIdentifierTerms("where is preferred_consultant_ids defined"))
      .toEqual(["preferred_consultant_ids"]);
  });

  test("dedupes case-insensitively", () => {
    expect(extractIdentifierTerms("DecimalInput decimalinput_field"))
      .toEqual(["DecimalInput", "decimalinput_field"]);
  });
});

describe("textMatchesIdentifier — whole-token exact match", () => {
  const idents = ["preferred_consultant_ids"];

  test("matches when the identifier is a whole token in the title/path", () => {
    expect(textMatchesIdentifier("Vocabulary: preferred_consultant_ids (domain)", idents)).toBe(true);
    expect(textMatchesIdentifier("app/schemas/preferred_consultant_ids.py", idents)).toBe(true);
  });

  test("does not match a partial/substring token", () => {
    expect(textMatchesIdentifier("src/lib/ids.ts", idents)).toBe(false);
    // 'consultant' must not match inside 'preferred_consultant_ids'
    expect(textMatchesIdentifier("preferred_consultant_ids", ["consultant"])).toBe(false);
  });

  test("camelCase identifiers match around non-identifier boundaries", () => {
    expect(textMatchesIdentifier("src/components/ui/DecimalInput.tsx", ["DecimalInput"])).toBe(true);
    expect(textMatchesIdentifier("MyDecimalInputWrapper", ["DecimalInput"])).toBe(false);
  });

  test("empty inputs are safe", () => {
    expect(textMatchesIdentifier(undefined, idents)).toBe(false);
    expect(textMatchesIdentifier("anything", [])).toBe(false);
  });
});

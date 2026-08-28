// 코사인 유사도(의미 검색)와 BM25(정확한 표기 검색)를 함께 수행하는 하이브리드 검색.
import type { FabotChunk, SearchHit } from "./types";

export const WEAK_EVIDENCE_THRESHOLD = 0.55;

function cosine(a: number[], b: number[]): number {
  // 두 벡터 모두 이미 L2 정규화되어 있으므로 내적이 곧 코사인 유사도다.
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

// 간단한 BM25 구현 (k1=1.5, b=0.75)
function bm25Scores(query: string, chunks: FabotChunk[]): number[] {
  const k1 = 1.5;
  const b = 0.75;
  const docsTokens = chunks.map((c) => tokenize(c.text));
  const avgLen = docsTokens.reduce((s, t) => s + t.length, 0) / (docsTokens.length || 1);
  const queryTokens = Array.from(new Set(tokenize(query)));

  const df = new Map<string, number>();
  for (const term of queryTokens) {
    let count = 0;
    for (const doc of docsTokens) if (doc.includes(term)) count++;
    df.set(term, count);
  }
  const N = chunks.length;

  return docsTokens.map((doc) => {
    let score = 0;
    const docLen = doc.length || 1;
    for (const term of queryTokens) {
      const freq = doc.filter((t) => t === term).length;
      if (freq === 0) continue;
      const n = df.get(term) ?? 0;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (docLen / avgLen)));
    }
    return score;
  });
}

// 코사인 상위 10개를 먼저 고르고, 아직 없는 청크 중 BM25 상위 5개를 추가한다.
export function hybridSearch(queryVector: number[], query: string, chunks: FabotChunk[]): SearchHit[] {
  const cosineRanked = chunks
    .map((chunk) => ({ chunk, score: cosine(queryVector, chunk.vector), method: "vector" as const }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  const picked = new Set(cosineRanked.map((h) => h.chunk.id));
  const bm25Raw = bm25Scores(query, chunks);
  const maxBm25 = Math.max(...bm25Raw, 1e-9);

  const bm25Ranked = chunks
    .map((chunk, i) => ({ chunk, score: bm25Raw[i] / maxBm25, method: "bm25" as const }))
    .filter((h) => !picked.has(h.chunk.id) && h.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  return [...cosineRanked, ...bm25Ranked];
}

export function topSimilarity(hits: SearchHit[]): number {
  const vectorHits = hits.filter((h) => h.method === "vector");
  return vectorHits.length ? Math.max(...vectorHits.map((h) => h.score)) : 0;
}

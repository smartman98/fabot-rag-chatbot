import type { FabotChunk } from "./types";

let cache: FabotChunk[] | null = null;

export async function loadDocs(): Promise<FabotChunk[]> {
  if (cache) return cache;
  const res = await fetch(`${import.meta.env.BASE_URL}fabot-docs.json`);
  if (!res.ok) throw new Error("문서 벡터스토어 로드 실패");
  cache = await res.json();
  return cache!;
}

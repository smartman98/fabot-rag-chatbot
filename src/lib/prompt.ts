import type { SearchHit } from "./types";
import { WEAK_EVIDENCE_THRESHOLD, topSimilarity } from "./search";

function nowKST(): string {
  return new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    dateStyle: "full",
    timeStyle: "short",
  }).format(new Date());
}

export function buildPrompt(question: string, hits: SearchHit[]): { prompt: string; weakEvidence: boolean } {
  const weakEvidence = topSimilarity(hits) < WEAK_EVIDENCE_THRESHOLD;
  const evidenceBlock = hits
    .map((h) => `[${h.chunk.id} | ${h.chunk.section}] ${h.chunk.text}`)
    .join("\n");

  const weakNotice = weakEvidence
    ? "\n주의: 검색된 조각의 유사도가 낮습니다. 질문과 완전히 맞는 근거가 아닐 수 있으니, 근거에 있는 내용만 짧게 답하고 자료에 없는 부분은 없다고 말합니다."
    : "";

  const prompt = `다음 자료는 FABOT(F&G 기반 투자 자동화) 프로젝트의 공개 GitHub 문서에서 뽑은 조각입니다.${weakNotice}
근거가 된 조각의 [ID]를 답 안에서 표시합니다.
자료에 없는 미래 예측이나 투자 결정("사세요/파세요")은 하지 않고, 규칙과 현재 상태만 안내합니다.
현재 시각은 ${nowKST()}입니다. '지금', '오늘' 같은 상대 표현은 이 시각을 기준으로 해석합니다.

[자료]
${evidenceBlock}

[질문]
${question}`;

  return { prompt, weakEvidence };
}

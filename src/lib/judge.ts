import type { JudgeResult, SearchHit } from "./types";
import { judgeChat } from "./ollama";

function buildJudgePrompt(question: string, hits: SearchHit[], answer: string): string {
  const evidence = hits.map((h) => `[${h.chunk.id}] ${h.chunk.text}`).join("\n");
  return `당신은 RAG 챗봇 답변의 평가자입니다. 아래 [질문], [근거자료], [답변]을 읽고 다음 기준으로 JSON만 출력합니다.
grounded: 답변 내용이 근거자료에서 나왔는가 (true/false)
noHalluc: 근거에 없는 사실을 지어내지 않았는가 (true/false)
cited: 답변 안에 근거 조각의 [ID] 표시가 있는가 (true/false)
refusal: 질문이 자료 범위 밖이거나(날씨·예측 등) 챗봇 권한 밖(주문 실행, 타인 정보 등)이라 답을 하지 않고 거절/제한 안내를 한 경우 true. "제공된 정보가 없습니다", "저는 ~할 수 없습니다", "관련 규정이 없습니다"처럼 이유를 설명하며 거절한 경우도 포함한다. 실제로 자료 내용을 근거로 답한 경우만 false.
score: 0-100 정수 (grounded·noHalluc·cited 반영)
comment: 한두 문장 평어 (한국어)
출력 형식: {"grounded":bool,"noHalluc":bool,"cited":bool,"refusal":bool,"score":int,"comment":"..."} — JSON 외 텍스트 금지.
[질문] ${question}
[근거자료] ${evidence}
[답변] ${answer}`;
}

export async function judgeAnswer(question: string, hits: SearchHit[], answer: string): Promise<JudgeResult> {
  const raw = await judgeChat(buildJudgePrompt(question, hits, answer));
  const parsed = JSON.parse(raw);

  // 모델이 가끔 5점 만점처럼 score를 반환하는 경우를 100점 만점으로 환산한다.
  let score = Number(parsed.score) || 0;
  if (score <= 5) score = Math.round((score / 5) * 100);

  return {
    grounded: Boolean(parsed.grounded),
    noHalluc: Boolean(parsed.noHalluc),
    cited: Boolean(parsed.cited),
    refusal: Boolean(parsed.refusal),
    score,
    comment: String(parsed.comment ?? ""),
  };
}

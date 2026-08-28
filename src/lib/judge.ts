import type { JudgeResult, SearchHit } from "./types";
import { judgeChat } from "./ollama";

function buildJudgePrompt(question: string, hits: SearchHit[], answer: string): string {
  const evidence = hits.map((h) => `[${h.chunk.id}] ${h.chunk.text}`).join("\n");
  return `당신은 RAG 챗봇 답변의 평가자입니다. 아래 [질문], [근거자료], [답변]을 읽고 다음 기준으로 JSON만 출력합니다.
grounded: 답변 내용이 근거자료에서 나왔는가 (true/false)
noHalluc: 근거에 없는 사실을 지어내지 않았는가 (true/false)
cited: 답변 안에 근거 조각의 [ID] 표시가 있는가 (true/false)
refusal: 근거에 답이 없어서 '없다'고 답한 경우 true, 그 외 false
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

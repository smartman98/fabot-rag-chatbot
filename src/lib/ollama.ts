// 로컬 Ollama(qwen3.5:2b)와 통신한다. 정적 배포 페이지도 사용자 브라우저가
// 직접 localhost:11434를 호출하므로, Ollama 쪽에 OLLAMA_ORIGINS 허용이 필요하다.
const OLLAMA_BASE = "http://localhost:11434";
export const OLLAMA_MODEL = "qwen3.5:2b";

export async function checkOllamaStatus(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function streamChat(
  prompt: string,
  onToken: (partial: string) => void,
  signal: AbortSignal
): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    signal,
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      stream: true,
      think: false,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Ollama 응답 실패: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const chunk = JSON.parse(line);
        const piece = chunk?.message?.content ?? "";
        if (piece) {
          full += piece;
          onToken(full);
        }
      } catch {
        // 파싱 실패한 조각은 건너뛴다 (스트림 경계에 걸린 불완전한 줄일 수 있음)
      }
    }
  }
  return full;
}

export async function judgeChat(prompt: string): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      stream: false,
      think: false,
      format: "json",
      options: { temperature: 0 },
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`judge 호출 실패: ${res.status}`);
  const data = await res.json();
  return data?.message?.content ?? "";
}

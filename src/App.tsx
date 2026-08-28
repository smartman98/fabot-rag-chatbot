import { useEffect, useRef, useState } from "react";
import "./App.css";
import { loadDocs } from "./lib/docs";
import { embedQuery, type EmbedProgress } from "./lib/embed";
import { hybridSearch } from "./lib/search";
import { buildPrompt } from "./lib/prompt";
import { checkOllamaStatus, streamChat, OLLAMA_MODEL } from "./lib/ollama";
import { judgeAnswer } from "./lib/judge";
import type { ChatTurn, FabotChunk } from "./lib/types";

type EmbedStatus = EmbedProgress["status"] | "idle" | "error";

export default function App() {
  const [docs, setDocs] = useState<FabotChunk[] | null>(null);
  const [ollamaUp, setOllamaUp] = useState<boolean | null>(null);
  const [embedStatus, setEmbedStatus] = useState<EmbedStatus>("idle");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    loadDocs().then(setDocs).catch(() => setDocs([]));
    refreshOllama();
  }, []);

  async function refreshOllama() {
    setOllamaUp(await checkOllamaStatus());
  }

  async function handleAsk() {
    const q = question.trim();
    if (!q || busy || !docs) return;
    setQuestion("");
    setBusy(true);

    const turn: ChatTurn = {
      role: "user",
      question: q,
      answer: "",
      hits: [],
      weakEvidence: false,
      judge: null,
      judgeError: false,
      feedback: null,
    };
    setTurns((prev) => [...prev, turn]);
    const idx = turns.length;

    try {
      setEmbedStatus("loading-tokenizer");
      const queryVector = await embedQuery(q);
      setEmbedStatus("ready");

      const hits = hybridSearch(queryVector, q, docs);
      const { prompt, weakEvidence } = buildPrompt(q, hits);

      setTurns((prev) => {
        const next = [...prev];
        next[idx] = { ...next[idx], hits, weakEvidence };
        return next;
      });

      const up = await checkOllamaStatus();
      setOllamaUp(up);
      if (!up) {
        setTurns((prev) => {
          const next = [...prev];
          next[idx] = { ...next[idx], answer: "(Ollama가 꺼져 있어 답변을 만들 수 없습니다. 실행 후 재시도해주세요.)" };
          return next;
        });
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;
      const answer = await streamChat(prompt, (partial) => {
        setTurns((prev) => {
          const next = [...prev];
          next[idx] = { ...next[idx], answer: partial };
          return next;
        });
      }, controller.signal);

      try {
        const judge = await judgeAnswer(q, hits, answer);
        setTurns((prev) => {
          const next = [...prev];
          next[idx] = { ...next[idx], judge };
          return next;
        });
      } catch {
        setTurns((prev) => {
          const next = [...prev];
          next[idx] = { ...next[idx], judgeError: true };
          return next;
        });
      }
    } catch (err) {
      setTurns((prev) => {
        const next = [...prev];
        next[idx] = { ...next[idx], answer: `(오류: ${(err as Error).message})` };
        return next;
      });
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  function setFeedback(idx: number, fb: "up" | "down") {
    setTurns((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], feedback: next[idx].feedback === fb ? null : fb };
      return next;
    });
  }

  return (
    <div className="app">
      <header className="intro">
        <h1>FABOT 전략 안내 챗봇</h1>
        <p>
          FABOT(F&amp;G 기반 투자 자동화) 프로젝트의 공개 GitHub 문서를 근거로 매매 규칙을 안내합니다.
          실제 주문은 실행하지 않으며, 조회·설명만 담당합니다.
        </p>
        <p className="conditions">
          사용 조건: 이 컴퓨터에서 <code>Ollama</code>가 실행 중이어야 하고 <code>{OLLAMA_MODEL}</code> 모델이 준비돼 있어야 합니다.
          첫 방문 시 임베딩 모델(약 200MB)을 내려받습니다. Chrome/Edge 권장, Safari는 불안정할 수 있습니다.
        </p>
        <div className="status-row">
          <span className={`badge ${ollamaUp ? "ok" : "warn"}`}>
            Ollama: {ollamaUp === null ? "확인 중" : ollamaUp ? "연결됨" : "꺼짐"}
          </span>
          {!ollamaUp && (
            <button onClick={refreshOllama} className="retry">
              재시도
            </button>
          )}
          <span className={`badge ${docs ? "ok" : "warn"}`}>
            문서: {docs ? `${docs.length}개 로드됨` : "로딩 중"}
          </span>
          <span className="badge">임베딩 상태: {embedStatus}</span>
        </div>
      </header>

      <main className="chat">
        {turns.map((t, i) => (
          <div className="turn" key={i}>
            <div className="question">Q. {t.question}</div>
            <div className="answer">{t.answer || "..."}</div>

            {t.weakEvidence && <div className="weak-warning">⚠ 약한 근거 — 질문과 완전히 맞는 자료가 아닐 수 있습니다.</div>}

            {t.hits.length > 0 && (
              <div className="chips">
                {t.hits.map((h) => (
                  <a
                    key={h.chunk.id + h.method}
                    className={`chip ${h.method}`}
                    href={h.chunk.url}
                    target="_blank"
                    rel="noreferrer"
                    title={h.chunk.section}
                  >
                    {h.chunk.id} · {h.method} · {h.score.toFixed(2)}
                  </a>
                ))}
              </div>
            )}

            {t.judge && (
              <div className="judge-badge">
                <span className={t.judge.grounded ? "ok" : "warn"}>grounded:{String(t.judge.grounded)}</span>
                <span className={t.judge.noHalluc ? "ok" : "warn"}>noHalluc:{String(t.judge.noHalluc)}</span>
                <span className={t.judge.cited ? "ok" : "warn"}>cited:{String(t.judge.cited)}</span>
                <span className={t.judge.refusal ? "info" : ""}>refusal:{String(t.judge.refusal)}</span>
                <span className="score">score:{t.judge.score}</span>
                <span className="comment">{t.judge.comment}</span>
              </div>
            )}
            {t.judgeError && <div className="judge-error">judgeError — 답변은 유지됩니다</div>}

            <div className="feedback">
              <button
                className={t.feedback === "up" ? "active" : ""}
                onClick={() => setFeedback(i, "up")}
              >
                👍
              </button>
              <button
                className={t.feedback === "down" ? "active" : ""}
                onClick={() => setFeedback(i, "down")}
              >
                👎
              </button>
            </div>
          </div>
        ))}
      </main>

      <footer className="composer">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="예: TQQQ는 언제 팔아야 해?"
          disabled={busy}
        />
        {busy ? (
          <button onClick={handleStop}>중지</button>
        ) : (
          <button onClick={handleAsk} disabled={!docs}>
            질문
          </button>
        )}
      </footer>
    </div>
  );
}

# FABOT 전략 안내 챗봇 (브라우저 RAG)

FABOT(F&G 기반 투자 자동화) 프로젝트의 공개 GitHub 문서를 근거로 매매 규칙을 안내하는
서버 없는 RAG 챗봇입니다. 브라우저에서 임베딩과 하이브리드 검색을 수행하고, 로컬
Ollama(`qwen3.5:2b`)가 답변을 스트리밍으로 생성하며, 답변마다 LLM-as-a-Judge 판정
배지가 붙습니다. 서버는 정적 파일만 서빙하고, 실제 답변 생성은 이 페이지를 연
사용자 컴퓨터의 Ollama가 담당합니다.

## 사용 조건

- 이 컴퓨터에서 [Ollama](https://ollama.com)가 실행 중이어야 하고 `qwen3.5:2b` 모델이
  준비돼 있어야 합니다 (`ollama pull qwen3.5:2b`).
- 첫 방문 시 임베딩 모델(`onnx-community/embeddinggemma-300m-ONNX`, 약 200MB)을
  내려받습니다.
- GitHub Pages 배포 주소에서 쓰려면 Ollama에 아래 CORS 허용이 필요합니다 (터미널
  재시작 필요):

  ```bash
  # macOS
  launchctl setenv OLLAMA_ORIGINS "https://smartman98.github.io"
  # Windows (PowerShell)
  setx OLLAMA_ORIGINS "https://smartman98.github.io"
  ```

- Chrome/Edge 권장, Safari는 Ollama 연결과 임베딩 WASM이 불안정할 수 있습니다.

## 로컬 실행

```bash
npm install
node scripts/embed-docs.mjs   # public/fabot-docs.json 벡터스토어 생성 (자료를 바꿨을 때만 재실행)
npm run dev
```

## 설계 결정 요약

- **자료 범위**: `smartman98/my-service-agent`(design-packet.md, README.md,
  tool-definition.md, RAG 코드)와 `smartman98/bis1`(전략 개요 문서)의 공개 GitHub
  문서 12개 청크. 실시간 F&G 점수나 실제 계좌 잔고는 다루지 않습니다 — 이건 별도
  서비스(fabot-trade-journal)의 역할이고, 이 챗봇은 "규칙이 무엇인지"만 정적 문서로
  답합니다.
- **하이브리드 검색**: 코사인 유사도 상위 10개 + 미중복 BM25 상위 5개. 최고 유사도가
  0.55 미만이면 결과를 버리지 않고 프롬프트를 보수화(약한 근거 경고)합니다.
- **judge를 같은 로컬 2B 모델로 둔 이유**: API 키 없이도 학생이 근거성·환각·인용을
  관찰할 수 있게 하기 위함입니다. 단, 답변 모델과 평가 모델이 같으므로 독립 심사가
  아니라 거친 교육용 신호로 읽어야 합니다.

## 실험 로그

고정 질문 세트(FABOT 전략 질문 5개, 도메인 밖 질문 2개)로 2026-08-28에 1차 실행한 기록.

| # | 질문 | 유형 | top-1 근거 | 판정 |
|---|---|---|---|---|
| 1 | "TQQQ는 언제 팔아야 해?" | 정상 | FB-003 (score 0.74) | grounded:true, cited:true, score:90 |
| 2 | "내일 날씨 어때?" | 도메인 밖 | FB-011 (score 0.56, 약한 근거 아님) | grounded:false, noHalluc:true, **refusal:false**, score:20 |

### 실측으로 발견한 한계

1. **judge가 정당한 거부를 refusal:true로 못 잡음** — 질문 2에서 모델은 "저는 투자
   도구만 다룹니다"라고 날씨를 정확히 거절했지만, judge는 `refusal:false`로 판정하고
   score를 20점까지 깎았다. judge 프롬프트의 refusal 정의("근거에 답이 없어서 '없다'고
   답한 경우")가 "도메인 밖이라 답하지 않음"이라는 이번 케이스와 문구상 안 맞아서로
   보인다. 다음 개선 과제로 judge 프롬프트에 이 경우도 refusal로 명시하는 문구 추가가
   필요하다.
2. **소형 로컬 모델의 한중일 문자 혼입** — `qwen3.5:2b`가 한국어 답변 중간에 이따금
   중국어 표현("天气预报", "명以上")을 섞어 쓰는 현상이 관찰됐다. 내용 자체는 정확했지만
   가독성에 영향을 준다.
3. **[ID] 인용이 규칙을 안 지킴** — 프롬프트에 "근거가 된 조각의 [ID]를 답 안에서
   표시"라고 명시했지만, 질문 1의 답변에는 실제로 `[FB-003]`이 등장하지 않았는데도
   judge가 `cited:true`로 판정했다. judge 자체도 완벽하지 않다는 근거다.

이 세 가지는 코드 버그가 아니라 실제로 관찰된 로컬 소형 모델·judge의 한계이며, 다음
실험(judge 프롬프트 개선, 더 큰 모델로 교체)의 출발점으로 남긴다.

## 하지 않은 것

- 실제 매수·매도 주문 실행
- 실시간 F&G 지수·계좌 잔고 조회 (fabot-trade-journal의 역할)
- 로그인, 다국어, 모바일 최적화

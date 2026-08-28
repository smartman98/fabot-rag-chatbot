export type FabotChunk = {
  id: string;
  text: string;
  url: string;
  section: string;
  vector: number[];
};

export type SearchHit = {
  chunk: FabotChunk;
  score: number;
  method: "vector" | "bm25";
};

export type JudgeResult = {
  grounded: boolean;
  noHalluc: boolean;
  cited: boolean;
  refusal: boolean;
  score: number;
  comment: string;
};

export type ChatTurn = {
  role: "user" | "assistant";
  question: string;
  answer: string;
  hits: SearchHit[];
  weakEvidence: boolean;
  judge: JudgeResult | null;
  judgeError: boolean;
  feedback: "up" | "down" | null;
};

// 브라우저에서 질문을 768차원 벡터로 바꾼다.
// 기본 pipeline()의 q4/q8 경로는 WASM ONNX Runtime이 GatherBlockQuantized 연산을
// 지원하지 않아 실패하므로, 토크나이저와 추론 세션을 분리하고
// model_no_gather_q4.onnx를 직접 불러오는 방식으로 우회한다.
import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";

const MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX";
const MODEL_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx`;
const MODEL_URL = `${MODEL_BASE}/model_no_gather_q4.onnx`;
const MODEL_DATA_URL = `${MODEL_BASE}/model_no_gather_q4.onnx_data`;

let tokenizerPromise: ReturnType<typeof AutoTokenizer.from_pretrained> | null = null;
let sessionPromise: Promise<ort.InferenceSession> | null = null;

export type EmbedProgress = { status: "loading-tokenizer" | "loading-model" | "ready" };

export async function loadEmbedder(onProgress?: (p: EmbedProgress) => void) {
  onProgress?.({ status: "loading-tokenizer" });
  if (!tokenizerPromise) {
    tokenizerPromise = AutoTokenizer.from_pretrained(MODEL_ID);
  }
  const tokenizer = await tokenizerPromise;

  onProgress?.({ status: "loading-model" });
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const [modelBuf, dataBuf] = await Promise.all([
        fetch(MODEL_URL).then((r) => r.arrayBuffer()),
        fetch(MODEL_DATA_URL).then((r) => r.arrayBuffer()),
      ]);
      return ort.InferenceSession.create(new Uint8Array(modelBuf), {
        executionProviders: ["wasm"],
        externalData: [{ path: "model_no_gather_q4.onnx_data", data: new Uint8Array(dataBuf) }],
      });
    })();
  }
  const session = await sessionPromise;

  onProgress?.({ status: "ready" });
  return { tokenizer, session };
}

// attention_mask가 가리키는 실제 토큰만 평균낸 뒤 L2 정규화한다.
function meanPoolAndNormalize(hidden: Float32Array, dims: readonly number[], mask: unknown[]): number[] {
  const [, seqLen, hiddenSize] = dims;
  const pooled = new Float64Array(hiddenSize);
  let count = 0;
  for (let t = 0; t < seqLen; t++) {
    const m = Number(mask[t]);
    if (m === 0) continue;
    count += 1;
    for (let h = 0; h < hiddenSize; h++) {
      pooled[h] += hidden[t * hiddenSize + h];
    }
  }
  const denom = count || 1; // 0으로 나누는 경우 방어
  for (let h = 0; h < hiddenSize; h++) pooled[h] /= denom;

  let norm = 0;
  for (let h = 0; h < hiddenSize; h++) norm += pooled[h] * pooled[h];
  norm = Math.sqrt(norm) || 1;
  return Array.from(pooled, (v) => v / norm);
}

export async function embedQuery(text: string): Promise<number[]> {
  const { tokenizer, session } = await loadEmbedder();
  const encoded = tokenizer(text, { padding: true, truncation: true });

  const inputIds = encoded.input_ids;
  const attentionMask = encoded.attention_mask;
  const seqLen = inputIds.dims[1];

  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(Array.from(inputIds.data as any, (v: any) => BigInt(v))), [1, seqLen]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(Array.from(attentionMask.data as any, (v: any) => BigInt(v))), [1, seqLen]),
  };

  const output = await session.run(feeds);
  const lastHidden = output["last_hidden_state"] ?? output[Object.keys(output)[0]];
  return meanPoolAndNormalize(
    lastHidden.data as Float32Array,
    lastHidden.dims,
    Array.from(attentionMask.data as any)
  );
}

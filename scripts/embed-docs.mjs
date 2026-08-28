// FABOT 공개 문서 청크를 읽어 브라우저 RAG용 벡터스토어를 만든다.
// onnx-community/embeddinggemma-300m-ONNX 사용, dtype q4, mean pooling, L2 정규화, 768차원.
import { pipeline } from "@huggingface/transformers";
import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const chunks = JSON.parse(readFileSync(join(__dirname, "fabot-chunks.json"), "utf-8"));

const extractor = await pipeline(
  "feature-extraction",
  "onnx-community/embeddinggemma-300m-ONNX",
  { dtype: "q4" }
);

const withVectors = [];
for (const chunk of chunks) {
  const output = await extractor(chunk.text, { pooling: "mean", normalize: true });
  const vector = Array.from(output.data);
  if (vector.length !== 768) {
    throw new Error(`${chunk.id}: expected 768-dim vector, got ${vector.length}`);
  }
  withVectors.push({ ...chunk, vector });
  console.log(`embedded ${chunk.id} (${vector.length} dims)`);
}

const outPath = join(__dirname, "..", "public", "fabot-docs.json");
writeFileSync(outPath, JSON.stringify(withVectors));
console.log(`wrote ${withVectors.length} chunks to ${outPath}`);

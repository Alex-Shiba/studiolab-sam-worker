// Run the actual StudioLab ServerSelector, capturing its encoder inputs and outputs.
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const [studioArg, modelArg, outputArg, ...frames] = process.argv.slice(2);
if (!outputArg || frames.length !== 2) throw new Error('Usage: node studio-reference.mjs STUDIO MODEL_DIR OUTPUT FRAME1 FRAME2');
const studio = resolve(studioArg), out = resolve(outputArg);
const require = createRequire(join(studio, 'package.json'));
const ort = require('onnxruntime-node'), sharp = require('sharp');
const { ServerSelector } = await import(pathToFileURL(join(studio, 'src/select/engine.ts')).href);
const { sessionOptions } = await import(pathToFileURL(join(studio, 'src/select/runtime.ts')).href);
await mkdir(out, { recursive: true });
const session = await ort.InferenceSession.create(join(resolve(modelArg), 'vision_encoder.onnx'), sessionOptions('cpu'));
let index = 0;
const metadata = [];
const selector = new ServerSelector({
  available: async () => true,
  load: async () => ({
    ort, provider: 'cpu', model: 'large', loading: { encoder: 'path', decoder: 'path' },
    decoder: { release: async () => {} },
    encoder: {
      release: () => session.release(),
      run: async feeds => {
        await writeFile(join(out, `${index}.input.f32`), Buffer.from(feeds.pixel_values.data.buffer));
        const start = performance.now();
        const results = await session.run(feeds);
        metadata[index].encode_seconds = (performance.now() - start) / 1000;
        for (let i = 0; i < 3; i++) {
          const tensor = results[`image_embeddings.${i}`];
          await writeFile(join(out, `${index}.${i}.f32`), Buffer.from(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength));
        }
        return results;
      },
    },
  }),
});
try {
  for (index = 0; index < frames.length; index++) {
    const bytes = await readFile(resolve(studio, frames[index]));
    // This mirrors engine.ts's raw stage; its normalized output is checked byte-for-byte in Python.
    const rgb = await sharp(bytes, { limitInputPixels: 64_000_000 }).autoOrient().removeAlpha().toColourspace('srgb')
      .resize(1024, 1024, { fit: 'fill', kernel: 'linear' }).raw().toBuffer();
    await writeFile(join(out, `${index}.rgb`), rgb);
    await sharp(rgb, { raw: { width: 1024, height: 1024, channels: 3 } }).png().toFile(join(out, `${index}.png`));
    metadata.push({ frame: frames[index], sha256: createHash('sha256').update(bytes).digest('hex') });
    await selector.select(`parity-${index}`, async () => bytes, { points: [] });
    console.log(`Reference ${index + 1}/2: ${metadata[index].encode_seconds.toFixed(2)} s`);
  }
} finally { await selector.dispose(); }
await writeFile(join(out, 'reference.json'), JSON.stringify({ node: process.version, ort: require('onnxruntime-node/package.json').version, frames: metadata }, null, 2) + '\n');

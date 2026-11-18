/*
 * train.js — 브라우저에서 CNN을 직접 학습합니다 (역전파 + Adam).
 *
 * 구조는 코랩 노트북과 같습니다:
 *   Rescaling → Conv2D(8) → MaxPool → Conv2D(16) → MaxPool → Conv2D(32) → MaxPool
 *   → Flatten → Dense(24, relu) → Dropout(0.3) → Dense(3, softmax)
 * 가중치 초기화(Glorot uniform), 최적화(Adam, 학습률 0.001), 손실(교차 엔트로피)도 Keras 기본값과 같습니다.
 * 학습이 끝나면 js/cnn.js 가 읽는 mouth-cnn-v1 형식으로 내보냅니다.
 */
(function (global) {
  'use strict';

  // 재현 가능한 난수 (시드 고정)
  function makeRandom(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const DEFAULT_ARCH = [
    { type: 'Rescaling', scale: 1 / 255, offset: 0 },
    { type: 'Conv2D', filters: 8, kernel: 3, padding: 'same', activation: 'relu' },
    { type: 'MaxPooling2D', pool: 2 },
    { type: 'Conv2D', filters: 16, kernel: 3, padding: 'same', activation: 'relu' },
    { type: 'MaxPooling2D', pool: 2 },
    { type: 'Conv2D', filters: 32, kernel: 3, padding: 'same', activation: 'relu' },
    { type: 'MaxPooling2D', pool: 2 },
    { type: 'Flatten' },
    { type: 'Dense', units: 24, activation: 'relu' },
    { type: 'Dropout', rate: 0.3 },
    { type: 'Dense', units: 3, activation: 'softmax' },
  ];

  function glorot(n, fanIn, fanOut, rand) {
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = (rand() * 2 - 1) * limit;
    return w;
  }

  function f32ToB64(arr) {
    const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  class Net {
    constructor(arch, inputShape, seed) {
      const rand = makeRandom(seed == null ? 42 : seed);
      let [h, w, c] = inputShape;
      this.inputShape = inputShape.slice();
      this.rand = makeRandom((seed || 42) + 1); // 드롭아웃용
      this.layers = [];
      for (const a of arch) {
        const L = Object.assign({}, a, { inH: h, inW: w, inC: c, params: [] });
        if (a.type === 'Conv2D') {
          const k = a.kernel;
          L.pad = a.padding === 'same' ? (k - 1) >> 1 : 0;
          L.outH = a.padding === 'same' ? h : h - k + 1;
          L.outW = a.padding === 'same' ? w : w - k + 1;
          L.outC = a.filters;
          L.params = [glorot(k * k * c * a.filters, k * k * c, k * k * a.filters, rand), new Float32Array(a.filters)];
        } else if (a.type === 'MaxPooling2D') {
          L.outH = Math.floor(h / a.pool);
          L.outW = Math.floor(w / a.pool);
          L.outC = c;
        } else if (a.type === 'Flatten') {
          L.outH = 1; L.outW = 1; L.outC = h * w * c;
        } else if (a.type === 'Dense') {
          if (h !== 1 || w !== 1) throw new Error('Dense 앞에는 Flatten이 필요합니다');
          L.outH = 1; L.outW = 1; L.outC = a.units;
          L.params = [glorot(c * a.units, c, a.units, rand), new Float32Array(a.units)];
        } else if (a.type === 'Rescaling' || a.type === 'Dropout') {
          L.outH = h; L.outW = w; L.outC = c;
        } else {
          throw new Error('학습에서 지원하지 않는 층: ' + a.type);
        }
        L.grads = L.params.map((p) => new Float32Array(p.length));
        L.m = L.params.map((p) => new Float32Array(p.length));
        L.v = L.params.map((p) => new Float32Array(p.length));
        L.out = new Float32Array(L.outH * L.outW * L.outC);
        if (a.type === 'MaxPooling2D') L.arg = new Int32Array(L.out.length);
        if (a.type === 'Dropout') L.mask = new Float32Array(L.out.length);
        this.layers.push(L);
        h = L.outH; w = L.outW; c = L.outC;
      }
      const last = this.layers[this.layers.length - 1];
      if (last.type !== 'Dense' || last.activation !== 'softmax') throw new Error('마지막 층은 softmax Dense여야 합니다');
      this.firstTrainable = this.layers.findIndex((L) => L.params.length > 0);
      this.step = 0;
      this.dx = this.layers.map((L) => new Float32Array(L.inH * L.inW * L.inC));
    }

    get paramCount() {
      return this.layers.reduce((s, L) => s + L.params.reduce((a, p) => a + p.length, 0), 0);
    }

    // ---------- 순전파 ----------
    forward(x, training) {
      let t = x;
      for (const L of this.layers) {
        const o = L.out;
        switch (L.type) {
          case 'Rescaling':
            for (let i = 0; i < o.length; i++) o[i] = t[i] * L.scale + L.offset;
            break;
          case 'Conv2D': {
            const H = L.inH, W = L.inW, C = L.inC, F = L.outC, k = L.kernel, pad = L.pad;
            const K = L.params[0], B = L.params[1], OH = L.outH, OW = L.outW;
            for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
              const ob = (oy * OW + ox) * F;
              for (let f = 0; f < F; f++) o[ob + f] = B[f];
              for (let ky = 0; ky < k; ky++) {
                const iy = oy + ky - pad;
                if (iy < 0 || iy >= H) continue;
                for (let kx = 0; kx < k; kx++) {
                  const ix = ox + kx - pad;
                  if (ix < 0 || ix >= W) continue;
                  const xi = (iy * W + ix) * C, kb = (ky * k + kx) * C * F;
                  for (let ci = 0; ci < C; ci++) {
                    const v = t[xi + ci];
                    if (v === 0) continue;
                    const kr = kb + ci * F;
                    for (let f = 0; f < F; f++) o[ob + f] += v * K[kr + f];
                  }
                }
              }
            }
            if (L.activation === 'relu') for (let i = 0; i < o.length; i++) if (o[i] < 0) o[i] = 0;
            break;
          }
          case 'MaxPooling2D': {
            const W = L.inW, C = L.inC, p = L.pool, OH = L.outH, OW = L.outW;
            for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) for (let c = 0; c < C; c++) {
              let best = -Infinity, bi = 0;
              for (let ky = 0; ky < p; ky++) for (let kx = 0; kx < p; kx++) {
                const idx = ((oy * p + ky) * W + (ox * p + kx)) * C + c;
                if (t[idx] > best) { best = t[idx]; bi = idx; }
              }
              const oi = (oy * OW + ox) * C + c;
              o[oi] = best;
              L.arg[oi] = bi;
            }
            break;
          }
          case 'Flatten':
            o.set(t);
            break;
          case 'Dropout':
            if (training) {
              const keep = 1 - L.rate;
              for (let i = 0; i < o.length; i++) {
                L.mask[i] = this.rand() < L.rate ? 0 : 1 / keep;
                o[i] = t[i] * L.mask[i];
              }
            } else o.set(t);
            break;
          case 'Dense': {
            const I = L.inC, U = L.outC, K = L.params[0], B = L.params[1];
            for (let u = 0; u < U; u++) o[u] = B[u];
            for (let i = 0; i < I; i++) {
              const v = t[i];
              if (v === 0) continue;
              const kr = i * U;
              for (let u = 0; u < U; u++) o[u] += v * K[kr + u];
            }
            if (L.activation === 'relu') for (let u = 0; u < U; u++) { if (o[u] < 0) o[u] = 0; }
            else if (L.activation === 'softmax') {
              let mx = -Infinity;
              for (let u = 0; u < U; u++) mx = Math.max(mx, o[u]);
              let s = 0;
              for (let u = 0; u < U; u++) { o[u] = Math.exp(o[u] - mx); s += o[u]; }
              for (let u = 0; u < U; u++) o[u] /= s;
            }
            break;
          }
        }
        t = o;
      }
      return t;
    }

    // ---------- 역전파 (기울기를 grads에 더함) ----------
    backward(x, label) {
      const n = this.layers.length;
      const probs = this.layers[n - 1].out;
      let g = new Float32Array(probs.length);
      for (let i = 0; i < g.length; i++) g[i] = probs[i] - (i === label ? 1 : 0); // softmax + 교차 엔트로피
      for (let li = n - 1; li >= 0; li--) {
        const L = this.layers[li];
        const inp = li === 0 ? x : this.layers[li - 1].out;
        const needDx = li > this.firstTrainable;
        if (L.type === 'Rescaling' || (!needDx && L.params.length === 0)) break;
        const dx = this.dx[li];
        switch (L.type) {
          case 'Dense': {
            const I = L.inC, U = L.outC, K = L.params[0], gK = L.grads[0], gB = L.grads[1];
            if (L.activation === 'relu') for (let u = 0; u < U; u++) if (L.out[u] <= 0) g[u] = 0;
            for (let u = 0; u < U; u++) gB[u] += g[u];
            for (let i = 0; i < I; i++) {
              const v = inp[i], kr = i * U;
              let acc = 0;
              for (let u = 0; u < U; u++) {
                if (v !== 0) gK[kr + u] += v * g[u];
                acc += K[kr + u] * g[u];
              }
              if (needDx) dx[i] = acc;
            }
            break;
          }
          case 'Dropout':
            for (let i = 0; i < dx.length; i++) dx[i] = g[i] * L.mask[i];
            break;
          case 'Flatten':
            dx.set(g);
            break;
          case 'MaxPooling2D':
            dx.fill(0);
            for (let i = 0; i < g.length; i++) dx[L.arg[i]] += g[i];
            break;
          case 'Conv2D': {
            const H = L.inH, W = L.inW, C = L.inC, F = L.outC, k = L.kernel, pad = L.pad;
            const K = L.params[0], gK = L.grads[0], gB = L.grads[1], OH = L.outH, OW = L.outW;
            const gf = new Float32Array(F);
            if (needDx) dx.fill(0);
            for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
              const ob = (oy * OW + ox) * F;
              let any = false;
              for (let f = 0; f < F; f++) {
                const v = L.activation === 'relu' && L.out[ob + f] <= 0 ? 0 : g[ob + f];
                gf[f] = v;
                if (v !== 0) { any = true; gB[f] += v; }
              }
              if (!any) continue;
              for (let ky = 0; ky < k; ky++) {
                const iy = oy + ky - pad;
                if (iy < 0 || iy >= H) continue;
                for (let kx = 0; kx < k; kx++) {
                  const ix = ox + kx - pad;
                  if (ix < 0 || ix >= W) continue;
                  const xi = (iy * W + ix) * C, kb = (ky * k + kx) * C * F;
                  for (let ci = 0; ci < C; ci++) {
                    const v = inp[xi + ci], kr = kb + ci * F;
                    if (v !== 0) for (let f = 0; f < F; f++) gK[kr + f] += v * gf[f];
                    if (needDx) {
                      let acc = 0;
                      for (let f = 0; f < F; f++) acc += K[kr + f] * gf[f];
                      dx[xi + ci] += acc;
                    }
                  }
                }
              }
            }
            break;
          }
        }
        if (!needDx) break;
        g = dx;
      }
    }

    zeroGrad() {
      for (const L of this.layers) for (const gr of L.grads) gr.fill(0);
    }

    // Adam (Keras 기본값: beta1 0.9, beta2 0.999, epsilon 1e-7)
    adam(lr, batchSize) {
      this.step += 1;
      const b1 = 0.9, b2 = 0.999, eps = 1e-7;
      const alpha = (lr * Math.sqrt(1 - Math.pow(b2, this.step))) / (1 - Math.pow(b1, this.step));
      for (const L of this.layers) {
        for (let pi = 0; pi < L.params.length; pi++) {
          const p = L.params[pi], gr = L.grads[pi], m = L.m[pi], v = L.v[pi];
          for (let j = 0; j < p.length; j++) {
            const gj = gr[j] / batchSize;
            m[j] = b1 * m[j] + (1 - b1) * gj;
            v[j] = b2 * v[j] + (1 - b2) * gj * gj;
            p[j] -= (alpha * m[j]) / (Math.sqrt(v[j]) + eps);
          }
        }
      }
    }

    predict(x) {
      return Array.from(this.forward(x, false));
    }

    // js/cnn.js 와 코랩 내보내기 셀이 쓰는 형식(mouth-cnn-v1)으로 변환
    toSpec(classes, note) {
      const layers = this.layers.map((L, i) => {
        const name = L.type.toLowerCase() + '_' + i;
        switch (L.type) {
          case 'Rescaling': return { type: 'Rescaling', name, scale: L.scale, offset: L.offset };
          case 'Conv2D': return {
            type: 'Conv2D', name, filters: L.filters, kernel_size: [L.kernel, L.kernel], strides: [1, 1],
            dilation_rate: [1, 1], padding: L.padding, activation: L.activation, use_bias: true,
            weights: L.params.map(f32ToB64), weight_shapes: [[L.kernel, L.kernel, L.inC, L.filters], [L.filters]],
          };
          case 'MaxPooling2D': return { type: 'MaxPooling2D', name, pool_size: [L.pool, L.pool], strides: [L.pool, L.pool], padding: 'valid' };
          case 'Flatten': return { type: 'Flatten', name };
          case 'Dropout': return { type: 'Dropout', name, rate: L.rate };
          case 'Dense': return {
            type: 'Dense', name, units: L.units, activation: L.activation, use_bias: true,
            weights: L.params.map(f32ToB64), weight_shapes: [[L.inC, L.units], [L.units]],
          };
        }
        return null;
      });
      const [h, w, c] = this.inputShape;
      const expected = this.predict(global.MouthCNN.makeTestPattern(h, w, c));
      const d = new Date();
      const pad = (n) => String(n).padStart(2, '0');
      return {
        format: 'mouth-cnn-v1',
        classes: classes.slice(),
        input_shape: [h, w, c],
        created: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
        note: note || '',
        source: 'browser',
        layers,
        selftest: { expected },
      };
    }
  }

  // ---------- 데이터 증강: 좌우 반전, ±3픽셀 이동, 밝기 ±15% ----------
  function augment(src, out, H, W, C, rand) {
    const flip = rand() < 0.5;
    const dx = Math.round((rand() * 2 - 1) * 3), dy = Math.round((rand() * 2 - 1) * 3);
    const bright = 1 + (rand() * 2 - 1) * 0.15;
    for (let y = 0; y < H; y++) {
      const sy = Math.min(H - 1, Math.max(0, y - dy));
      for (let x = 0; x < W; x++) {
        let sx = flip ? W - 1 - x : x;
        sx = Math.min(W - 1, Math.max(0, sx - dx));
        const si = (sy * W + sx) * C, oi = (y * W + x) * C;
        for (let c = 0; c < C; c++) out[oi + c] = Math.min(255, src[si + c] * bright);
      }
    }
    return out;
  }

  // 분류별로 같은 비율을 검증용으로 떼어냄
  function stratifiedSplit(samples, nClasses, valFrac, rand) {
    const train = [], val = [];
    for (let k = 0; k < nClasses; k++) {
      const items = samples.filter((s) => s.y === k);
      shuffle(items, rand);
      const nv = items.length >= 5 ? Math.max(1, Math.round(items.length * valFrac)) : 0;
      val.push(...items.slice(0, nv));
      train.push(...items.slice(nv));
    }
    return { train, val };
  }

  function shuffle(a, rand) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function evaluate(net, set, nClasses) {
    let loss = 0, correct = 0;
    const confusion = Array.from({ length: nClasses }, () => new Array(nClasses).fill(0));
    for (const s of set) {
      const p = net.forward(s.x, false);
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      loss += -Math.log(p[s.y] + 1e-7);
      if (best === s.y) correct++;
      confusion[s.y][best]++;
    }
    return { loss: set.length ? loss / set.length : NaN, correct, total: set.length, confusion };
  }

  const tick = () => new Promise((r) => setTimeout(r, 0));

  /*
   * samples: [{ x: Float32Array(64*64*3, 0~255), y: 분류 번호 }]
   * opts: { epochs, batchSize, lr, valFrac, seed, onProgress(info), shouldStop() }
   */
  async function trainModel(samples, opts) {
    // 배치 4, 학습률 0.001 = 코랩 노트북과 같은 설정.
    // epochs를 정하지 않으면 사진 수와 상관없이 가중치를 약 targetSteps번 업데이트하도록 epoch 수를 정함
    // (사진이 적을 때 epoch만 정하면 업데이트 횟수가 너무 적어 한 분류를 아예 못 배우는 문제가 있었음)
    const o = Object.assign({ batchSize: 4, lr: 0.001, targetSteps: 600, valFrac: 0.2, seed: 42, inputShape: [64, 64, 3], nClasses: 3 }, opts);
    const rand = makeRandom(o.seed + 7);
    const net = new Net(o.arch || DEFAULT_ARCH, o.inputShape, o.seed);
    const { train, val } = stratifiedSplit(samples.slice(), o.nClasses, o.valFrac, rand);
    if (train.length < o.nClasses) throw new Error('학습할 사진이 너무 적습니다');
    const [H, W, C] = o.inputShape;
    const buf = new Float32Array(H * W * C);
    const history = [];
    const stepsPerEpoch = Math.ceil(train.length / o.batchSize);
    if (!o.epochs) o.epochs = Math.min(200, Math.max(15, Math.ceil(o.targetSteps / stepsPerEpoch)));
    let best = null;
    const t0 = performance.now();
    let lastYield = performance.now();

    // 끝날 때(중지 포함) 검증 손실이 가장 낮았던 가중치로 되돌림 (Keras의 restore_best_weights와 같은 생각)
    const finish = (stopped) => {
      if (best && o.restoreBest !== false) {
        net.layers.forEach((L, i) => L.params.forEach((p, j) => p.set(best.snapshot[i][j])));
      }
      const finalVal = val.length ? evaluate(net, val, o.nClasses) : null;
      const finalTrain = evaluate(net, train, o.nClasses); // 증강 없이 학습 사진을 다시 분류
      return { net, history, stopped, train, val, best, finalVal, finalTrain, epochs: o.epochs, steps: net.step, seconds: (performance.now() - t0) / 1000 };
    };

    for (let ep = 1; ep <= o.epochs; ep++) {
      shuffle(train, rand);
      let lossSum = 0, correct = 0;
      for (let b = 0; b < stepsPerEpoch; b++) {
        const batch = train.slice(b * o.batchSize, (b + 1) * o.batchSize);
        net.zeroGrad();
        for (const s of batch) {
          const x = augment(s.x, buf, H, W, C, rand);
          const p = net.forward(x, true);
          lossSum += -Math.log(p[s.y] + 1e-7);
          let arg = 0;
          for (let i = 1; i < p.length; i++) if (p[i] > p[arg]) arg = i;
          if (arg === s.y) correct++;
          net.backward(x, s.y);
        }
        net.adam(o.lr, batch.length);
        if (performance.now() - lastYield > 30) {
          if (o.onProgress) o.onProgress({ phase: 'batch', epoch: ep, epochs: o.epochs, step: b + 1, steps: stepsPerEpoch, history });
          await tick();
          lastYield = performance.now();
          if (o.shouldStop && o.shouldStop()) return finish(true);
        }
      }
      const v = val.length ? evaluate(net, val, o.nClasses) : null;
      const rec = { epoch: ep, loss: lossSum / train.length, acc: correct / train.length, valLoss: v ? v.loss : null, valAcc: v ? v.correct / v.total : null };
      history.push(rec);
      // 검증 손실이 가장 낮았던 시점의 가중치를 기억
      if (v && (!best || v.loss < best.valLoss)) {
        best = { epoch: ep, valLoss: v.loss, eval: v, snapshot: net.layers.map((L) => L.params.map((p) => p.slice())) };
      }
      if (o.onProgress) o.onProgress({ phase: 'epoch', epoch: ep, epochs: o.epochs, history, val: v, seconds: (performance.now() - t0) / 1000 });
      await tick();
      lastYield = performance.now();
      if (o.shouldStop && o.shouldStop()) return finish(true);
    }
    return finish(false);
  }

  const api = { Net, DEFAULT_ARCH, trainModel, augment, evaluate, makeRandom };
  global.MouthTrain = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);

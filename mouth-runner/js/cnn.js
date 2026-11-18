/*
 * cnn.js — 코랩(Keras)에서 학습한 CNN을 브라우저에서 그대로 계산합니다.
 *
 * TensorFlow.js 없이 순수 자바스크립트로 합성곱·풀링·완전연결 층을 직접 계산합니다.
 * 가중치는 코랩의 "게임용 모델 내보내기" 셀이 만든 model/mouth_model.js 에서 읽습니다.
 *
 * 지원하는 층: Rescaling, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization,
 *             Flatten, GlobalAveragePooling2D, Dense, Dropout(추론 때는 통과), Activation
 * 데이터 순서: Keras 기본값과 같은 channels_last (세로, 가로, 채널)
 */
(function (global) {
  'use strict';

  // ---------- 가중치 읽기: base64 → float32 배열 ----------
  function b64ToFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  // ---------- 활성화 함수 ----------
  const ACTIVATIONS = {
    linear: null,
    relu(d) {
      for (let i = 0; i < d.length; i++) if (d[i] < 0) d[i] = 0;
    },
    sigmoid(d) {
      for (let i = 0; i < d.length; i++) d[i] = 1 / (1 + Math.exp(-d[i]));
    },
    tanh(d) {
      for (let i = 0; i < d.length; i++) d[i] = Math.tanh(d[i]);
    },
    // 마지막 축(채널) 기준 softmax
    softmax(d, c) {
      for (let base = 0; base < d.length; base += c) {
        let max = -Infinity;
        for (let i = 0; i < c; i++) if (d[base + i] > max) max = d[base + i];
        let sum = 0;
        for (let i = 0; i < c; i++) {
          d[base + i] = Math.exp(d[base + i] - max);
          sum += d[base + i];
        }
        for (let i = 0; i < c; i++) d[base + i] /= sum;
      }
    },
  };

  function activate(name, t) {
    if (!name || name === 'linear') return t;
    const fn = ACTIVATIONS[name];
    if (!fn) throw new Error('지원하지 않는 활성화 함수: ' + name);
    fn(t.d, t.c);
    return t;
  }

  // ---------- 패딩 계산 (TensorFlow 규칙과 동일) ----------
  function padInfo(size, k, stride, dilation, padding) {
    const eff = (k - 1) * dilation + 1;
    if (padding === 'same') {
      const out = Math.ceil(size / stride);
      const total = Math.max((out - 1) * stride + eff - size, 0);
      return { out, before: Math.floor(total / 2) };
    }
    if (padding === 'valid') {
      return { out: Math.floor((size - eff) / stride) + 1, before: 0 };
    }
    throw new Error('지원하지 않는 padding: ' + padding);
  }

  // ---------- 층별 계산 ----------
  // 텐서 t = { h, w, c, d }  (d는 길이 h*w*c 인 Float32Array, 순서는 [세로][가로][채널])

  function rescaling(t, L) {
    const d = t.d;
    for (let i = 0; i < d.length; i++) d[i] = d[i] * L.scale + L.offset;
    return t;
  }

  function conv2d(t, L) {
    const H = t.h, W = t.w, C = t.c, x = t.d;
    const [kh, kw] = L.kernel_size;
    const [sh, sw] = L.strides;
    const [dh, dw] = L.dilation_rate;
    const F = L.filters, K = L.kernel, B = L.bias;
    const py = padInfo(H, kh, sh, dh, L.padding);
    const px = padInfo(W, kw, sw, dw, L.padding);
    const OH = py.out, OW = px.out;
    const out = new Float32Array(OH * OW * F);
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        const o = (oy * OW + ox) * F;
        if (B) for (let f = 0; f < F; f++) out[o + f] = B[f];
        for (let ky = 0; ky < kh; ky++) {
          const iy = oy * sh + ky * dh - py.before;
          if (iy < 0 || iy >= H) continue; // 바깥은 0으로 채운 것과 같음
          for (let kx = 0; kx < kw; kx++) {
            const ix = ox * sw + kx * dw - px.before;
            if (ix < 0 || ix >= W) continue;
            const xi = (iy * W + ix) * C;
            const kb = (ky * kw + kx) * C * F;
            for (let ci = 0; ci < C; ci++) {
              const v = x[xi + ci];
              if (v === 0) continue;
              const kr = kb + ci * F;
              for (let f = 0; f < F; f++) out[o + f] += v * K[kr + f];
            }
          }
        }
      }
    }
    return activate(L.activation, { h: OH, w: OW, c: F, d: out });
  }

  function pool2d(t, L, mode) {
    const H = t.h, W = t.w, C = t.c, x = t.d;
    const [ph, pw] = L.pool_size;
    const [sh, sw] = L.strides;
    const py = padInfo(H, ph, sh, 1, L.padding);
    const px = padInfo(W, pw, sw, 1, L.padding);
    const OH = py.out, OW = px.out;
    const out = new Float32Array(OH * OW * C);
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        for (let c = 0; c < C; c++) {
          let acc = mode === 'max' ? -Infinity : 0;
          let n = 0;
          for (let ky = 0; ky < ph; ky++) {
            const iy = oy * sh + ky - py.before;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < pw; kx++) {
              const ix = ox * sw + kx - px.before;
              if (ix < 0 || ix >= W) continue;
              const v = x[(iy * W + ix) * C + c];
              if (mode === 'max') { if (v > acc) acc = v; } else acc += v;
              n++;
            }
          }
          out[(oy * OW + ox) * C + c] = mode === 'max' ? acc : acc / n;
        }
      }
    }
    return { h: OH, w: OW, c: C, d: out };
  }

  function batchNorm(t, L) {
    const C = t.c, d = t.d;
    const mul = new Float32Array(C), add = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      const g = L.gamma ? L.gamma[c] : 1;
      const b = L.beta ? L.beta[c] : 0;
      const inv = g / Math.sqrt(L.moving_variance[c] + L.epsilon);
      mul[c] = inv;
      add[c] = b - L.moving_mean[c] * inv;
    }
    for (let i = 0; i < d.length; i++) {
      const c = i % C;
      d[i] = d[i] * mul[c] + add[c];
    }
    return t;
  }

  function flatten(t) {
    return { h: 1, w: 1, c: t.h * t.w * t.c, d: t.d };
  }

  function globalAvgPool(t) {
    const C = t.c, n = t.h * t.w, out = new Float32Array(C);
    for (let i = 0; i < n; i++) for (let c = 0; c < C; c++) out[c] += t.d[i * C + c];
    for (let c = 0; c < C; c++) out[c] /= n;
    return { h: 1, w: 1, c: C, d: out };
  }

  // Keras의 Dense는 마지막 축에 적용됨 (Flatten 다음이면 일반적인 완전연결층)
  function dense(t, L) {
    const I = t.c, U = L.units, K = L.kernel, B = L.bias;
    const n = t.h * t.w;
    const out = new Float32Array(n * U);
    for (let p = 0; p < n; p++) {
      const o = p * U, xi = p * I;
      if (B) for (let u = 0; u < U; u++) out[o + u] = B[u];
      for (let i = 0; i < I; i++) {
        const v = t.d[xi + i];
        if (v === 0) continue;
        const kr = i * U;
        for (let u = 0; u < U; u++) out[o + u] += v * K[kr + u];
      }
    }
    return activate(L.activation, { h: t.h, w: t.w, c: U, d: out });
  }

  // ---------- 모델 파일 해석 ----------
  function expectWeights(L, count) {
    if (!L._w || L._w.length !== count) {
      throw new Error(`${L.type}(${L.name}) 층의 가중치 개수가 맞지 않습니다`);
    }
  }

  function prepareLayer(raw) {
    const L = Object.assign({}, raw);
    L._w = (raw.weights || []).map((b64, i) => {
      const arr = b64ToFloat32(b64);
      const shape = raw.weight_shapes[i];
      const size = shape.reduce((a, b) => a * b, 1);
      if (arr.length !== size) throw new Error(`${raw.type} 층 가중치 크기 오류`);
      return arr;
    });
    switch (L.type) {
      case 'Conv2D':
        expectWeights(L, L.use_bias ? 2 : 1);
        L.kernel = L._w[0];
        L.bias = L.use_bias ? L._w[1] : null;
        L.dilation_rate = L.dilation_rate || [1, 1];
        break;
      case 'Dense':
        expectWeights(L, L.use_bias ? 2 : 1);
        L.kernel = L._w[0];
        L.bias = L.use_bias ? L._w[1] : null;
        break;
      case 'BatchNormalization': {
        let k = 0;
        L.gamma = L.scale ? L._w[k++] : null;
        L.beta = L.center ? L._w[k++] : null;
        L.moving_mean = L._w[k++];
        L.moving_variance = L._w[k++];
        expectWeights(L, k);
        break;
      }
      case 'MaxPooling2D':
      case 'AveragePooling2D':
        L.strides = L.strides || L.pool_size;
        break;
      case 'Rescaling':
      case 'Flatten':
      case 'GlobalAveragePooling2D':
      case 'Dropout':
      case 'Activation':
        break;
      default:
        throw new Error('지원하지 않는 층: ' + L.type);
    }
    return L;
  }

  function runLayer(t, L) {
    switch (L.type) {
      case 'Rescaling': return rescaling(t, L);
      case 'Conv2D': return conv2d(t, L);
      case 'MaxPooling2D': return pool2d(t, L, 'max');
      case 'AveragePooling2D': return pool2d(t, L, 'avg');
      case 'BatchNormalization': return batchNorm(t, L);
      case 'Flatten': return flatten(t);
      case 'GlobalAveragePooling2D': return globalAvgPool(t);
      case 'Dense': return dense(t, L);
      case 'Dropout': return t; // 추론 때는 아무것도 하지 않음
      case 'Activation': return activate(L.activation, t);
    }
    throw new Error('지원하지 않는 층: ' + L.type);
  }

  // 코랩 내보내기 셀과 똑같은 규칙으로 만드는 검증용 입력(사진이 아닌 숫자 무늬)
  function makeTestPattern(h, w, c) {
    const d = new Float32Array(h * w * c);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let k = 0; k < c; k++) d[(y * w + x) * c + k] = (y * 31 + x * 17 + k * 101 + y * x) % 256;
    return d;
  }

  class CNN {
    constructor(spec) {
      if (!spec || spec.format !== 'mouth-cnn-v1') {
        throw new Error('모델 파일 형식이 올바르지 않습니다 (mouth-cnn-v1 아님)');
      }
      this.spec = spec;
      this.classes = spec.classes.slice();
      [this.h, this.w, this.c] = spec.input_shape;
      this.layers = spec.layers.map(prepareLayer);
    }

    // input: 길이 h*w*c 인 배열, RGB 0~255 (학습 때와 같은 범위; 정규화는 모델의 Rescaling 층이 처리)
    predict(input) {
      if (input.length !== this.h * this.w * this.c) throw new Error('입력 크기가 모델과 다릅니다');
      let t = { h: this.h, w: this.w, c: this.c, d: Float32Array.from(input) };
      for (const L of this.layers) t = runLayer(t, L);
      return Array.from(t.d);
    }

    // 코랩에서 계산한 결과와 브라우저 계산 결과가 같은지 확인
    selfTest(tolerance = 1e-3) {
      const expected = this.spec.selftest && this.spec.selftest.expected;
      if (!expected) return { ok: false, maxDiff: NaN, message: '검증값 없음' };
      const got = this.predict(makeTestPattern(this.h, this.w, this.c));
      let maxDiff = 0;
      for (let i = 0; i < expected.length; i++) maxDiff = Math.max(maxDiff, Math.abs(got[i] - expected[i]));
      return { ok: maxDiff <= tolerance, maxDiff, got, expected };
    }
  }

  const api = { CNN, makeTestPattern, b64ToFloat32 };
  global.MouthCNN = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);

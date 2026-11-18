/*
 * mouth-cam.js — 웹캠 영상에서 입 부분을 잘라 CNN 입력(64×64)을 만들고,
 *                예측 결과를 안정적인 조작 명령으로 바꿉니다.
 *
 * 게임(index.html)과 데이터 수집(collect.html)이 같은 코드를 쓰므로
 * "학습할 때 본 사진"과 "게임할 때 보는 사진"이 같은 방식으로 만들어집니다.
 */
(function (global) {
  'use strict';

  const SIZE = 64;          // CNN 입력 크기 (코랩 노트북과 동일)
  const BOX_RATIO = 1.45;   // 안내 박스 가로:세로 비율 (학습 사진의 평균 비율)
  const SETTINGS_KEY = 'mouthRunner.settings.v1';

  // 분류 이름 → 게임 동작
  const ACTIONS = {
    closed: { key: 'run', label: '달리기', hint: '입 다물기', color: '#3b82f6' },
    open: { key: 'jump', label: '점프', hint: '입 벌리기', color: '#16a34a' },
    teeth: { key: 'duck', label: '수그리기', hint: '이 보이게 웃기', color: '#f59e0b' },
  };

  function loadSettings() {
    const base = { cx: 0.5, cy: 0.66, size: 0.3, threshold: 0.6 };
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
      return Object.assign(base, saved);
    } catch (e) {
      return base;
    }
  }

  function saveSettings(s) {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
    } catch (e) {
      /* 저장이 안 되는 환경이면 무시 */
    }
  }

  // ---------------------------------------------------------------
  // 웹캠 + 입 영역 자르기
  // ---------------------------------------------------------------
  class MouthCam {
    constructor({ preview, aiView, settings }) {
      this.preview = preview;               // 거울처럼 뒤집힌 웹캠 미리보기 캔버스
      this.aiView = aiView || null;         // "AI가 보는 화면" 캔버스
      this.settings = settings;
      this.video = document.createElement('video');
      this.video.muted = true;
      this.video.playsInline = true;
      this.video.setAttribute('playsinline', '');
      this.work = document.createElement('canvas');
      this.work.width = this.work.height = SIZE;
      this.workCtx = this.work.getContext('2d', { willReadFrequently: true });
      this.stream = null;

      // 미리보기를 클릭하면 그 위치로 박스 이동
      if (preview) {
        preview.addEventListener('pointerdown', (e) => {
          const r = preview.getBoundingClientRect();
          this.settings.cx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
          this.settings.cy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
          saveSettings(this.settings);
        });
      }
    }

    async start() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('이 브라우저에서는 웹캠을 쓸 수 없습니다 (https 주소인지 확인하세요)');
      }
      this.stop();
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      for (let i = 0; i < 100 && !this.video.videoWidth; i++) await new Promise((r) => setTimeout(r, 30));
      if (this.preview) {
        this.preview.width = 320;
        this.preview.height = Math.round((320 * this.video.videoHeight) / this.video.videoWidth) || 240;
      }
    }

    stop() {
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }

    get ready() {
      return this.video.readyState >= 2 && this.video.videoWidth > 0;
    }

    // 원본 영상 좌표계에서 잘라낼 사각형 (미리보기는 좌우 반전이므로 x를 뒤집어 계산)
    cropRect() {
      const vw = this.video.videoWidth, vh = this.video.videoHeight;
      let sh = this.settings.size * vh;
      let sw = sh * BOX_RATIO;
      if (sw > vw) { sw = vw; sh = sw / BOX_RATIO; }
      const cx = (1 - this.settings.cx) * vw;
      const cy = this.settings.cy * vh;
      const sx = Math.min(Math.max(cx - sw / 2, 0), vw - sw);
      const sy = Math.min(Math.max(cy - sh / 2, 0), vh - sh);
      return { sx, sy, sw, sh, vw, vh };
    }

    // 입 부분을 64×64로 만들어 RGB(0~255) 배열로 돌려줌.
    // 코랩의 tf.image.resize_with_pad 처럼 비율을 유지하고 위아래를 검은색으로 채움.
    grab() {
      const r = this.cropRect();
      const ctx = this.workCtx;
      const dh = Math.floor(SIZE / BOX_RATIO);
      const dy = Math.floor((SIZE - SIZE / BOX_RATIO) / 2);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium';
      ctx.setTransform(-1, 0, 0, 1, SIZE, 0); // 미리보기와 같은 방향(거울)으로
      ctx.drawImage(this.video, r.sx, r.sy, r.sw, r.sh, 0, dy, SIZE, dh);
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
      const out = new Float32Array(SIZE * SIZE * 3);
      for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
        out[j] = px[i];
        out[j + 1] = px[i + 1];
        out[j + 2] = px[i + 2];
      }
      if (this.aiView) {
        const v = this.aiView.getContext('2d');
        v.imageSmoothingEnabled = false;
        v.drawImage(this.work, 0, 0, this.aiView.width, this.aiView.height);
      }
      return out;
    }

    // 현재 64×64 입력을 PNG(base64)로 — 데이터 수집용
    snapshotPNG() {
      return this.work.toDataURL('image/png').split(',')[1];
    }

    drawPreview(statusText) {
      const c = this.preview;
      if (!c) return;
      const g = c.getContext('2d');
      if (!this.ready) {
        g.fillStyle = '#1f2937';
        g.fillRect(0, 0, c.width, c.height);
        g.fillStyle = '#e5e7eb';
        g.font = '14px system-ui, sans-serif';
        g.textAlign = 'center';
        g.fillText(statusText || '웹캠 준비 중…', c.width / 2, c.height / 2);
        return;
      }
      g.save();
      g.translate(c.width, 0);
      g.scale(-1, 1);
      g.drawImage(this.video, 0, 0, c.width, c.height);
      g.restore();

      const r = this.cropRect();
      const k = c.width / r.vw;
      const x = (r.vw - (r.sx + r.sw)) * k; // 거울 좌표로 변환
      const y = r.sy * k;
      g.lineWidth = 3;
      g.strokeStyle = '#facc15';
      g.strokeRect(x, y, r.sw * k, r.sh * k);
      g.fillStyle = 'rgba(0,0,0,0.55)';
      g.fillRect(0, c.height - 22, c.width, 22);
      g.fillStyle = '#fff';
      g.font = '12px system-ui, sans-serif';
      g.textAlign = 'center';
      g.fillText('입을 노란 박스 가운데에 — 화면을 클릭하면 박스 이동', c.width / 2, c.height - 7);
    }
  }

  // ---------------------------------------------------------------
  // 예측 → 조작 명령 (흔들림 줄이기)
  // ---------------------------------------------------------------
  class MouthControl {
    constructor(cam, model, settings) {
      this.cam = cam;
      this.model = model;
      this.settings = settings;
      this.classes = model ? model.classes : Object.keys(ACTIONS);
      this.smooth = 0.5;      // 지수이동평균 비율 (클수록 새 값 반영이 빠름)
      this.stableNeeded = 2;  // 같은 판단이 연속 몇 번 나와야 명령을 바꿀지
      this.intervalMs = 50;   // 최대 초당 20번 예측
      this.probs = null;
      this.raw = null;
      this.command = 'unsure';
      this.pending = null;
      this.pendingCount = 0;
      this.lastT = 0;
      this.inferMs = 0;
      this.paused = false;
    }

    setModel(model) {
      this.model = model;
      this.classes = model ? model.classes : Object.keys(ACTIONS);
      this.probs = null;
      this.command = 'unsure';
      this.pending = null;
      this.pendingCount = 0;
    }

    // true를 돌려주면 이번 프레임에 새 예측을 했다는 뜻
    update(now) {
      if (this.paused || !this.model || !this.cam.ready || now - this.lastT < this.intervalMs) return false;
      this.lastT = now;
      const x = this.cam.grab();
      const t0 = performance.now();
      const p = this.model.predict(x);
      this.inferMs = performance.now() - t0;
      this.raw = p;
      if (!this.probs) this.probs = p.slice();
      else for (let i = 0; i < p.length; i++) this.probs[i] = this.smooth * p[i] + (1 - this.smooth) * this.probs[i];

      let best = 0;
      for (let i = 1; i < this.probs.length; i++) if (this.probs[i] > this.probs[best]) best = i;
      const decided = this.probs[best] >= this.settings.threshold ? this.classes[best] : 'unsure';
      if (decided === this.pending) this.pendingCount++;
      else { this.pending = decided; this.pendingCount = 1; }
      if (this.pendingCount >= this.stableNeeded) this.command = decided;
      return true;
    }
  }

  // ---------------------------------------------------------------
  // 화면 표시 도우미
  // ---------------------------------------------------------------
  function buildBars(container, classes) {
    container.innerHTML = '';
    const rows = {};
    for (const name of classes) {
      const a = ACTIONS[name] || { label: name, hint: '', color: '#6b7280' };
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML =
        `<span class="bar-label">${a.label}<small>${a.hint}</small></span>` +
        `<span class="bar-track"><span class="bar-fill" style="background:${a.color}"></span></span>` +
        `<span class="bar-value">0%</span>`;
      container.appendChild(row);
      rows[name] = { row, fill: row.querySelector('.bar-fill'), value: row.querySelector('.bar-value') };
    }
    return rows;
  }

  function updateBars(rows, classes, probs, command) {
    if (!probs) return;
    classes.forEach((name, i) => {
      const r = rows[name];
      if (!r) return;
      r.fill.style.width = (probs[i] * 100).toFixed(1) + '%';
      r.value.textContent = Math.round(probs[i] * 100) + '%';
      r.row.classList.toggle('active', name === command);
    });
  }

  // ---------------------------------------------------------------
  // 모델 고르기: 이 브라우저에서 학습한 모델(localStorage) 또는 사이트의 model/mouth_model.js
  // ---------------------------------------------------------------
  const LOCAL_MODEL_KEY = 'mouthRunner.localModel.v1';

  function readLocalModel() {
    try {
      const s = localStorage.getItem(LOCAL_MODEL_KEY);
      return s ? JSON.parse(s) : null;
    } catch (e) {
      return null;
    }
  }

  function saveLocalModel(spec) {
    try {
      localStorage.setItem(LOCAL_MODEL_KEY, JSON.stringify(spec));
      return true;
    } catch (e) {
      return false;
    }
  }

  function clearLocalModel() {
    try { localStorage.removeItem(LOCAL_MODEL_KEY); } catch (e) { /* 무시 */ }
  }

  function modelToScript(spec) {
    return '// 자동으로 만든 모델 파일입니다. 직접 고치지 마세요.\nwindow.MOUTH_MODEL = ' + JSON.stringify(spec) + ';\n';
  }

  function makeModel(spec) {
    const model = new global.MouthCNN.CNN(spec);
    const missing = model.classes.filter((c) => !ACTIONS[c]);
    if (missing.length) throw new Error('알 수 없는 분류 이름: ' + missing.join(', '));
    return model;
  }

  // preferFile: 둘 다 있을 때 사이트 모델 파일을 쓸지
  function loadModel(preferFile) {
    const local = readLocalModel();
    const file = global.MOUTH_MODEL || null;
    const info = { hasLocal: !!local, hasFile: !!file };
    let spec = null, source = null;
    if (local && !(preferFile && file)) { spec = local; source = 'local'; }
    else if (file) { spec = file; source = 'file'; }
    if (!spec) {
      return Object.assign(info, {
        model: null, ok: false, source: null,
        message: '아직 학습된 모델이 없습니다. "내 입모양 학습하기"에서 몇 분이면 만들 수 있어요. (지금은 키보드로만 조작)',
      });
    }
    try {
      const model = makeModel(spec);
      const t = model.selfTest();
      const where = source === 'local' ? '이 브라우저에서 학습한 모델' : '사이트 모델 파일';
      const created = spec.created ? ` · ${spec.created}` : '';
      const note = spec.note ? ` · ${spec.note}` : '';
      return Object.assign(info, t.ok
        ? { model, ok: true, source, message: `${where} 사용 중${created}${note} (계산 확인 완료)` }
        : { model, ok: false, source, message: `${where}: 저장된 검증값과 계산이 다릅니다 (차이 ${t.maxDiff})` });
    } catch (e) {
      return Object.assign(info, { model: null, ok: false, source, message: '모델을 읽지 못했습니다: ' + e.message });
    }
  }

  global.MouthKit = {
    SIZE, BOX_RATIO, ACTIONS,
    MouthCam, MouthControl,
    loadSettings, saveSettings, loadModel, buildBars, updateBars,
    makeModel, readLocalModel, saveLocalModel, clearLocalModel, modelToScript,
  };
})(window);

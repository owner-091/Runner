/*
 * collect.js — 게임과 같은 전처리로 64×64 입 사진을 모으고, 브라우저에서 바로 CNN을 학습합니다.
 *   · 사진 JSON은 코랩 노트북의 "웹캠 데이터 추가" 셀에서 그대로 읽힙니다.
 *   · 학습한 모델은 이 브라우저에 저장하거나 mouth_model.js 로 내려받을 수 있습니다.
 */
(function () {
  'use strict';
  const K = window.MouthKit;
  const $ = (id) => document.getElementById(id);
  const CLASSES = ['closed', 'open', 'teeth'];
  const NAMES = {
    closed: { title: '입 다묾', sub: '게임에서 달리기 · 키 1' },
    open: { title: '입 벌림', sub: '게임에서 점프 · 키 2' },
    teeth: { title: '이 보이게 웃기', sub: '게임에서 수그리기 · 키 3' },
  };
  const SHORT = { closed: '다묾', open: '벌림', teeth: '웃기' };
  const SHOT_INTERVAL = 200; // ms
  const MIN_PER_CLASS = 20;

  const settings = K.loadSettings();
  const cam = new K.MouthCam({ preview: $('preview'), aiView: $('aiview'), settings });
  const loaded = K.loadModel(settings.preferFile);
  const control = new K.MouthControl(cam, loaded.model, settings);
  let bars = loaded.model ? K.buildBars($('bars'), control.classes) : null;
  $('barsHint').textContent = loaded.model
    ? '지금 저장된 모델의 예측입니다. 자주 틀리는 분류를 더 모으세요.'
    : '아직 모델이 없어도 사진 모으기와 학습은 됩니다.';

  // 박스 크기 슬라이더
  const size = $('boxSize');
  size.value = settings.size;
  $('boxSizeOut').textContent = Math.round(settings.size * 100) + '%';
  size.addEventListener('input', () => {
    settings.size = parseFloat(size.value);
    $('boxSizeOut').textContent = Math.round(settings.size * 100) + '%';
    K.saveSettings(settings);
  });

  // 회차 이름 기본값: s월일_시분
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  $('session').value = `s${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
  const sessionName = () => ($('session').value || 's1').trim().replace(/[^\w가-힣-]/g, '_');

  // ---------------- 웹캠 ----------------
  async function startCamera() {
    $('camStatus').textContent = '웹캠 연결 중…';
    $('camStatus').classList.remove('warn');
    try {
      await cam.start();
      $('camStatus').textContent = `웹캠 연결됨 (${cam.video.videoWidth}×${cam.video.videoHeight})`;
    } catch (e) {
      $('camStatus').textContent = '웹캠 오류: ' + (e && e.name === 'NotAllowedError'
        ? '카메라 권한이 거부되었습니다. 주소창 왼쪽 아이콘에서 허용하세요.'
        : e && e.name === 'NotReadableError' ? '다른 프로그램이 웹캠을 쓰고 있습니다. 닫고 다시 연결하세요.'
          : (e && e.message) || String(e));
      $('camStatus').classList.add('warn');
    }
  }
  $('camBtn').addEventListener('click', startCamera);
  startCamera();

  // ---------------- 사진 데이터 ----------------
  const records = []; // { name, label, session, png, x(Float32Array) }
  let seq = 0;
  let busy = false;
  let overlay = '';
  let dirty = false;
  const cards = {};

  CLASSES.forEach((label) => {
    const a = K.ACTIONS[label];
    const card = document.createElement('div');
    card.className = 'class-card';
    card.style.setProperty('--c', a.color);
    card.innerHTML =
      `<h3>${NAMES[label].title}</h3><p>${NAMES[label].sub}</p>` +
      `<div class="count">0<small> 장</small></div>` +
      `<button type="button">연속 촬영</button>`;
    card.querySelector('button').addEventListener('click', () => capture(label));
    $('classGrid').appendChild(card);
    cards[label] = card;
  });

  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    const idx = ['Digit1', 'Digit2', 'Digit3'].indexOf(e.code);
    if (idx >= 0) capture(CLASSES[idx]);
  });

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function setBusy(v) {
    busy = v;
    document.querySelectorAll('.class-card button, #download, #clear, #loadJson, #trainBtn').forEach((b) => { b.disabled = v; });
  }

  async function capture(label) {
    if (busy) return;
    if (!cam.ready) { $('saveStatus').textContent = '웹캠이 먼저 연결되어야 합니다'; return; }
    const session = sessionName();
    setBusy(true);
    for (let c = 3; c > 0; c--) {
      overlay = `${NAMES[label].title} 준비 ${c}`;
      await sleep(700);
    }
    const n = parseInt($('batch').value, 10);
    for (let i = 0; i < n; i++) {
      overlay = `${NAMES[label].title} 촬영 ${i + 1}/${n}`;
      const x = cam.grab();
      seq += 1;
      records.push({
        name: `web_${session}_${label}_${String(seq).padStart(4, '0')}.png`,
        label, session, png: cam.snapshotPNG(), x,
      });
      await sleep(SHOT_INTERVAL);
    }
    overlay = '';
    dirty = true;
    setBusy(false);
    render();
  }

  function render() {
    for (const label of CLASSES) {
      const n = records.filter((r) => r.label === label).length;
      cards[label].querySelector('.count').innerHTML = `${n}<small> 장</small>`;
    }
    const sessions = new Set(records.map((r) => r.session));
    $('saveStatus').textContent = records.length
      ? `총 ${records.length}장 · 회차 ${sessions.size}개${dirty ? ' · 아직 내려받지 않음' : ''}`
      : '';
    const box = $('thumbs');
    box.innerHTML = '';
    const isWrong = (r) => r.pred && r.pred !== r.label;
    const wrong = records.filter(isWrong);
    const recent = records.slice(-80).reverse().filter((r) => !isWrong(r));
    [...wrong, ...recent].forEach((r) => {
      const fig = document.createElement('div');
      fig.className = 'thumb' + (isWrong(r) ? ' wrong' : '');
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + r.png;
      img.title = r.pred
        ? `${r.name} — 정답 ${SHORT[r.label]}, 예측 ${SHORT[r.pred]}${r.isVal ? ' (검증용)' : ''} · 눌러서 삭제`
        : `${r.name} — 눌러서 삭제`;
      img.style.setProperty('--c', K.ACTIONS[r.label].color);
      img.addEventListener('click', () => {
        if (busy) return;
        const i = records.indexOf(r);
        if (i >= 0) { records.splice(i, 1); dirty = true; render(); }
      });
      fig.appendChild(img);
      if (isWrong(r)) {
        const b = document.createElement('span');
        b.className = 'badge';
        b.textContent = '→' + SHORT[r.pred];
        fig.appendChild(b);
      }
      box.appendChild(fig);
    });
  }

  function downloadText(text, filename, type) {
    const blob = new Blob([text], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }

  $('download').addEventListener('click', () => {
    if (!records.length) { $('saveStatus').textContent = '아직 찍은 사진이 없습니다'; return; }
    const data = {
      format: 'mouth-data-v1',
      created: new Date().toISOString(),
      classes: CLASSES,
      input_size: K.SIZE,
      box_ratio: K.BOX_RATIO,
      records: records.map(({ name, label, session, png }) => ({ name, label, session, png })),
    };
    downloadText(JSON.stringify(data), `mouth_data_${sessionName()}.json`, 'application/json');
    dirty = false;
    render();
  });

  // 예전에 내려받은 사진 JSON 다시 불러오기
  function pngToPixels(b64) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = c.height = K.SIZE;
        const g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(img, 0, 0, K.SIZE, K.SIZE);
        const d = g.getImageData(0, 0, K.SIZE, K.SIZE).data;
        const x = new Float32Array(K.SIZE * K.SIZE * 3);
        for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { x[j] = d[i]; x[j + 1] = d[i + 1]; x[j + 2] = d[i + 2]; }
        resolve(x);
      };
      img.onerror = () => reject(new Error('사진을 읽지 못했습니다'));
      img.src = 'data:image/png;base64,' + b64;
    });
  }
  $('loadJson').addEventListener('click', () => $('loadFile').click());
  $('loadFile').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;
    setBusy(true);
    let added = 0;
    try {
      const names = new Set(records.map((r) => r.name));
      for (const f of files) {
        const d = JSON.parse(await f.text());
        if (d.format !== 'mouth-data-v1') throw new Error(`${f.name}: 이 페이지에서 내려받은 파일이 아닙니다`);
        for (const r of d.records) {
          if (names.has(r.name) || !CLASSES.includes(r.label)) continue;
          records.push({ name: r.name, label: r.label, session: r.session, png: r.png, x: await pngToPixels(r.png) });
          names.add(r.name);
          const m = /_(\d+)\.png$/.exec(r.name);
          if (m) seq = Math.max(seq, parseInt(m[1], 10));
          added++;
        }
      }
      $('saveStatus').textContent = `${added}장 불러옴`;
    } catch (err) {
      $('saveStatus').textContent = '불러오기 오류: ' + err.message;
    }
    setBusy(false);
    render();
    if (added) $('saveStatus').textContent = `${added}장 불러옴 · 총 ${records.length}장`;
  });

  $('clear').addEventListener('click', () => {
    if (!records.length) return;
    if (!window.confirm(`찍은 사진 ${records.length}장을 모두 지울까요?`)) return;
    records.length = 0;
    dirty = false;
    render();
  });

  window.addEventListener('beforeunload', (e) => {
    if (dirty && records.length) { e.preventDefault(); e.returnValue = ''; }
  });

  // ---------------- 학습 곡선 그래프 (손실 / 정확도 두 칸) ----------------
  const SERIES = { train: { label: '학습', color: '#2a78d6' }, val: { label: '검증', color: '#eb6834' } };
  let history = [];
  let totalEpochs = 20;
  let hoverEpoch = null;

  function drawChart() {
    const cv = $('trainChart');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const Wc = cv.clientWidth || 760, Hc = cv.clientHeight || 170;
    if (cv.width !== Math.round(Wc * dpr)) { cv.width = Math.round(Wc * dpr); cv.height = Math.round(Hc * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, Wc, Hc);
    g.font = '12px system-ui, sans-serif';

    // 범례 (오른쪽 위)
    let lx = Wc - 12;
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (const key of ['val', 'train']) {
      const s = SERIES[key];
      g.fillStyle = '#374151';
      g.fillText(s.label, lx, 14);
      const tw = g.measureText(s.label).width;
      g.fillStyle = s.color;
      g.fillRect(lx - tw - 22, 12, 16, 3);
      lx -= tw + 34;
    }

    const panels = [
      { title: '손실 (낮을수록 좋음)', key: ['loss', 'valLoss'], max: Math.max(1.2, ...history.map((h) => Math.max(h.loss, h.valLoss || 0))), fmt: (v) => v.toFixed(1) },
      { title: '정확도', key: ['acc', 'valAcc'], max: 1, fmt: (v) => Math.round(v * 100) + '%' },
    ];
    const gap = 24, left = 40, right = 12, top = 32, bottom = 26;
    const pw = (Wc - left * 2 - gap - right) / 2;
    panels.forEach((P, pi) => {
      const x0 = left + pi * (pw + gap + left);
      const ph = Hc - top - bottom;
      g.textAlign = 'left';
      g.textBaseline = 'alphabetic';
      g.fillStyle = '#111827';
      g.font = '600 12px system-ui, sans-serif';
      g.fillText(P.title, x0, 14);
      g.font = '11px system-ui, sans-serif';
      // 격자와 축 눈금
      for (let i = 0; i <= 2; i++) {
        const v = (P.max * i) / 2;
        const y = top + ph - (v / P.max) * ph;
        g.strokeStyle = i === 0 ? '#c9c3b8' : '#efebe4';
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(x0, y + 0.5); g.lineTo(x0 + pw, y + 0.5); g.stroke();
        g.fillStyle = '#6b7280';
        g.textAlign = 'right';
        g.fillText(P.fmt(v), x0 - 6, y + 4);
      }
      g.textAlign = 'center';
      g.fillStyle = '#6b7280';
      g.fillText('epoch', x0 + pw / 2, Hc - 6);
      const E = Math.max(totalEpochs, history.length, 2);
      const X = (ep) => x0 + ((ep - 1) / (E - 1)) * pw;
      const Y = (v) => top + ph - (Math.min(v, P.max) / P.max) * ph;
      // 마우스를 올린 epoch 표시
      if (hoverEpoch && hoverEpoch <= history.length) {
        g.strokeStyle = '#d1d5db';
        g.beginPath(); g.moveTo(X(hoverEpoch) + 0.5, top); g.lineTo(X(hoverEpoch) + 0.5, top + ph); g.stroke();
      }
      ['train', 'val'].forEach((sk, si) => {
        const k = P.key[si];
        const pts = history.filter((h) => h[k] != null).map((h) => [X(h.epoch), Y(h[k])]);
        if (!pts.length) return;
        g.strokeStyle = SERIES[sk].color;
        g.lineWidth = 2;
        g.lineJoin = 'round';
        g.beginPath();
        pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
        g.stroke();
        const [lxp, lyp] = pts[pts.length - 1];
        g.fillStyle = SERIES[sk].color;
        g.beginPath(); g.arc(lxp, lyp, 4, 0, Math.PI * 2); g.fill();
        g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
        if (hoverEpoch && history[hoverEpoch - 1] && history[hoverEpoch - 1][k] != null) {
          g.fillStyle = SERIES[sk].color;
          g.beginPath(); g.arc(X(hoverEpoch), Y(history[hoverEpoch - 1][k]), 4, 0, Math.PI * 2); g.fill();
        }
      });
      P.x0 = x0; P.pw = pw;
    });
    drawChart.panels = panels;
  }

  $('trainChart').addEventListener('mousemove', (e) => {
    if (!history.length || !drawChart.panels) return;
    const r = e.target.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const P = drawChart.panels.find((p) => mx >= p.x0 - 10 && mx <= p.x0 + p.pw + 10) || drawChart.panels[0];
    const E = Math.max(totalEpochs, history.length, 2);
    const ep = Math.min(history.length, Math.max(1, Math.round(((mx - P.x0) / P.pw) * (E - 1)) + 1));
    hoverEpoch = ep;
    const h = history[ep - 1];
    $('chartHover').textContent = `epoch ${ep} · 학습 손실 ${h.loss.toFixed(3)} · 검증 손실 ${h.valLoss != null ? h.valLoss.toFixed(3) : '–'} · 학습 정확도 ${Math.round(h.acc * 100)}% · 검증 정확도 ${h.valAcc != null ? Math.round(h.valAcc * 100) + '%' : '–'}`;
    drawChart();
  });
  $('trainChart').addEventListener('mouseleave', () => { hoverEpoch = null; drawChart(); });
  window.addEventListener('resize', drawChart);
  drawChart();

  // ---------------- 브라우저 학습 ----------------
  let training = false;
  let stopReq = false;
  let trainedSpec = null;

  function confusionTable(conf) {
    let html = '<table class="confusion"><tr><th>정답 \\ 예측</th>' + CLASSES.map((c) => `<th>${SHORT[c]}</th>`).join('') + '</tr>';
    conf.forEach((row, i) => {
      html += `<tr><th>${SHORT[CLASSES[i]]}</th>` + row.map((v, j) => `<td class="${i === j ? 'hit' : ''}">${v}</td>`).join('') + '</tr>';
    });
    return html + '</table>';
  }

  $('stopBtn').addEventListener('click', () => { stopReq = true; $('trainStatus').textContent = '중지하는 중…'; });

  $('trainBtn').addEventListener('click', async () => {
    if (training || busy) return;
    const counts = CLASSES.map((c) => records.filter((r) => r.label === c).length);
    if (Math.min(...counts) < MIN_PER_CLASS) {
      $('trainStatus').textContent = `분류마다 최소 ${MIN_PER_CLASS}장이 필요합니다 (지금: 다묾 ${counts[0]} · 벌림 ${counts[1]} · 웃기 ${counts[2]})`;
      return;
    }
    training = true;
    stopReq = false;
    setBusy(true);
    $('stopBtn').disabled = false;
    $('doneBox').hidden = true;
    control.paused = true; // 학습하는 동안 실시간 예측은 잠시 멈춤
    const targetSteps = parseInt($('amount').value, 10);
    totalEpochs = 20; // 첫 진행 보고에서 실제 epoch 수로 바뀜
    history = [];
    hoverEpoch = null;
    drawChart();
    records.forEach((r) => { delete r.pred; delete r.isVal; });
    const samples = records.map((r) => ({ x: r.x, y: CLASSES.indexOf(r.label), ref: r }));
    let lastEpochSec = 0;
    try {
      const res = await window.MouthTrain.trainModel(samples, {
        targetSteps,
        seed: Math.floor(Math.random() * 100000),
        shouldStop: () => stopReq,
        onProgress: (p) => {
          totalEpochs = p.epochs;
          const frac = p.phase === 'batch' ? (p.epoch - 1 + p.step / p.steps) / p.epochs : p.epoch / p.epochs;
          $('trainBar').style.width = (frac * 100).toFixed(1) + '%';
          if (p.phase === 'epoch') {
            history = p.history.slice();
            lastEpochSec = p.seconds;
            const h = history[history.length - 1];
            const left = Math.round((p.seconds / p.epoch) * (p.epochs - p.epoch));
            $('trainStatus').textContent = `epoch ${p.epoch}/${p.epochs} · 학습 손실 ${h.loss.toFixed(3)} · 검증 정확도 ${h.valAcc != null ? Math.round(h.valAcc * 100) + '%' : '–'} · 남은 시간 약 ${left}초`;
            drawChart();
          } else if (!history.length) {
            $('trainStatus').textContent = `첫 epoch 학습 중… (${p.step}/${p.steps})`;
          }
        },
      });
      history = res.history.slice();
      drawChart();
      const fv = res.finalVal;
      const pct = fv ? Math.round((fv.correct / fv.total) * 100) : null;
      // 모든 사진을 학습된 모델로 다시 분류 → 틀린 사진은 썸네일에 빨간 표시
      const valRefs = new Set(res.val.map((s) => s.ref));
      for (const r of records) {
        const p = res.net.predict(r.x);
        r.pred = CLASSES[p.indexOf(Math.max(...p))];
        r.isVal = valRefs.has(r);
      }
      const per = CLASSES.map((c) => {
        const tr = records.filter((r) => r.label === c && !r.isVal), va = records.filter((r) => r.label === c && r.isVal);
        const ok = (a) => a.filter((r) => r.pred === r.label).length;
        return { c, trOk: ok(tr), trN: tr.length, vaOk: ok(va), vaN: va.length };
      });
      const weak = per.filter((q) => q.trN && q.trOk / q.trN < 0.9).map((q) => NAMES[q.c].title);
      const nWrong = records.filter((r) => r.pred !== r.label).length;
      $('perClass').innerHTML =
        per.map((q) => `${NAMES[q.c].title}: 학습 ${q.trOk}/${q.trN} · 검증 ${q.vaOk}/${q.vaN}`).join('<br>') +
        (nWrong ? `<br>틀린 사진 ${nWrong}장은 아래 썸네일 맨 앞에 빨간 테두리로 표시했습니다. 입모양이 애매한 사진이면 지우고 다시 학습하세요.` : '<br>모든 사진을 맞혔습니다.') +
        (weak.length ? `<br><b>${weak.join(', ')}</b>을(를) 아직 잘 못 배웠어요. 학습량을 '오래'로 하거나 그 입모양 사진을 더 찍어 주세요.` : '');
      render();
      const note = `브라우저 학습 · 사진 ${samples.length}장` + (fv ? ` · 검증 ${fv.correct}/${fv.total}` : '');
      trainedSpec = res.net.toSpec(CLASSES, note);
      const model = K.makeModel(trainedSpec);
      control.setModel(model);
      if (!bars) bars = K.buildBars($('bars'), control.classes);
      $('barsHint').textContent = '방금 학습한 모델의 실시간 예측입니다. 입모양을 바꿔 보며 확인하세요.';
      $('trainStatus').textContent = (res.stopped ? `중간에 멈췄습니다 (${res.history.length} epoch` : `학습 완료 (${res.epochs} epoch`) +
        `, 가중치 업데이트 ${res.steps}번, ${Math.round(res.seconds || lastEpochSec)}초).`;
      $('doneText').innerHTML = fv
        ? `검증 사진 <b>${fv.correct}/${fv.total}장</b> 정답 (${pct}%) · 검증 손실이 가장 낮았던 epoch ${res.best ? res.best.epoch : '-'}의 가중치를 씁니다.<br>` +
          '<span class="hint">같은 자리·같은 조명에서 찍은 사진이라 실제 게임보다 높게 나올 수 있어요. 오른쪽 위 확률 막대로 실제 반응을 확인하세요.</span>'
        : '학습이 끝났습니다.';
      $('confusion').innerHTML = fv ? confusionTable(fv.confusion) : '';
      $('doneBox').hidden = false;
      $('trainBar').style.width = '100%';
    } catch (err) {
      $('trainStatus').textContent = '학습 오류: ' + err.message;
    } finally {
      training = false;
      control.paused = false;
      $('stopBtn').disabled = true;
      setBusy(false);
    }
  });

  $('useModel').addEventListener('click', () => {
    if (!trainedSpec) return;
    if (!K.saveLocalModel(trainedSpec)) {
      $('doneText').textContent = '이 브라우저에 모델을 저장하지 못했습니다. mouth_model.js를 내려받아 model 폴더에 넣으세요.';
      return;
    }
    settings.preferFile = false;
    K.saveSettings(settings);
    // 새 탭에서 게임을 열어 이 탭의 사진은 그대로 남겨 둠 (다시 학습할 때 사용)
    const w = window.open('index.html', 'mouthRunnerGame');
    if (!w) location.href = 'index.html';
  });

  $('downloadModel').addEventListener('click', () => {
    if (trainedSpec) downloadText(K.modelToScript(trainedSpec), 'mouth_model.js', 'text/javascript');
  });

  // ---------------- 화면 갱신 ----------------
  let lastGrab = 0;
  function frame(now) {
    if (control.model && !control.paused) {
      if (control.update(now) && bars) K.updateBars(bars, control.classes, control.probs, control.command);
    } else if (cam.ready && !busy && now - lastGrab > 100) {
      cam.grab(); // 모델이 없어도 "저장될 64×64" 미리보기는 갱신
      lastGrab = now;
    }
    cam.drawPreview();
    if (overlay && cam.ready) {
      const c = $('preview'), g = c.getContext('2d');
      g.fillStyle = 'rgba(0,0,0,0.5)';
      g.fillRect(0, 0, c.width, 40);
      g.fillStyle = '#fff';
      g.font = '700 18px system-ui, sans-serif';
      g.textAlign = 'center';
      g.fillText(overlay, c.width / 2, 27);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  window.__collect = { records, getSpec: () => trainedSpec };
})();

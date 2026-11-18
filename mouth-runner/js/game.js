/*
 * game.js — 입모양으로 조종하는 횡스크롤 러닝 게임
 *   입 다물기(closed)     = 달리기, 1초 유지하면 시작/재시작
 *   입 벌리기(open)       = 점프
 *   이 보이게 웃기(teeth) = 수그리기 (공중에서는 빨리 떨어지기)
 */
(function () {
  'use strict';
  const K = window.MouthKit;

  // ---------------- 화면 / 물리 상수 ----------------
  const W = 900, H = 300, GROUND = 250;
  const GRAVITY = 2300;        // px/s²
  const JUMP_V = 860;          // 점프 속도 → 최고 높이 약 160px, 체공 약 0.75초
  const FAST_FALL = 3;         // 공중에서 수그리면 중력 3배
  const START_SPEED = 300;     // px/s
  const ACCEL = 6;             // 1초마다 빨라지는 정도
  const MAX_SPEED = 640;
  const START_HOLD = 1.0;      // 입을 몇 초 다물면 시작할지
  const BEST_KEY = 'mouthRunner.best';

  // ---------------- 캐릭터 그림 (assets/knight.png, 16×24 칸 9개) ----------------
  const SPRITE = new Image();
  SPRITE.src = 'assets/knight.png';
  const PX = 3;                       // 픽셀 하나를 3배로 키워 그림
  const CELL_W = 16, CELL_H = 24;
  const F = { JUMP: 0, IDLE: [1, 2], LAND: 3, RUN: [4, 5, 6, 7], DUCK: 8 };
  // 충돌 판정 상자 (투구 몸통 기준, 깃털 끝은 제외)
  const STAND = { w: 12 * PX, h: 18 * PX };
  const DUCK = { w: 12 * PX, h: 9 * PX };
  const BEE_BOTTOM = 40;              // 벌의 아래쪽이 땅에서 40px 위 → 수그리면 통과

  // ---------------- DOM ----------------
  const $ = (id) => document.getElementById(id);
  const canvas = $('game');
  const ctx = canvas.getContext('2d');
  const settings = K.loadSettings();
  const loaded = K.loadModel(settings.preferFile);
  const model = loaded.model;
  const cam = new K.MouthCam({ preview: $('preview'), aiView: $('aiview'), settings });
  const control = new K.MouthControl(cam, model, settings);
  const bars = K.buildBars($('bars'), control.classes);

  $('modelStatus').textContent = loaded.message;
  $('modelStatus').classList.toggle('warn', !loaded.ok);

  // 모델 바꾸기 버튼
  (function modelButtons() {
    const box = $('modelActions');
    const add = (label, fn, cls) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      if (cls) b.className = cls;
      b.addEventListener('click', fn);
      box.appendChild(b);
    };
    if (loaded.source === 'local' && loaded.hasFile) {
      add('사이트 모델 파일 쓰기', () => { settings.preferFile = true; K.saveSettings(settings); location.reload(); });
    }
    if (loaded.source === 'file' && loaded.hasLocal) {
      add('내 브라우저 모델 쓰기', () => { settings.preferFile = false; K.saveSettings(settings); location.reload(); });
    }
    if (loaded.hasLocal) {
      add('브라우저 모델 지우기', () => {
        if (window.confirm('이 브라우저에 저장된 학습 모델을 지울까요?')) { K.clearLocalModel(); location.reload(); }
      });
    }
    const a = document.createElement('a');
    a.href = 'collect.html';
    a.className = 'button-link' + (model ? '' : ' primary');
    a.textContent = model ? '다시 학습하기' : '내 입모양 학습하러 가기';
    box.appendChild(a);
  })();

  // "작동 원리" 카드에 실제 모델 구조 표시
  if (model) {
    const L = model.spec.layers;
    const params = L.reduce((sum, l) => sum + (l.weight_shapes || []).reduce((s, sh) => s + sh.reduce((a, b) => a * b, 1), 0), 0);
    const nConv = L.filter((l) => l.type === 'Conv2D').length;
    const nDense = L.filter((l) => l.type === 'Dense').length;
    $('archShort').textContent = `합성곱 ${nConv}층 + 완전연결 ${nDense}층, 가중치 ${params.toLocaleString()}개 → 3가지 확률`;
    $('archDetail').textContent = '모델 구조: ' + L.map((l) => {
      if (l.type === 'Conv2D') return `Conv2D(${l.filters}, ${l.kernel_size.join('×')}, ${l.activation})`;
      if (l.type === 'Dense') return `Dense(${l.units}, ${l.activation})`;
      if (l.type.endsWith('Pooling2D')) return `${l.type.replace('Pooling2D', 'Pool')}(${l.pool_size.join('×')})`;
      return l.type;
    }).join(' → ');
  }

  function bindSlider(id, key, fmt) {
    const input = $(id), out = $(id + 'Out');
    input.value = settings[key];
    out.textContent = fmt(settings[key]);
    input.addEventListener('input', () => {
      settings[key] = parseFloat(input.value);
      out.textContent = fmt(settings[key]);
      K.saveSettings(settings);
    });
  }
  bindSlider('boxSize', 'size', (v) => Math.round(v * 100) + '%');
  bindSlider('threshold', 'threshold', (v) => Math.round(v * 100) + '%');

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener('resize', resize);

  // ---------------- 웹캠 ----------------
  let camMessage = '웹캠 연결 중…';
  async function startCamera() {
    $('camStatus').textContent = '웹캠 연결 중…';
    $('camStatus').classList.remove('warn');
    try {
      await cam.start();
      camMessage = '';
      $('camStatus').textContent = `웹캠 연결됨 (${cam.video.videoWidth}×${cam.video.videoHeight})`;
    } catch (e) {
      camMessage = '웹캠을 쓸 수 없어요';
      const why = e && e.name === 'NotAllowedError'
        ? '카메라 권한이 거부되었습니다. 주소창 왼쪽 아이콘에서 카메라를 허용하고 다시 연결하세요.'
        : e && e.name === 'NotFoundError' ? '연결된 카메라를 찾지 못했습니다.'
          : e && e.name === 'NotReadableError' ? '다른 프로그램(줌, 카메라 앱 등)이 웹캠을 쓰고 있습니다. 닫고 다시 연결하세요.'
            : (e && e.message) || String(e);
      $('camStatus').textContent = '웹캠 오류: ' + why;
      $('camStatus').classList.add('warn');
    }
  }
  $('camBtn').addEventListener('click', startCamera);
  startCamera();

  // ---------------- 키보드 / 터치 ----------------
  const keys = { jump: false, duck: false };
  let startPressed = false;
  window.addEventListener('keydown', (e) => {
    if (e.target && e.target.tagName === 'INPUT') return;
    if (['Space', 'ArrowUp', 'KeyW'].includes(e.code)) { keys.jump = true; startPressed = true; e.preventDefault(); }
    if (['ArrowDown', 'KeyS'].includes(e.code)) { keys.duck = true; e.preventDefault(); }
    if (e.code === 'Enter') startPressed = true;
  });
  window.addEventListener('keyup', (e) => {
    if (['Space', 'ArrowUp', 'KeyW'].includes(e.code)) keys.jump = false;
    if (['ArrowDown', 'KeyS'].includes(e.code)) keys.duck = false;
  });
  canvas.addEventListener('pointerdown', () => { startPressed = true; keys.jump = true; });
  window.addEventListener('pointerup', () => { keys.jump = false; });

  function readInput() {
    const mouthOn = !!model && cam.ready;
    const cmd = mouthOn ? control.command : 'unsure';
    return {
      mouthOn,
      cmd,
      jump: keys.jump || cmd === 'open',
      duck: keys.duck || cmd === 'teeth',
      closed: cmd === 'closed',
      start: startPressed,
    };
  }

  // ---------------- 게임 상태 ----------------
  function loadBest() {
    try { return parseInt(localStorage.getItem(BEST_KEY) || '0', 10) || 0; } catch (e) { return 0; }
  }
  function saveBest(v) {
    try { localStorage.setItem(BEST_KEY, String(v)); } catch (e) { /* 무시 */ }
  }

  const game = {
    state: 'ready', // ready → play → over → (다시) play
    time: 0, speed: START_SPEED, dist: 0, score: 0, best: loadBest(),
    obstacles: [], sinceSpawn: 0, nextGap: 500,
    hold: 0, overTime: 0, shake: 0,
    player: { x: 80, y: 0, vy: 0, duck: false, airDuck: false, onGround: true, run: 0, idle: 0, land: 0 },
    clouds: Array.from({ length: 5 }, (_, i) => ({ x: i * 200 + Math.random() * 120, y: 30 + Math.random() * 70, s: 0.6 + Math.random() * 0.6 })),
    pebbles: Array.from({ length: 28 }, () => ({ x: Math.random() * W, y: GROUND + 6 + Math.random() * 30, w: 2 + Math.random() * 5 })),
  };

  function resetRun() {
    Object.assign(game, { state: 'play', time: 0, speed: START_SPEED, dist: 0, score: 0, obstacles: [], sinceSpawn: 0, nextGap: 520, hold: 0, shake: 0 });
    Object.assign(game.player, { y: 0, vy: 0, duck: false, airDuck: false, onGround: true, land: 0 });
  }

  function playerBox() {
    const p = game.player;
    const s = p.duck ? DUCK : STAND;
    return { x: p.x, y: GROUND - p.y - s.h, w: s.w, h: s.h };
  }

  function spawnObstacle() {
    const allowBee = game.score >= 120;
    if (allowBee && Math.random() < 0.35) {
      // 머리 높이로 날아오는 벌: 수그리면 아래로 지나감 (타이밍 맞춰 점프해도 됨)
      game.obstacles.push({ type: 'bee', x: W + 20, y: GROUND - BEE_BOTTOM - 26, w: 40, h: 26, phase: Math.random() * 6 });
    } else {
      const n = game.speed > 420 && Math.random() < 0.3 ? 3 : Math.random() < 0.45 ? 2 : 1;
      const h = 34 + Math.random() * 12;
      game.obstacles.push({ type: 'cones', n, x: W + 20, y: GROUND - h, w: n * 24 + (n - 1) * 4, h });
    }
    // 반응할 시간이 생기도록 속도에 비례해 간격을 둠
    const minGap = game.speed * 0.85 + 180;
    game.nextGap = minGap + Math.random() * minGap * 0.7;
    game.sinceSpawn = 0;
  }

  function overlap(a, b, inset) {
    return a.x + inset < b.x + b.w - inset && a.x + a.w - inset > b.x + inset &&
           a.y + inset < b.y + b.h - inset && a.y + a.h - inset > b.y + inset;
  }

  // ---------------- 갱신 ----------------
  function step(dt, input) {
    const bgSpeed = game.state === 'play' ? game.speed : 90;
    for (const c of game.clouds) {
      c.x -= bgSpeed * 0.12 * c.s * dt;
      if (c.x < -100) { c.x = W + Math.random() * 100; c.y = 30 + Math.random() * 70; }
    }
    for (const p of game.pebbles) {
      p.x -= bgSpeed * dt;
      if (p.x < -10) p.x += W + 20;
    }
    const p = game.player;
    p.idle += dt * 3;
    if (game.state === 'play') p.run += dt * (game.speed / 30);
    if (p.land > 0) p.land -= dt;

    if (game.state !== 'play') {
      if (game.state === 'over') game.overTime += dt;
      const canStart = game.state === 'ready' || game.overTime > 0.8;
      if (!canStart) return;
      game.hold = input.closed ? game.hold + dt : 0;
      if (input.start || game.hold >= START_HOLD) resetRun();
      return;
    }

    game.time += dt;
    game.speed = Math.min(MAX_SPEED, START_SPEED + ACCEL * game.time);
    game.dist += game.speed * dt;
    game.score = Math.floor(game.dist / 12);

    // 플레이어
    if (p.onGround) {
      p.duck = input.duck;
      if (input.jump && !input.duck) {
        p.vy = JUMP_V;
        p.onGround = false;
        p.duck = false;
      }
    }
    if (!p.onGround) {
      p.airDuck = input.duck;
      p.vy -= (input.duck ? GRAVITY * FAST_FALL : GRAVITY) * dt;
      p.y += p.vy * dt;
      if (p.y <= 0) {
        p.y = 0; p.vy = 0; p.onGround = true; p.duck = input.duck; p.airDuck = false;
        p.land = 0.09; // 착지 순간 잠깐 웅크린 그림
      }
    }

    // 장애물
    game.sinceSpawn += game.speed * dt;
    if (game.sinceSpawn >= game.nextGap) spawnObstacle();
    for (const o of game.obstacles) {
      o.x -= game.speed * (o.type === 'bee' ? 1.08 : 1) * dt;
      if (o.type === 'bee') o.phase += dt * 18;
    }
    game.obstacles = game.obstacles.filter((o) => o.x + o.w > -20);

    // 충돌
    const pb = playerBox();
    for (const o of game.obstacles) {
      if (overlap(pb, o, 5)) {
        game.state = 'over';
        game.overTime = 0;
        game.hold = 0;
        game.shake = 0.25;
        if (game.score > game.best) { game.best = game.score; saveBest(game.best); }
        break;
      }
    }
  }

  // ---------------- 그리기 ----------------
  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawBackground() {
    const sky = ctx.createLinearGradient(0, 0, 0, GROUND);
    sky.addColorStop(0, '#dff1ff');
    sky.addColorStop(1, '#fdf7ec');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#ffffff';
    for (const c of game.clouds) {
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, 34 * c.s, 12 * c.s, 0, 0, Math.PI * 2);
      ctx.ellipse(c.x + 18 * c.s, c.y - 8 * c.s, 20 * c.s, 13 * c.s, 0, 0, Math.PI * 2);
      ctx.ellipse(c.x - 16 * c.s, c.y - 4 * c.s, 16 * c.s, 10 * c.s, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = '#efe3cf';
    ctx.fillRect(0, GROUND, W, H - GROUND);
    ctx.fillStyle = '#7c6a55';
    ctx.fillRect(0, GROUND, W, 3);
    ctx.fillStyle = '#c9b597';
    for (const p of game.pebbles) ctx.fillRect(p.x, p.y, p.w, 2);
  }

  function playerFrame() {
    const p = game.player;
    if (game.state === 'over') return F.LAND;
    if (game.state !== 'play') return F.IDLE[Math.floor(p.idle) % F.IDLE.length];
    if (!p.onGround) return p.airDuck ? F.DUCK : F.JUMP;
    if (p.duck) return F.DUCK;
    if (p.land > 0) return F.LAND;
    return F.RUN[Math.floor(p.run) % F.RUN.length];
  }

  function drawPlayer() {
    const p = game.player;
    // 그림자 (높이 뛸수록 작아짐)
    const shrink = Math.max(0.35, 1 - p.y / 220);
    ctx.fillStyle = 'rgba(80,60,40,0.18)';
    ctx.beginPath();
    ctx.ellipse(p.x + STAND.w / 2, GROUND + 2, 20 * shrink, 4 * shrink, 0, 0, Math.PI * 2);
    ctx.fill();

    const x = Math.round(p.x - 3 * PX);               // 그림 칸의 3번째 열이 투구 왼쪽 테두리
    const y = Math.round(GROUND - p.y - CELL_H * PX); // 그림 칸의 맨 아래가 발바닥
    if (SPRITE.complete && SPRITE.naturalWidth) {
      ctx.imageSmoothingEnabled = false;              // 픽셀이 번지지 않게
      ctx.drawImage(SPRITE, playerFrame() * CELL_W, 0, CELL_W, CELL_H, x, y, CELL_W * PX, CELL_H * PX);
    } else {
      const b = playerBox();
      ctx.fillStyle = '#8bd4cd';
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }
  }

  function drawCone(x, y, w, h) {
    ctx.fillStyle = '#5b4636';
    roundRect(x - 2, y + h - 5, w + 4, 5, 2); ctx.fill();
    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.moveTo(x + w / 2 - 3, y);
    ctx.lineTo(x + w / 2 + 3, y);
    ctx.lineTo(x + w, y + h - 5);
    ctx.lineTo(x, y + h - 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    const sy = y + h * 0.42;
    const t1 = (sy - y) / (h - 5), t2 = (sy + 6 - y) / (h - 5);
    ctx.beginPath();
    ctx.moveTo(x + w / 2 - 3 - (w / 2 - 3) * t1, sy);
    ctx.lineTo(x + w / 2 + 3 + (w / 2 - 3) * t1, sy);
    ctx.lineTo(x + w / 2 + 3 + (w / 2 - 3) * t2, sy + 6);
    ctx.lineTo(x + w / 2 - 3 - (w / 2 - 3) * t2, sy + 6);
    ctx.closePath();
    ctx.fill();
  }

  function drawBee(o) {
    const cx = o.x + o.w / 2, cy = o.y + o.h / 2 + Math.sin(o.phase * 0.35) * 2;
    ctx.fillStyle = 'rgba(200,230,255,0.85)';
    const flap = Math.sin(o.phase) * 5;
    ctx.beginPath(); ctx.ellipse(cx + 2, cy - 12 - flap * 0.4, 8, 6 + flap * 0.3, -0.4, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx + 10, cy - 10 + flap * 0.4, 7, 5, 0.3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#facc15';
    ctx.strokeStyle = '#92400e';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.ellipse(cx, cy, o.w / 2, o.h / 2 - 2, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.save();
    ctx.beginPath(); ctx.ellipse(cx, cy, o.w / 2, o.h / 2 - 2, 0, 0, Math.PI * 2); ctx.clip();
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(cx - 2, cy - 15, 5, 30);
    ctx.fillRect(cx + 8, cy - 15, 5, 30);
    ctx.restore();
    ctx.fillStyle = '#1f2937';
    ctx.beginPath(); ctx.arc(o.x + 6, cy - 2, 2.2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.moveTo(o.x + o.w, cy); ctx.lineTo(o.x + o.w + 7, cy + 1); ctx.lineTo(o.x + o.w, cy + 4); ctx.fill();
  }

  function drawObstacles() {
    for (const o of game.obstacles) {
      if (o.type === 'bee') drawBee(o);
      else for (let i = 0; i < o.n; i++) drawCone(o.x + i * 28, o.y, 24, o.h);
    }
  }

  const CMD_TEXT = { closed: '달리기', open: '점프', teeth: '수그리기', unsure: '판단 보류' };
  const CMD_COLOR = { closed: '#3b82f6', open: '#16a34a', teeth: '#f59e0b', unsure: '#9ca3af' };

  function drawHUD(input) {
    ctx.font = '600 16px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('최고 ' + String(game.best).padStart(5, '0'), W - 120, 28);
    ctx.fillStyle = '#111827';
    ctx.fillText('점수 ' + String(game.score).padStart(5, '0'), W - 16, 28);

    ctx.textAlign = 'left';
    ctx.font = '600 13px system-ui, sans-serif';
    const label = input.mouthOn ? '입 인식: ' + CMD_TEXT[input.cmd] : '키보드 모드';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = input.mouthOn ? CMD_COLOR[input.cmd] : '#6b7280';
    roundRect(12, 12, tw + 20, 24, 12); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(label, 22, 29);
  }

  function centerPanel(lines, progress) {
    const w = 440, h = progress == null ? 104 : 124;
    const x = (W - w) / 2, y = 54;
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    roundRect(x, y, w, h, 14); ctx.fill();
    ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1; ctx.stroke();
    ctx.textAlign = 'center';
    lines.forEach((ln, i) => {
      ctx.font = i === 0 ? '800 26px system-ui, sans-serif' : '500 15px system-ui, sans-serif';
      ctx.fillStyle = i === 0 ? '#111827' : '#374151';
      ctx.fillText(ln, W / 2, y + 38 + i * 26);
    });
    if (progress != null) {
      ctx.fillStyle = '#e5e7eb';
      roundRect(x + 60, y + h - 26, w - 120, 10, 5); ctx.fill();
      ctx.fillStyle = '#3b82f6';
      roundRect(x + 60, y + h - 26, Math.max(10, (w - 120) * progress), 10, 5); ctx.fill();
    }
  }

  function draw(input) {
    ctx.save();
    if (game.shake > 0) {
      ctx.translate((Math.random() - 0.5) * 32 * game.shake, (Math.random() - 0.5) * 32 * game.shake);
      game.shake = Math.max(0, game.shake - 1 / 60);
    }
    drawBackground();
    drawObstacles();
    drawPlayer();
    drawHUD(input);
    ctx.restore();

    const startHint = input.mouthOn ? '입을 다물고 1초 기다리면 시작 (또는 Enter)' : 'Enter 또는 Space를 누르면 시작';
    if (game.state === 'ready') {
      centerPanel(['입으로 달려라!', startHint], input.mouthOn ? Math.min(1, game.hold / START_HOLD) : null);
    } else if (game.state === 'over') {
      const lines = ['게임 오버', `점수 ${game.score}  ·  최고 ${game.best}`];
      if (game.overTime > 0.8) lines.push(input.mouthOn ? '입을 1초 다물면 다시 시작 (또는 Enter)' : 'Enter 또는 Space로 다시 시작');
      centerPanel(lines, input.mouthOn && game.overTime > 0.8 ? Math.min(1, game.hold / START_HOLD) : null);
    }
  }

  // ---------------- 메인 루프 ----------------
  let last = performance.now();
  let lastGrab = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!model && cam.ready && now - lastGrab > 100) {
      cam.grab(); // 모델이 없어도 "AI가 보는 64×64" 미리보기는 갱신 (박스 맞추기용)
      lastGrab = now;
    }
    if (control.update(now)) {
      K.updateBars(bars, control.classes, control.probs, control.command);
      $('cmd').textContent = CMD_TEXT[control.command] || control.command;
      $('cmd').style.color = CMD_COLOR[control.command] || '#111827';
      $('infer').textContent = control.inferMs.toFixed(1) + ' ms';
    }
    cam.drawPreview(camMessage);
    const input = readInput();
    step(dt, input);
    draw(input);
    startPressed = false;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // 테스트·발표 때 상태 확인용
  window.__mouthRunner = { game, control, cam, loaded };
})();

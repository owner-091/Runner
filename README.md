# Mouth Runner

입모양 CNN 프로그램과 연결할 횡스크롤 아케이드 게임의 1차 프로토타입입니다.
현재 게임만 구현되어 있으며 CNN 모델, 웹캠 촬영, 입모양 인식은 포함되지 않습니다.

## 바로 실행
압축을 풀고 index.html을 Chrome, Edge, Safari 등 최신 브라우저로 엽니다.
외부 라이브러리·서버·설치·빌드 과정이 없습니다.

| 입모양 | 동작 | 시험용 입력 |
|---|---|---|
| 입 닫기 | 시작·기본 자세·재시작 | Enter 또는 입 닫기 버튼 |
| 입 크게 열기 | 점프 | Space / ↑ 또는 크게 열기 버튼 |
| 이빨이 보이도록 입꼬리 올리기 | 수그리기 | ↓ 또는 웃기 버튼을 누르고 유지 |

캐릭터는 자동으로 달립니다. 지상 상자는 점프, 공중 드론은 수그리기로 피합니다.
접촉하면 게임이 끝납니다. 거리에 따라 점수가 오르며 속도는 제한값까지 서서히 증가합니다.
최고 점수는 브라우저에 저장합니다. 저장이 차단되어도 게임은 실행됩니다.
P / 일시정지 버튼으로 멈춥니다. 다른 탭으로 이동하거나 창이 포커스를 잃으면 자동으로 일시정지합니다.

## GitHub Pages 배포
1. GitHub에 로그인하고 새 공개 저장소를 만듭니다. 예: mouth-runner.
2. Add file → Upload files에서 이 폴더의 **내용물**을 업로드하고 Commit changes를 누릅니다.
   index.html이 저장소 첫 화면에 보여야 합니다. ZIP 자체나 바깥 폴더를 업로드하면 안 됩니다.
3. Settings → Pages → Build and deployment에서 Source를 Deploy from a branch로 선택합니다.
4. Branch를 main, 폴더를 /(root)로 선택하고 Save를 누릅니다.
5. 배포가 끝나면 같은 화면의 Visit site에서 실제 게임 주소를 확인합니다.
   보통 https://사용자명.github.io/mouth-runner/ 형식입니다. 이는 형식 예시이며 발급된 실제 링크가 아닙니다.

게임 소스는 owner-091/Runner 저장소에 업로드되었습니다. 실제 배포 여부와 주소는 저장소 Settings → Pages에서 확인하세요.
Pages 설정 공식 안내: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## CNN 연결 인터페이스
모델을 이 페이지에서 로드하고, 각 프레임의 분류 결과를 아래 함수로 전달합니다.

```js
// label: 'closed' | 'open' | 'smile', confidence: 0~1
window.MouthRunner.setPose(label, confidence);
// 예: 모델이 입 크게 열기로 판정한 프레임
window.MouthRunner.setPose('open', 0.95);
```

- 같은 페이지의 JavaScript에서 호출합니다. Colab에서 학습만 했다고 이 페이지에 자동 연결되지는 않습니다.
- 모델의 클래스 이름을 closed / open / smile로 변환해야 합니다.
- 이 함수는 매 추론 프레임마다 반복 호출합니다(권장: 초당 10회 이상).
- 신뢰도 0.8 이상인 동일 판정이 최소 120ms 동안 유지되어야 반영됩니다.
- 시작·게임오버 상태에서는 안정적인 closed 입력이 있어야 달리기를 시작합니다.
- open으로 바뀌는 순간 한 번만 점프합니다. 입을 계속 벌려도 착지 후 자동 재점프하지 않습니다.
- smile 유지 중에는 수그리며 closed로 돌아오면 일어납니다.
- AI 입력을 사용하기 시작한 후 유효 입력이 700ms 넘게 없으면 일시정지합니다.
- 일시정지 해제는 현재 화면의 계속 버튼으로 진행합니다.
- window.MouthRunner.pause()와 getState()도 제공합니다.
- 임계값은 아직 실제 데이터로 검증하지 않은 초기값입니다. 학습·시연 결과에 맞춰 조정하세요.

웹캠을 나중에 연결할 때는 GitHub Pages의 HTTPS 주소에서 카메라 권한을 허용하도록 구현합니다.

## 파일
- index.html: 화면, 게임 로직, 입력 인터페이스를 포함하는 단일 실행 파일
- .nojekyll: GitHub Pages의 정적 파일 처리용
- README.md: 실행·배포·AI 연결 안내

## 검증 범위
JavaScript 문법 및 시뮬레이션 검사: 시작 조건, 자동 이동, 점프, 점프 반복 방지,
지상 장애물 충돌·점프로 통과, 공중 장애물 충돌·수그리기로 통과, 재시작, 일시정지.
실제 브라우저의 화면 배치·터치와 실제 CNN 연결은 별도 확인이 필요합니다.

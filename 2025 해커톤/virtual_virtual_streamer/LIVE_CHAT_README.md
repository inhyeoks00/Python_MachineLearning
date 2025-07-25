# April AI - 라이브 채팅 기능

April AI가 유튜브 라이브 스트림의 채팅을 실시간으로 모니터링하고 응답하는 기능입니다.

## 🎯 주요 기능

- **실시간 채팅 수집**: 유튜브 라이브 스트림에서 실시간으로 댓글 수집
- **지능형 응답**: 10초마다 수집된 댓글들을 분석하여 자연스러운 응답 생성
- **TTS 음성 출력**: 에이프릴의 응답을 음성으로 출력
- **스마트 필터링**: 스팸이나 반복 메시지 자동 필터링
- **메모리 통합**: 중요한 댓글 내용을 코어 메모리에 저장

## 📦 설치

### 1. 필수 패키지 설치
```bash
# 자동 설치 스크립트 실행
./install_dependencies.sh

# 또는 수동 설치
pip install pytchat asyncio-throttle
```

### 2. API 키 설정
`app/config.py`에서 Gemini API 키가 설정되어 있는지 확인하세요.

## 🚀 사용법

### 라이브 채팅 모드
```bash
python main.py live <유튜브_비디오_ID> [응답간격초]
```

**예시:**
```bash
# 10초마다 응답 (기본값)
python main.py live dQw4w9WgXcQ

# 15초마다 응답
python main.py live dQw4w9WgXcQ 15
```

### 기타 모드
```bash
# 텍스트 채팅 모드
python main.py text

# 음성 대화 모드
python main.py voice
```

## 📋 라이브 채팅 동작 방식

1. **채팅 수집**: 설정된 간격(기본 10초)마다 새로운 댓글들을 수집
2. **필터링**: 스팸, 반복 메시지, 너무 적은 댓글 등을 필터링
3. **집계**: 수집된 댓글들을 하나의 맥락으로 정리
4. **응답 생성**: 에이프릴이 전체 댓글 분위기를 파악하여 자연스러운 응답 생성
5. **음성 출력**: TTS로 음성 출력
6. **기록**: 대화 내용을 로그에 저장

## 🎛️ 설정 옵션

### 응답 조건
- 최소 3개 이상의 댓글이 있어야 응답
- 댓글의 70% 이상이 유니크해야 함 (스팸 방지)

### 메모리 관리
- 최근 50개의 댓글만 버퍼에 유지
- 중요한 댓글 내용은 프로파일러를 통해 코어 메모리에 저장

## 💡 사용 팁

### 유튜브 비디오 ID 찾기
유튜브 URL에서 비디오 ID를 추출하세요:
- `https://www.youtube.com/watch?v=dQw4w9WgXcQ` → `dQw4w9WgXcQ`
- `https://youtu.be/dQw4w9WgXcQ` → `dQw4w9WgXcQ`

### 최적의 응답 간격
- **활발한 채팅**: 5-10초 간격 권장
- **느린 채팅**: 15-30초 간격 권장
- **테스트용**: 5초 간격으로 빠른 반응 확인

### 라이브 스트림 요구사항
- 유튜브 라이브 스트림이 활성화되어 있어야 함
- 채팅이 활성화되어 있어야 함
- 비공개 스트림의 경우 접근 권한 필요

## 🔧 문제 해결

### 자주 발생하는 오류

1. **"pytchat 라이브러리가 필요합니다"**
   ```bash
   pip install pytchat
   ```

2. **"비디오 ID가 설정되지 않았습니다"**
   - 올바른 유튜브 비디오 ID를 입력했는지 확인
   - 라이브 스트림이 활성화되어 있는지 확인

3. **"채팅 세션 시작 오류"**
   - 인터넷 연결 확인
   - 해당 스트림의 채팅이 활성화되어 있는지 확인

### 로그 확인
프로그램 실행 중 로그를 통해 상태를 확인할 수 있습니다:
- INFO: 일반적인 동작 상황
- ERROR: 오류 발생 상황
- DEBUG: 상세한 디버그 정보

## 📄 예시 사용 시나리오

### 1. 개인 스트리밍
```bash
# 내 라이브 스트림에서 15초마다 응답
python main.py live MY_STREAM_ID 15
```

### 2. 테스트 모드
```bash
# 빠른 테스트를 위해 5초마다 응답
python main.py live TEST_STREAM_ID 5
```

### 3. 관찰 모드
```bash
# 긴 간격으로 주요 반응만 체크
python main.py live STREAM_ID 30
```

## ⚠️ 주의사항

- 유튜브 라이브 스트림의 이용 약관을 준수하세요
- 과도한 API 호출을 피하기 위해 적절한 간격을 설정하세요
- 개인정보나 민감한 정보가 포함된 댓글에 주의하세요
- 봇 사용이 허용된 채널에서만 사용하세요

## 🔄 업데이트 예정

- [ ] 트위치 라이브 채팅 지원
- [ ] 디스코드 연동
- [ ] 감정 분석 기반 응답
- [ ] 커스텀 필터링 규칙
- [ ] 웹 대시보드 인터페이스

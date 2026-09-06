# 프로브 — "이 제품 뭐야" 에 VLM 이 답할 수 있는가

이 프로브 하나가 노선 전체를 판정한다. 결과가 안 나오면 **Mac 도 $99 도 사지 않는다.**

관련 결정: `storage/product-retrieval/STATUS.md` D-9(노선) · D-10(D-1 하이브리드)

## 왜 이걸 먼저 재나

```
클라우드 VLM (실제 이미지를 봄, 큰 모델)     ← 지금 재는 것 = 상한
        ↓ 이보다 잘 될 수 없다
온디바이스 VLM (텍스트만 넘김, 3B급)         ← 제품 목표 (D-8)
```

상한이 낮으면 온디바이스는 볼 것도 없다.

## 두 단계로 나눠 잰다

한 번의 그라운딩 호출로 재면 어느 쪽이 깨졌는지 모른다. 나눠야 정보량이 는다.

| | 질문 | 필요한 것 | 상태 |
|---|---|---|---|
| **Q-A** | 이미지에서 식별에 쓸 만한 텍스트를 뽑는가 | 이미지 입력만 | **무료로 가능** [확인] |
| **Q-B** | 그 텍스트로 실제 제품 페이지를 찾는가 | 웹 검색 | 결제 활성화 필요 |

이 분리는 우회가 아니라 **D-8/D-10 목표 아키텍처 그 자체**다 — 온디바이스 VLM 이 텍스트를 만들고, 검색은 별도 단계.

---

## 1. 실측된 환경 (2026-09-05)

```
모델        gemini-3.1-flash-lite   ← 무료 티어에서 동작 확인. 이미지·OCR·JSON 전부 OK
            gemini-2.5-*            404 "no longer available to new users"
            gemini-3.8-flash        429 (무료 티어 할당 없음)
            gemini-flash-latest     별칭이라 측정에 쓰지 말 것 — 모델이 밑에서 바뀐다
그라운딩     무료 티어에 없음 [확인: ai.google.dev/gemini-api/docs/pricing]
            5,000회/월 무료는 **결제 활성화 후**에만 적용. 이후 $14/1,000
의존성       없음. requests + PIL 만 사용
```

배관 검증 완료: 합성 이미지로 OCR 정확 · 제품 아닐 때 brand/model 을 `null` 로 반환(지어내지 않음) ·
JSON 파싱 · 토큰·지연 기록. `--list-models` 로 키 유효성과 모델 id 를 서버에 직접 확인.

## 2. 키

리포 밖에 둔다. 대화·커밋·로그 어디에도 남기지 않는다.

```bash
printf 'GEMINI_API_KEY=%s\n' '발급받은키' > ~/.config/product-retrieval/.env
chmod 600 ~/.config/product-retrieval/.env
```

스크립트 탐색 순서: 환경변수 → `~/.config/product-retrieval/.env` → 리포 `.env`(gitignore 됨).
키는 어떤 출력에도 나오지 않는다.

```bash
python3 probe/identify.py --list-models    # 키 유효성 + 쓸 수 있는 모델
```

## 3. 사진 준비 — 이게 가장 오래 걸린다

### ⚠️ 촬영 전에: 아이폰은 포맷을 바꾼다

**설정 > 카메라 > 포맷 > "높은 호환성"**

기본값 HEIC 는 이 환경에서 **읽히지 않는다.** pillow-heif 도 변환 도구(heif-convert·ImageMagick·ffmpeg)도
없다 [확인: 2026-09-06]. 찍은 뒤에 고치려면 번거로우니 촬영 전에 바꾼다.

### 무엇을 찍나

`probe/photos/` 에 **본인이 촬영한** 사진 20장. **타인이 등장하는 사진은 넣지 않는다** (D-10 하드룰).
이 디렉터리는 `.gitignore` 되어 리포에 올라가지 않는다.

| 구분 | 장수 | 조건 |
|---|---|---|
| `easy` | 10 | 정답을 아는 제품. 로고·태그가 보임. 정면·밝음 |
| `hard` | 10 | 정답은 알지만 어려움. 로고 안 보임 · 비스듬함 · 어두움 · 일부 가림 |

**정답을 아는 제품만 쓴다.** 본인이 산 물건, 영수증·주문내역이 있는 것.

### manifest 는 손으로 쓰지 않는다

```bash
python3 probe/prepare.py          # 사진 스캔 → manifest 뼈대 생성 (기존 항목은 보존)
#   ... 에디터로 difficulty · truth · manual_search 채우기 ...
python3 probe/prepare.py --check  # 검증. 문제 있으면 exit 1
```

`prepare.py` 가 잡아주는 것: 파일 없음 · id 중복 · bbox 이미지 밖 · difficulty 오타 ·
truth 누락 · manual_search 누락(주의) · easy/hard 표본 부족(주의) · HEIC(안내).

생성되는 형태:

```json
{
  "id": "p01",
  "file": "IMG_0042.jpg",
  "_size": [3024, 4032],
  "bbox": null,
  "difficulty": "easy",
  "truth": { "brand": "Uniqlo", "model": "U Crew Neck T-Shirt", "url": null },
  "manual_search": { "seconds": 95, "success": true }
}
```

| 필드 | 설명 |
|---|---|
| `_size` | 참고용. **EXIF 회전을 적용한 뒤** 크기다. bbox 는 이 좌표계로 적는다 |
| `bbox` | `[x, y, 너비, 높이]`. 제품만 감싸게. `null` 이면 전체 이미지 |
| `truth` | 정답. **identify.py 는 이 필드를 읽지 않는다.** 채점 때만 쓴다 |
| `manual_search` | 같은 사진을 **본인이 직접 검색**했을 때 걸린 시간과 성공 여부 |

`manual_search` 를 꼭 채운다. 모델이 60% 맞혀도 사람이 30초에 100% 찾으면 이식할 이유가 없다.

bbox 없이 재면 **분할이 기여하는가**는 측정되지 않는다 (2단계에서 잰다).

### EXIF 회전은 자동 적용된다

아이폰 사진은 회전이 EXIF 에만 있다. 적용하지 않으면 모델이 **옆으로 누운 사진**을 보고,
정확도가 떨어져도 원인을 알 수 없다. `identify.py` 와 `prepare.py` 가 둘 다 `exif_transpose` 를
적용한다 [확인: 800x400 + Orientation=6 → 400x800].

## 4. 실행

```bash
# Q-A (무료)
python3 probe/identify.py --model gemini-3.1-flash-lite --no-grounding
python3 probe/score.py run-<timestamp>.json

# Q-B (결제 활성화 후)
python3 probe/identify.py --model <grounding 되는 모델>
python3 probe/score.py run-<timestamp>.json
```

`score.py` 는 실행이 그라운딩을 썼는지 보고 **채점 기준을 자동으로 바꾼다.**

| | Q-A (그라운딩 OFF) | Q-B (그라운딩 ON) |
|---|---|---|
| 1 | 충분 — 브랜드+모델 또는 결정적 OCR | 동일 제품 — **근거 URL 확인됨** |
| 2 | 부분적 — 종류·속성만 | 비슷할 뿐 |
| 3 | 쓸모없음 — 틀렸거나 너무 일반적 | 근거 부족 |

Q-B 에서 **근거 URL 없이 댄 제품명은 통과가 아니다.** 지어낸 답이 통과하면 측정 전체가 무의미해진다.
score.py 가 이 경우를 자동 경고한다.

## 5. 사전등록 판정 기준 — ⚠️ 측정 전에 서명 필요

측정하고 나서 기준을 정하면 무슨 숫자가 나와도 성공이 된다. **돌리기 전에 정한다.**

```
Q-A 계속한다     easy 검색가능서술률 ≥ 70%  그리고  전체 ≥ 50%
    중단한다     전체 < 30%   → 이미지에서 단서를 못 뽑는다. Q-B 는 볼 것도 없다

Q-B 계속한다     easy 정확식별률 ≥ 50%  그리고  전체 ≥ 30%
    재설계한다   전체 10~30%  → 프롬프트·bbox·모델 바꿔 1회 재시도
    중단한다     전체 < 10%   또는  손검색 성공률보다 낮다
```

**[추정] 이며 사용자 서명 대상이다.** 근거 없는 값이지만, 목적은 "옳은 값"이 아니라
**사후 조정을 불가능하게 만드는 것**이다.

## 6. 이 프로브가 답하지 않는 것

- 온디바이스 모델의 정확도 (더 낮을 것 [추정])
- 실기기 지연·앱 용량 (Mac 필요)
- 자동 분할의 기여 (2단계. 수동 bbox 대비 비교로 잰다)
- 20장은 **착수 판단용 표본**이다. 정식 비교는 미사용 사진으로 따로 한다

# 백혈구 8종 분류와 설명가능성

현미경 혈액 도말 이미지를 백혈구 8종으로 분류하고, **모델이 이미지의 어디를 보고
그렇게 판단했는지** Grad-CAM 히트맵으로 보여준다. 2주 개인 프로젝트.

> ## ⚠ 교육 목적입니다. 임상 진단에 사용할 수 없습니다.
>
> 공개 데이터셋 하나로 학습했고 임상 검증을 거치지 않았습니다. 그럴 목적으로
> 만들지 않았습니다. 의료 판단에 사용하지 마세요.

---

## 무엇을 찾았나

정확도를 올리는 것이 아니라, **남은 오류가 무엇인지 아는 것**이 이 프로젝트의 결과다.

| | |
| --- | ---: |
| test macro F1 (EfficientNet-B0) | **0.9791** |
| 틀린 것 중 미성숙 과립구가 낀 것 | **83%** |
| **서로 다른 다섯 모델이 똑같이 틀린 장** | **26장** |
| 그게 우연히 겹칠 기댓값 | 0.00장 |

구조 3종(ResNet18 · ResNet50 · EfficientNet-B0) · 시드 2종 · 해상도 2종으로
학습한 다섯 모델이 **같은 26장**을 틀리고, 25장은 **같은 답으로**, 13장은
**다섯 다 90% 이상 확신**하며 틀린다.

**모델을 더해도 줄지 않는다** — ResNet18 셋만으로 40장, EfficientNet-B0 을
더해 27장, ResNet50 까지 더해 26장. 겹침이 사라지는 것이 아니라 단단한 핵에서
멈춘다. 학습의 우연으로는 설명되지 않는다 — 원인은 그 이미지들에 있다.

그 이미지들을 눈으로 보면 **핵이 분엽되지 않았다.** 성숙한 호중구는 핵이
여러 조각으로 나뉘는데, "neutrophil" 라벨이 붙은 이 장들은 아직 안 나뉜 모습이다.

자세한 내용은 [`docs/results.md`](docs/results.md), 발표 자료는
[`docs/slides.html`](docs/slides.html).

---

## 시작하기

### 설치

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

PyTorch 는 CUDA 빌드를 별도 인덱스에서 받아야 한다. GTX 1050 Ti(sm_61)에서
`torch 2.5.1+cu121` 로 확인했다.

```powershell
.\.venv\Scripts\pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

GPU 가 없어도 돌아간다 — 자동으로 CPU 로 넘어간다. 다만 학습은 몇 시간 걸린다.

### 전체 재현

```powershell
.\tools\run_all.ps1 -Quick     # 3에폭 · 파이프라인이 도는지만 확인
.\tools\run_all.ps1            # 20에폭 · 실제 결과
.\tools\run_all.ps1 -SkipTrain # 학습 건너뛰고 분석만 다시 (실측 2분 6초)
```

데이터 확인 → 모델 4개 학습 → 평가 → Grad-CAM → 오류 분석 → 발표 자료까지 간다.
**한 단계라도 실패하면 거기서 멈춘다** — 낡은 결과로 뒷 단계가 돌면 틀린 그림이
슬라이드에 실린다.

`-Quick` 은 완성된 산출물을 건드리지 않고 `docs/slides.quick.html` 에 따로 쓴다.

---

## 개별 명령

```powershell
# 학습 — runs/<이름>/ 에 설정·지표·곡선·체크포인트가 남는다
python train.py --epochs 20 --input-size 112 --name my_run

# 평가 — 학습과 분리돼 있어 30분짜리 학습을 다시 돌릴 필요가 없다
python evaluate.py --run my_run --save

# Grad-CAM — 클래스별 히트맵, 정오답 비교
python explain.py --run my_run

# 라벨 모호성 가설 검증 — 확률 분포로 P1~P4 를 판정한다
python analyze.py --run my_run --save

# 여러 모델이 같은 장을 틀리는가 (핵심 발견)
python tools/consensus_errors.py --runs my_run other_run --save runs/consensus.json

# 오분류를 정답·예측 클래스의 대표와 나란히 놓고 본다
python inspect_errors.py --run my_run --from-json runs/consensus.json

# 발표 자료 다시 빌드
python tools/extract_cells.py --run my_run
python tools/build_slides.py --run my_run
```

주요 옵션 — `--arch resnet18|resnet50|efficientnet_b0` · `--input-size` ·
`--batch-size` · `--seed` · `--cpu`. VRAM 이 모자라면 `--batch-size 16` 부터
줄인다 (안내 메시지가 알려준다).

---

## 구조

```
train.py              학습
evaluate.py           평가 — 지표, 혼동행렬, 성공 기준 판정
explain.py            Grad-CAM 히트맵
analyze.py            라벨 모호성 가설 검증 (S5)
inspect_errors.py     오분류 비교 격자 — 사람이 판단할 수 있게

src/
  data.py             BloodMNIST 로딩, 증강
  model.py            전이학습 모델 (ResNet18/50, EfficientNet-B0)
  engine.py           학습·추론 루프
  metrics.py          macro F1, 혼동행렬, 상위 혼동 쌍
  gradcam.py          Grad-CAM 구현
  ambiguity.py        가설 검증 지표 — 부트스트랩 신뢰구간 판정
  seeding.py          재현성 (S3)
  cli.py              공통 진입 처리 — 실패를 사람이 읽을 수 있게
  plots.py            그림 저장

tools/
  fetch_data.py       데이터 내려받기
  check_gpu.py        GPU 실동작 검증
  bench.py            병목 측정 — 데이터 로딩 vs GPU 연산
  consensus_errors.py 모델 간 공통 오분류
  extract_cells.py    발표용 세포 낱장
  build_slides.py     슬라이드 빌드
  run_all.ps1         전체 재현

docs/
  PLAN.md             2주 계획과 진행 체크리스트
  results.md          결과와 오류 분석
  experiments.md      실험 비교표
  slides.html         발표 자료 (17장)

tests/                단위 테스트 70개
```

---

## 데이터

**BloodMNIST** (MedMNIST v2) — 현미경 혈액 도말 이미지 17,092장, 8클래스,
28×28 픽셀. `tools/fetch_data.py` 가 자동으로 내려받는다.

| 라벨 | 세포 | | 라벨 | 세포 |
| ---: | --- | --- | ---: | --- |
| 0 | basophil (호염기구) | | 4 | lymphocyte (림프구) |
| 1 | eosinophil (호산구) | | 5 | monocyte (단핵구) |
| 2 | erythroblast (적아구) | | 6 | neutrophil (호중구) |
| 3 | immature granulocytes (미성숙 과립구) | | 7 | platelet (혈소판) |

**정확도가 아니라 macro F1 을 기준으로 삼는다.** 클래스가 2.7배 불균형이라
정확도는 다수 클래스만 맞혀도 높게 나온다.

---

## 성공 기준

| # | 기준 | 결과 | |
| --- | --- | ---: | --- |
| S1 | test macro F1 ≥ 0.96 | 0.9791 | 통과 |
| S2 | 최저 클래스 F1 ≥ 0.90 | 0.9540 | 통과 |
| S3 | 같은 시드 재현 차이 < 0.01 | 0.0000000 | 통과 |
| S4 | 8클래스 Grad-CAM 생성 | 생성 | 통과 |
| S5 | 혼동 상위 3쌍 원인 서술 | 서술 | 통과 |
| S6 | 1회 학습 30분 이내 | 11.9분 | 통과 |

S3 는 기준을 통과한 정도가 아니라 **20에폭 학습곡선까지 완전히 일치**한다.
다만 `--num-workers` 를 바꾸면 시드가 같아도 결과가 0.0083 어긋난다 — 시드
고정은 명령줄이 한 글자도 다르지 않을 때만 재현을 보장한다.

---

## 한계

- **임상에 쓸 수 없다.** 검증을 거치지 않았다.
- **28×28 픽셀.** 고해상도 원본을 확보하지 못했다. 지금의 "112px 입력"은
  28px 를 늘린 것이다.
- **데이터셋 하나.** 다른 병원·장비·염색 조건은 보장할 수 없다.
- **오분류가 83장뿐이라** 확신도 비교 3건(P1~P3)은 통계적으로 판정하지 못했다.
  "차이가 없다"가 아니라 **"이 표본으로는 말할 수 없다"** 이다.
- **다섯 모델이 ImageNet 사전학습을 공유한다.** 마지막 공통점이라 완전히
  배제하지 못했다.
- **라벨이 틀렸다고 단정하지 않는다.** 전문가 검토가 필요하다. 발표자는
  혈액학 전문가가 아니다.

---

## 출처

- 데이터: [MedMNIST v2](https://medmnist.com/) — Yang et al., *Scientific Data*, 2023
- 원 데이터: Acevedo et al., *Data in Brief*, 2020 (Hospital Clínic de Barcelona)
- 사전학습 가중치: torchvision ImageNet

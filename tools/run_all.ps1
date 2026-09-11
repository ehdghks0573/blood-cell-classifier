<#
.SYNOPSIS
    학습부터 발표 자료까지 한 번에 재현한다 (Day 11).

.DESCRIPTION
    다른 사람이 재현할 수 있어야 결과다. 이 스크립트 하나로 데이터 확인 →
    학습 → 평가 → 설명 → 오류 분석 → 슬라이드까지 간다.

    중간에 하나라도 실패하면 **거기서 멈춘다.** 실패를 지나쳐 계속 돌면
    앞 단계의 낡은 결과로 뒷 단계가 돌아가서, 틀린 그림이 슬라이드에 실린다.

.PARAMETER Quick
    3에폭으로 짧게 돌린다. 성능은 의미 없고 파이프라인이 도는지만 본다.
    코드를 고친 뒤 전체를 돌리기 전에 이걸 먼저 쓴다 — 40분 뒤에 오타를
    발견하는 것보다 4분 뒤에 발견하는 편이 낫다.

.PARAMETER SkipTrain
    학습을 건너뛰고 이미 있는 runs/ 로 분석만 다시 한다.

.PARAMETER Hires
    224px 원본을 받아 여섯 번째 모델을 학습하고, 공통 오분류가 해상도 탓인지
    가른다. 기본 경로에서 빼 둔 이유는 **1.5GB 를 내려받기 때문**이다. 그것이
    없는 사람도 나머지는 전부 재현할 수 있어야 한다.

.EXAMPLE
    .\tools\run_all.ps1 -Quick
    .\tools\run_all.ps1
    .\tools\run_all.ps1 -SkipTrain
    .\tools\run_all.ps1 -Hires
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$SkipTrain,
    [switch]$Hires
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "가상환경이 없습니다: $python" -ForegroundColor Red
    Write-Host "먼저: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$epochs = if ($Quick) { 3 } else { 20 }
$suffix = if ($Quick) { "_quick" } else { "" }

# 재현에 필요한 네 모델. 시드·해상도·구조를 다르게 두는 것이 핵심 발견의 근거다.
$runs = @(
    @{ name = "res112_resnet18$suffix";   args = @("--arch", "resnet18", "--input-size", "112", "--seed", "42") },
    @{ name = "res112_seed43$suffix";     args = @("--arch", "resnet18", "--input-size", "112", "--seed", "43") },
    @{ name = "effb0_112$suffix";         args = @("--arch", "efficientnet_b0", "--input-size", "112", "--seed", "42") },
    # 이름이 데이터 해상도(28px)를 가리켜 헷갈리지만, docs/experiments.md 와
    # results.md 가 이 이름으로 기록돼 있다. 기록과 어긋나게 두는 쪽이 더 나쁘다.
    @{ name = "baseline_resnet18_28px$suffix"; args = @("--arch", "resnet18", "--input-size", "224", "--seed", "42") }
)
$best = "effb0_112$suffix"   # 발표에 쓰는 모델

# 점검용으로 돌릴 때 완성된 산출물을 덮어쓰면 안 된다. 3에폭짜리 그림이
# 발표 자료에 실리는 사고는 발표장에서야 발견된다.
$consensusJson = "runs\consensus_errors$suffix.json"
$hiresRun      = "hires224_res18_112$suffix"
$hiresJson     = "runs\consensus_errors_hires$suffix.json"
$survivorsJson = "runs\hires_survivors$suffix.json"
$figuresDir    = if ($Quick) { "docs\figures_quick" } else { "docs\figures" }
$slidesOut     = if ($Quick) { "docs\slides.quick.html" } else { "docs\slides.html" }

$steps = 0
$started = Get-Date

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Title,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $script:steps++
    $n = "{0:d2}" -f $script:steps
    Write-Host ""
    Write-Host "── $n · $Title " -ForegroundColor Cyan -NoNewline
    Write-Host ("─" * [Math]::Max(1, 62 - $Title.Length)) -ForegroundColor DarkGray

    $t0 = Get-Date
    & $python @Arguments
    $code = $LASTEXITCODE
    $took = (Get-Date) - $t0

    if ($code -ne 0) {
        Write-Host ""
        Write-Host "실패: $Title (종료 코드 $code)" -ForegroundColor Red
        Write-Host "여기서 멈춥니다. 이 단계를 고치고 다시 실행하세요." -ForegroundColor Red
        exit $code
    }
    Write-Host ("   {0:mm\:ss} 걸림" -f $took) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "백혈구 8종 분류 — 전체 재현" -ForegroundColor White
if ($Quick) {
    Write-Host "빠른 점검 모드 · 에폭 $epochs · 성능은 의미 없음" -ForegroundColor Yellow
} else {
    Write-Host "전체 모드 · 에폭 $epochs · 1시간 30분쯤 걸립니다" -ForegroundColor Yellow
}

# ── 1. 코드가 성한가 ────────────────────────────────────────────────
# 학습부터 돌리면 40분 뒤에 오타를 발견한다. 테스트가 먼저다.
Invoke-Step "단위 테스트" @("-m", "pytest", "-q")

# ── 2. 데이터 ───────────────────────────────────────────────────────
if (-not (Test-Path (Join-Path $root "data\bloodmnist.npz"))) {
    Invoke-Step "데이터 내려받기" @("tools\fetch_data.py")
} else {
    Write-Host ""
    Write-Host "── 데이터 있음 (data\bloodmnist.npz) ─" -ForegroundColor DarkGray
}

# ── 2-b. 224px 원본 ────────────────────────────────────────────────
# 정렬 검증이 학습보다 먼저다. 인덱스가 어긋나 있으면 20분을 학습한 뒤에야
# "비교할 수 없다"를 알게 된다.
if ($Hires) {
    if (-not (Test-Path (Join-Path $root "data\bloodmnist_224.npz"))) {
        Invoke-Step "224px 원본 내려받기 (1.5GB)" @("tools\fetch_hires.py", "--size", "224")
    } else {
        Write-Host ""
        Write-Host "── 224px 원본 있음 (data\bloodmnist_224.npz) ─" -ForegroundColor DarkGray
    }
    Invoke-Step "해상도 간 인덱스 정렬 검증" @("tools\check_alignment.py")
}

# ── 3. 학습 ─────────────────────────────────────────────────────────
if ($SkipTrain) {
    Write-Host ""
    Write-Host "── 학습 건너뜀 (-SkipTrain) ─" -ForegroundColor DarkGray
} else {
    foreach ($r in $runs) {
        Invoke-Step "학습 · $($r.name)" (@("train.py", "--epochs", "$epochs", "--name", $r.name) + $r.args)
    }
    if ($Hires) {
        # 바꾸는 것은 원본 해상도 하나뿐이다. 입력 크기도 구조도 시드도
        # res112_resnet18 과 같아야 둘을 견줄 수 있다.
        Invoke-Step "학습 · $hiresRun (224px 원본)" @("train.py", "--epochs", "$epochs",
            "--name", $hiresRun, "--arch", "resnet18", "--size", "224",
            "--input-size", "112", "--seed", "42")
    }
}

# ── 4. 평가와 설명 ──────────────────────────────────────────────────
Invoke-Step "평가 · $best"        @("evaluate.py", "--run", $best, "--save")
Invoke-Step "Grad-CAM · $best"    @("explain.py", "--run", $best)

# ── 5. 오류 분석 ────────────────────────────────────────────────────
Invoke-Step "라벨 모호성 검증"    @("analyze.py", "--run", $best, "--save")

$runNames = $runs | ForEach-Object { $_.name }
Invoke-Step "모델 간 공통 오분류" (@("tools\consensus_errors.py", "--runs") + $runNames +
                                   @("--save", $consensusJson))

Invoke-Step "공통 오분류 비교 격자" @("inspect_errors.py", "--run", $best,
                                      "--from-json", $consensusJson,
                                      "--pairs", "3", "--per-pair", "5", "--refs", "4")

if ($Hires) {
    # 모델마다 자기 원본 해상도로 잰다. 28px 로 배운 모델에 224px 를 먹이면
    # 그 모델이 본 적 없는 선명도가 들어가 비교가 기운다.
    $hiresSizes = @($runNames | ForEach-Object { "28" }) + @("224")
    Invoke-Step "고해상도까지 넣은 공통 오분류" (@("tools\consensus_errors.py", "--runs") +
                                                 $runNames + @($hiresRun, "--size") + $hiresSizes +
                                                 @("--save", $hiresJson))
    Invoke-Step "그 26장이 살아남았는가" @("tools\survivors.py",
                                           "--before", $consensusJson,
                                           "--after", $hiresJson,
                                           "--save", $survivorsJson)
}

# ── 6. 발표 자료 ────────────────────────────────────────────────────
Invoke-Step "세포 낱장 추출"      @("tools\extract_cells.py", "--run", $best,
                                    "--consensus", $consensusJson, "--out", $figuresDir)
Invoke-Step "슬라이드 빌드"       @("tools\build_slides.py", "--run", $best,
                                    "--figures", $figuresDir, "--out", $slidesOut)

# ── 끝 ──────────────────────────────────────────────────────────────
$total = (Get-Date) - $started
Write-Host ""
Write-Host ("전체 {0:hh\:mm\:ss} · {1}단계 모두 통과" -f $total, $steps) -ForegroundColor Green
Write-Host ""
Write-Host "결과"
Write-Host "  runs\$best\                지표 · 혼동행렬 · 히트맵 · 오류 격자"
Write-Host "  $consensusJson   모델 간 공통 오분류"
if ($Hires) {
    Write-Host "  $survivorsJson  해상도를 올려도 남은 장"
}
Write-Host "  $slidesOut          발표 자료"
Write-Host "  docs\results.md            결과 서술 (직접 갱신해야 함)"
if ($Quick) {
    Write-Host ""
    Write-Host "빠른 점검 모드였습니다. 숫자를 인용하지 마세요." -ForegroundColor Yellow
    Write-Host "완성본은 건드리지 않았습니다 — $slidesOut 에 따로 나왔습니다." -ForegroundColor Yellow
}

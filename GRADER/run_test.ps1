# run_test.ps1 - pure ASCII
$ErrorActionPreference = "Stop"

Write-Host "=== 1. Check input files ===" -ForegroundColor Cyan
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "data\questions.csv")) {
    Write-Host "ERROR: data\questions.csv not found" -ForegroundColor Red
    exit 1
}
Write-Host "OK: input files present" -ForegroundColor Green

Write-Host ""
Write-Host "=== 2. Python interpreter ===" -ForegroundColor Cyan
$py = $null
if ($env:VIRTUAL_ENV) {
    $py = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    Write-Host "Using active venv: $env:VIRTUAL_ENV"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $py = ".\.venv\Scripts\python.exe"
} else {
    python -m venv .venv
    $py = ".\.venv\Scripts\python.exe"
}
Write-Host "Python: $py"

Write-Host ""
Write-Host "=== 3. Install dependencies ===" -ForegroundColor Cyan
$t0 = Get-Date
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install -r requirements.txt
$installTime = [int]((Get-Date) - $t0).TotalSeconds
Write-Host "OK: install took $installTime sec" -ForegroundColor Green

Write-Host ""
Write-Host "=== 4. Get fresh GigaChat token ===" -ForegroundColor Cyan
if (-not $env:GIGACHAT_AUTH_KEY) {
    $authFile = ".gigachat_auth"
    if (Test-Path $authFile) {
        $env:GIGACHAT_AUTH_KEY = (Get-Content $authFile -Raw).Trim()
        Write-Host "Loaded AUTH_KEY from .gigachat_auth"
    } else {
        Write-Host "ERROR: GIGACHAT_AUTH_KEY not set and .gigachat_auth not found." -ForegroundColor Red
        Write-Host "Create .gigachat_auth with your base64 auth key (no 'Basic ' prefix)." -ForegroundColor Yellow
        exit 1
    }
}
$token = & $py get_token.py
if ($LASTEXITCODE -ne 0 -or -not $token) {
    Write-Host "ERROR: failed to get GigaChat token" -ForegroundColor Red
    exit 1
}
$token = $token.Trim()
$env:LLM_API_KEY = $token
Write-Host "OK: token received (len=$($token.Length), prefix=$($token.Substring(0,8))...)"

if (-not $env:LLM_BASE_URL) { $env:LLM_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1" }
if (-not $env:LLM_MODEL)    { $env:LLM_MODEL    = "GigaChat-2" }
if (-not $env:EMB_MODEL)    { $env:EMB_MODEL    = "BAAI/bge-m3" }
if (-not $env:DATA_DIR)     { $env:DATA_DIR     = "data" }
if (-not $env:MAX_WORKERS)  { $env:MAX_WORKERS  = "2" }

Write-Host ""
Write-Host "=== 5. Run solution.py ===" -ForegroundColor Cyan
$t0 = Get-Date
& $py solution.py
$runTime = [int]((Get-Date) - $t0).TotalSeconds
Write-Host ""
Write-Host "OK: solution.py finished in $runTime sec" -ForegroundColor Green

Write-Host ""
Write-Host "=== 6. Check submission.csv ===" -ForegroundColor Cyan
if (Test-Path "submission.csv") {
    $lines = (Get-Content "submission.csv").Count
    Write-Host "OK: submission.csv created, lines: $lines" -ForegroundColor Green
    Write-Host "First 5 lines:" -ForegroundColor Cyan
    Get-Content "submission.csv" -Head 5 | Write-Host
} else {
    Write-Host "ERROR: submission.csv NOT created!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "DONE." -ForegroundColor Green
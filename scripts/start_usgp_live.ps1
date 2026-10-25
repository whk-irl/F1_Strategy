param(
    [int]$Port = 8501,
    [switch]$SkipRefresh
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Install dependencies first: uv sync --extra dev'
}
if ($Port -lt 1024 -or $Port -gt 65535) {
    throw 'Choose a port between 1024 and 65535.'
}

if (-not $SkipRefresh) {
    & $pythonExe -m services.live.weekend --config configs/weekends/us-2026.json
    if ($LASTEXITCODE -ne 0) {
        throw 'Weekend refresh failed. Review the report before using -SkipRefresh with saved data.'
    }
}

$runDirectory = Join-Path $repoRoot ('runs\live\usgp-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$env:PITWALL_LIVE_PROVIDER = 'free'
$env:PITWALL_START_LIVE = '1'
$rawPath = Join-Path $runDirectory 'timing.txt'
$snapshotPath = Join-Path $repoRoot 'data\live\free-timing.json'
$arguments = @('-m', 'services.live.free_timing', '--raw', ('"' + $rawPath + '"'),
    '--snapshot', ('"' + $snapshotPath + '"'))
$receiver = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runDirectory 'receiver.out.log') `
    -RedirectStandardError (Join-Path $runDirectory 'receiver.err.log')

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
$app = $null
if (-not $listener) {
    $app = Start-Process -FilePath $pythonExe -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -ArgumentList @('-m', 'streamlit', 'run', 'services/frontend/app.py',
            '--server.headless', 'true', '--server.address', '127.0.0.1', '--server.port', $Port) `
        -PassThru -RedirectStandardOutput (Join-Path $runDirectory 'app.out.log') `
        -RedirectStandardError (Join-Path $runDirectory 'app.err.log')
} else {
    Write-Host "Port $Port already has a listener. Use its Live tab, or select another -Port."
}

$details = @{
    started_at = (Get-Date).ToUniversalTime().ToString('o')
    receiver_pid = $receiver.Id
    app_pid = if ($app) { $app.Id } else { $null }
    url = "http://localhost:$Port/"
    raw_path = $rawPath
    snapshot_path = $snapshotPath
}
$details | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDirectory 'processes.json')
Write-Host "Local race view: http://localhost:$Port/"
Write-Host "Logs and process IDs: $runDirectory"
Write-Host 'Start 20-30 minutes before the race. Free timing access must be checked while active.'
Write-Host 'Open the Live tab. No GPS or full telemetry is expected from the free source.'

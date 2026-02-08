# ============================================================
# GitHub Actions Self-Hosted Runner Setup for Windows
# ============================================================
# This script downloads, configures, and installs the GitHub
# Actions runner as a Windows service on your local machine.
#
# Usage (run as Administrator):
#   .\scripts\setup-runner.ps1
#
# Prerequisites:
#   - PowerShell 5.1+
#   - Git installed
#   - gh CLI authenticated (gh auth login)
#   - Run as Administrator (for service install)
# ============================================================

$ErrorActionPreference = "Stop"

$REPO = "vikasraoot-dot/RubberBandBot"
$RUNNER_DIR = "C:\actions-runner"
$RUNNER_NAME = "rubberband-local"
$RUNNER_LABELS = "self-hosted,Windows,X64,trading-bot"

Write-Host ""
Write-Host "  GitHub Actions Self-Hosted Runner Setup" -ForegroundColor Cyan
Write-Host "  Repo: $REPO"
Write-Host "  Runner Dir: $RUNNER_DIR"
Write-Host "  Runner Name: $RUNNER_NAME"
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "  WARNING: Not running as Administrator." -ForegroundColor Yellow
    Write-Host "  Service install will require admin. Continue anyway? (y/n)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -ne "y") { exit 1 }
}

# Step 1: Get registration token from GitHub
Write-Host "[1/6] Getting registration token from GitHub..." -ForegroundColor Green
$tokenJson = gh api "repos/$REPO/actions/runners/registration-token" --method POST 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to get registration token. Is gh CLI authenticated?" -ForegroundColor Red
    Write-Host "  Run: gh auth login" -ForegroundColor Yellow
    exit 1
}
$token = ($tokenJson | ConvertFrom-Json).token
Write-Host "  Token acquired." -ForegroundColor Green

# Step 2: Create runner directory
Write-Host "[2/6] Setting up runner directory..." -ForegroundColor Green
if (Test-Path $RUNNER_DIR) {
    Write-Host "  Runner directory already exists at $RUNNER_DIR"
    Write-Host "  If reconfiguring, delete it first: Remove-Item -Recurse $RUNNER_DIR"

    # Check if runner is already configured
    if (Test-Path "$RUNNER_DIR\.runner") {
        Write-Host "  Runner already configured. Skipping to service setup." -ForegroundColor Yellow
        Set-Location $RUNNER_DIR
        goto service_setup
    }
} else {
    New-Item -ItemType Directory -Path $RUNNER_DIR -Force | Out-Null
}
Set-Location $RUNNER_DIR

# Step 3: Download the runner
Write-Host "[3/6] Downloading GitHub Actions runner..." -ForegroundColor Green
$runnerVersion = "2.321.0"
$runnerUrl = "https://github.com/actions/runner/releases/download/v$runnerVersion/actions-runner-win-x64-$runnerVersion.zip"
$zipPath = "$RUNNER_DIR\actions-runner.zip"

if (-not (Test-Path "$RUNNER_DIR\config.cmd")) {
    Write-Host "  Downloading v$runnerVersion..."
    Invoke-WebRequest -Uri $runnerUrl -OutFile $zipPath -UseBasicParsing
    Write-Host "  Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $RUNNER_DIR -Force
    Remove-Item $zipPath -Force
    Write-Host "  Done." -ForegroundColor Green
} else {
    Write-Host "  Runner binary already present, skipping download." -ForegroundColor Yellow
}

# Step 4: Configure the runner
Write-Host "[4/6] Configuring runner..." -ForegroundColor Green
& "$RUNNER_DIR\config.cmd" `
    --url "https://github.com/$REPO" `
    --token $token `
    --name $RUNNER_NAME `
    --labels $RUNNER_LABELS `
    --work "_work" `
    --runasservice `
    --replace

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Runner configuration failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Runner configured." -ForegroundColor Green

# Step 5: Install as Windows service
:service_setup
Write-Host "[5/6] Installing runner as Windows service..." -ForegroundColor Green

# The --runasservice flag in config.cmd should have handled this,
# but let's verify and set up if needed
$serviceName = "actions.runner.$($REPO.Replace('/', '.')).$RUNNER_NAME"
$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue

if ($service) {
    Write-Host "  Service '$serviceName' already exists (Status: $($service.Status))"
    if ($service.Status -ne "Running") {
        Write-Host "  Starting service..."
        Start-Service -Name $serviceName
    }
} else {
    Write-Host "  Installing service..."
    & "$RUNNER_DIR\svc.cmd" install
    & "$RUNNER_DIR\svc.cmd" start
}

Write-Host "  Service installed and running." -ForegroundColor Green

# Step 6: Verify
Write-Host "[6/6] Verifying runner..." -ForegroundColor Green
$service = Get-Service -Name "actions.runner.*" -ErrorAction SilentlyContinue
if ($service -and $service.Status -eq "Running") {
    Write-Host ""
    Write-Host "  SUCCESS! Runner is installed and running." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Runner Name:    $RUNNER_NAME" -ForegroundColor Cyan
    Write-Host "  Service Name:   $($service.Name)" -ForegroundColor Cyan
    Write-Host "  Service Status: $($service.Status)" -ForegroundColor Cyan
    Write-Host "  Labels:         $RUNNER_LABELS" -ForegroundColor Cyan
    Write-Host "  Work Dir:       $RUNNER_DIR\_work" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Add secrets to GitHub repo:" -ForegroundColor Yellow
    Write-Host "     gh secret set APCA_API_KEY_ID" -ForegroundColor White
    Write-Host "     gh secret set APCA_API_SECRET_KEY" -ForegroundColor White
    Write-Host "  2. The bot will auto-run at 9:25 AM ET on weekdays" -ForegroundColor Yellow
    Write-Host "  3. Or trigger manually: gh workflow run ema-scalp.yml" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  WARNING: Service may not be running. Check manually:" -ForegroundColor Yellow
    Write-Host "  Get-Service 'actions.runner.*'" -ForegroundColor White
}

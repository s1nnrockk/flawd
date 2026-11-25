[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TrainTokens,
    [string]$TrainerPath = ".\build\Release\flawd_train.exe",
    [string]$CheckpointPath = ".\model.flawd",
    [string]$LogPath = ".\training-resumable.log",
    [ValidateRange(0.01, 8760)]
    [double]$Hours = 720,
    [ValidateRange(1, 8192)]
    [int]$Context = 64,
    [ValidateRange(1, 4096)]
    [int]$Embedding = 32,
    [ValidateRange(1, 16384)]
    [int]$FeedForward = 64,
    [ValidateRange(1, 64)]
    [int]$Layers = 2,
    [ValidateRange(1, 64)]
    [int]$Heads = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TrainTokens = (Resolve-Path -LiteralPath $TrainTokens).Path
$TrainerPath = (Resolve-Path -LiteralPath $TrainerPath).Path
$CheckpointPath = [System.IO.Path]::GetFullPath($CheckpointPath)
$LogPath = [System.IO.Path]::GetFullPath($LogPath)

if (-not (Test-Path -LiteralPath $TrainTokens -PathType Leaf)) {
    throw "Training token file does not exist: $TrainTokens"
}
if (-not (Test-Path -LiteralPath $TrainerPath -PathType Leaf)) {
    throw "Trainer executable does not exist: $TrainerPath"
}
if ([System.IO.Path]::GetFullPath($TrainTokens) -eq $CheckpointPath) {
    throw "Checkpoint path must be different from the training token file."
}

$checkpointDirectory = Split-Path -Parent $CheckpointPath
if ($checkpointDirectory -and -not (Test-Path -LiteralPath $checkpointDirectory)) {
    New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null
}
$logDirectory = Split-Path -Parent $LogPath
if ($logDirectory -and -not (Test-Path -LiteralPath $logDirectory)) {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
}

$transcriptStarted = $false
try {
    Start-Transcript -Path $LogPath -Append | Out-Null
    $transcriptStarted = $true
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $epochNumber = 0

    Write-Host "FLAWD resumable training"
    Write-Host "Training data: $TrainTokens"
    Write-Host "Checkpoint: $CheckpointPath"
    Write-Host "Time budget: $Hours hours"
    Write-Host "Checkpoint is saved after every completed epoch."
    Write-Host "Press Ctrl+C to stop; the last completed epoch remains saved."

    while ($timer.Elapsed.TotalHours -lt $Hours) {
        $epochNumber++
        $resumeArgument = if (Test-Path -LiteralPath $CheckpointPath -PathType Leaf) {
            $CheckpointPath
        } else {
            "-"
        }

        $arguments = @(
            $TrainTokens,
            $CheckpointPath,
            "1",
            "-",
            $resumeArgument,
            "$Context",
            "$Embedding",
            "$FeedForward",
            "$Layers",
            "$Heads"
        )

        Write-Host ""
        Write-Host ("Starting training chunk {0}; elapsed {1:N2} h" -f $epochNumber, $timer.Elapsed.TotalHours)
        & $TrainerPath @arguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "flawd_train exited with code $exitCode. The previous completed checkpoint was left untouched."
        }
        if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
            throw "Trainer finished without creating the expected checkpoint: $CheckpointPath"
        }
    }

    Write-Host ("Time budget reached after {0:N2} hours. Checkpoint: {1}" -f $timer.Elapsed.TotalHours, $CheckpointPath)
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}

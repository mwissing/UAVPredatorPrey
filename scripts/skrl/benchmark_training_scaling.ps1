param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$IsaacLabBat = "C:\RL\IsaacLab\isaaclab.bat",
    [string]$Task = "3v1-survival-soft-oob-teammate-vel-random-spawn-v0",
    [string]$Agent = "skrl_mappo_attention_critic_prey_attention_large_gru_cfg_entry_point",
    [int[]]$EnvCounts = @(4096, 6144, 8192),
    [int]$ReferenceEnvs = 4096,
    [int]$ReferenceIterations = 300,
    [int]$Rollouts = 24,
    [int]$Seed = 42,
    [double]$LearningRate = 0.00001,
    [double]$KlThreshold = 0.05,

    [ValidateSet("none", "predator", "prey")]
    [string]$FreezeAgent = "none",
    [string]$OpponentPool = "",
    [double]$PerEnvPoolProb = 0.5,
    [int]$PerEnvPoolMaxPolicies = 4,
    [string[]]$ExtraTrainArgs = @(),
    [string]$OutputDirectory = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$trainScript = Join-Path $PSScriptRoot "train.py"
if (-not (Test-Path -LiteralPath $IsaacLabBat)) {
    throw "Isaac Lab launcher not found: $IsaacLabBat"
}
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}
if ($OpponentPool -and -not (Test-Path -LiteralPath $OpponentPool)) {
    throw "Opponent pool not found: $OpponentPool"
}
if ($OpponentPool -and $FreezeAgent -eq "none") {
    throw "-OpponentPool requires -FreezeAgent predator or prey."
}
if ($ReferenceEnvs -le 0 -or $ReferenceIterations -le 0 -or $Rollouts -le 0) {
    throw "ReferenceEnvs, ReferenceIterations, and Rollouts must be positive."
}
if ($EnvCounts.Count -eq 0 -or ($EnvCounts | Where-Object { $_ -le 0 })) {
    throw "EnvCounts must contain only positive values."
}

if (-not $OutputDirectory) {
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $OutputDirectory = Join-Path $repoRoot "logs\benchmarks\training_scaling_$timestamp"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$referenceTransitions = [int64]$ReferenceEnvs * $ReferenceIterations * $Rollouts
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$learningRateArg = $LearningRate.ToString("R", $invariantCulture)
$klThresholdArg = $KlThreshold.ToString("R", $invariantCulture)
$poolProbabilityArg = $PerEnvPoolProb.ToString("R", $invariantCulture)
$results = [System.Collections.Generic.List[object]]::new()
$originalLocation = Get-Location

try {
    Set-Location $repoRoot
    foreach ($numEnvs in $EnvCounts) {
        $iterations = [math]::Max(1, [math]::Ceiling($referenceTransitions / ([double]$numEnvs * $Rollouts)))
        $actualTransitions = [int64]$numEnvs * $iterations * $Rollouts
        $logPath = Join-Path $OutputDirectory ("envs_{0}_iterations_{1}.log" -f $numEnvs, $iterations)

        $trainArgs = @(
            "-p", $trainScript,
            "--headless",
            "--task", $Task,
            "--agent", $Agent,
            "--algorithm", "MAPPO",
            "--num_envs", "$numEnvs",
            "--seed", "$Seed",
            "--checkpoint", $Checkpoint,
            "--max_iterations", "$iterations",
            "--learning-rate", $learningRateArg,
            "--kl-threshold", $klThresholdArg
        )
        if ($FreezeAgent -ne "none") {
            $trainArgs += @("--freeze-agents", $FreezeAgent)
        }
        if ($OpponentPool) {
            $trainArgs += @(
                "--per-env-opponent-pool", $OpponentPool,
                "--per-env-pool-prob", $poolProbabilityArg,
                "--per-env-pool-max-policies", "$PerEnvPoolMaxPolicies",
                "--per-env-pool-seed", "$Seed"
            )
        }
        $trainArgs += $ExtraTrainArgs

        Write-Host ""
        Write-Host ("[BENCHMARK] envs={0}, iterations={1}, transitions={2}" -f $numEnvs, $iterations, $actualTransitions)
        Write-Host ("& `"{0}`" {1}" -f $IsaacLabBat, ($trainArgs -join " "))
        if ($DryRun) {
            continue
        }

        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            # Windows PowerShell converts native stderr redirected through 2>&1
            # into error records. Isaac/skrl use stderr for normal INFO output.
            $ErrorActionPreference = "Continue"
            & $IsaacLabBat @trainArgs 2>&1 | Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
            $stopwatch.Stop()
        }

        $trainingSeconds = $null
        $logText = Get-Content -Raw -Path $logPath
        $trainingMatch = [regex]::Match($logText, "Training time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds")
        if ($trainingMatch.Success) {
            $trainingSeconds = [double]::Parse(
                $trainingMatch.Groups[1].Value,
                $invariantCulture
            )
        }

        $results.Add([pscustomobject]@{
            num_envs = $numEnvs
            iterations = $iterations
            rollouts = $Rollouts
            environment_transitions = $actualTransitions
            training_seconds = $trainingSeconds
            wall_seconds = $stopwatch.Elapsed.TotalSeconds
            transitions_per_training_second = if ($trainingSeconds) { $actualTransitions / $trainingSeconds } else { $null }
            transitions_per_wall_second = $actualTransitions / $stopwatch.Elapsed.TotalSeconds
            exit_code = $exitCode
            log_file = $logPath
        })

        $results | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OutputDirectory "results.json")
        if ($exitCode -ne 0) {
            throw "Training benchmark failed for $numEnvs envs with exit code $exitCode. See $logPath"
        }
    }
}
finally {
    Set-Location $originalLocation
}

if (-not $DryRun) {
    $results | Format-Table num_envs, iterations, training_seconds, wall_seconds, transitions_per_training_second -AutoSize
    Write-Host "[BENCHMARK] Results: $(Join-Path $OutputDirectory 'results.json')"
}

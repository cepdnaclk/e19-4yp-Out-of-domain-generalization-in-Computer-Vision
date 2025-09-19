# Crowd Pruning and Knee Point Analysis - All Few-Shot Configurations (PowerShell)
# This script runs the evaluation for 1-shot, 2-shot, 4-shot, and 8-shot configurations in parallel

Write-Host "=== Starting Crowd Pruning and Knee Point Analysis for All Few-Shot Configurations ===" -ForegroundColor Green
Write-Host "Timestamp: $(Get-Date)"

# Define few-shot configurations to run
$FewShots = @(1, 2, 4, 8)

# Define other common parameters
$Provider = "gemini"
$MaxCapacity = 1000
$FilterThreshold = 0.6
$CrowdingIterations = 3
$GroupSize = 30

# Create a logs directory for output
$LogsDir = "logs\crowd_pruning_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

Write-Host "Logs will be saved to: $LogsDir"

# Function to run evaluation for a specific few-shot configuration
function Start-Evaluation {
    param(
        [int]$FewShot
    )
    
    $LogFile = "$LogsDir\${FewShot}shot_evaluation.log"
    
    "Starting ${FewShot}-shot evaluation at $(Get-Date)" | Out-File -FilePath $LogFile
    
    try {
        $process = Start-Process -FilePath "python" -ArgumentList @(
            "evaluate_by_crowding_and_knee.py",
            "--few_shot", $FewShot,
            "--provider", $Provider,
            "--max_capacity", $MaxCapacity,
            "--filter_threshold", $FilterThreshold,
            "--crowding_iterations", $CrowdingIterations,
            "--group_size", $GroupSize
        ) -NoNewWindow -PassThru -RedirectStandardOutput "$LogFile.out" -RedirectStandardError "$LogFile.err"
        
        return $process
    }
    catch {
        Write-Host "❌ Failed to start ${FewShot}-shot evaluation: $_" -ForegroundColor Red
        "❌ Failed to start ${FewShot}-shot evaluation: $_" | Out-File -FilePath $LogFile -Append
        return $null
    }
}

# Store process objects for monitoring
$Processes = @()
$ProcessConfigs = @()

Write-Host "Starting parallel evaluations..."

# Start all evaluations in parallel
foreach ($FewShot in $FewShots) {
    Write-Host "Starting ${FewShot}-shot evaluation..."
    $Process = Start-Evaluation -FewShot $FewShot
    
    if ($Process) {
        $Processes += $Process
        $ProcessConfigs += $FewShot
        Write-Host "  └─ Process started with PID: $($Process.Id)" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "All evaluations started. Waiting for completion..."
Write-Host "Monitor progress with individual log files in: $LogsDir"
Write-Host ""

# Wait for all processes to complete and collect results
$Results = @()
for ($i = 0; $i -lt $Processes.Count; $i++) {
    $Process = $Processes[$i]
    $Config = $ProcessConfigs[$i]
    
    Write-Host "Waiting for ${Config}-shot evaluation (PID: $($Process.Id))..."
    $Process.WaitForExit()
    
    $LogFile = "$LogsDir\${Config}shot_evaluation.log"
    
    if ($Process.ExitCode -eq 0) {
        $Results += "✅ ${Config}-shot: SUCCESS"
        "✅ ${Config}-shot evaluation completed successfully at $(Get-Date)" | Out-File -FilePath $LogFile -Append
        Write-Host "✅ ${Config}-shot evaluation completed successfully" -ForegroundColor Green
    } else {
        $Results += "❌ ${Config}-shot: FAILED (exit code: $($Process.ExitCode))"
        "❌ ${Config}-shot evaluation failed with exit code $($Process.ExitCode) at $(Get-Date)" | Out-File -FilePath $LogFile -Append
        Write-Host "❌ ${Config}-shot evaluation failed with exit code $($Process.ExitCode)" -ForegroundColor Red
    }
}

# Merge stdout and stderr into main log files
foreach ($FewShot in $FewShots) {
    $MainLog = "$LogsDir\${FewShot}shot_evaluation.log"
    $OutLog = "$LogsDir\${FewShot}shot_evaluation.log.out"
    $ErrLog = "$LogsDir\${FewShot}shot_evaluation.log.err"
    
    if (Test-Path $OutLog) {
        "`n=== STDOUT ===" | Out-File -FilePath $MainLog -Append
        Get-Content $OutLog | Out-File -FilePath $MainLog -Append
        Remove-Item $OutLog
    }
    
    if (Test-Path $ErrLog) {
        "`n=== STDERR ===" | Out-File -FilePath $MainLog -Append
        Get-Content $ErrLog | Out-File -FilePath $MainLog -Append
        Remove-Item $ErrLog
    }
}

# Create summary report
$SummaryFile = "$LogsDir\summary_report.txt"
$SummaryContent = @"
=== Crowd Pruning and Knee Point Analysis Summary ===
Execution Date: $(Get-Date)
Script Location: $(Get-Location)\evaluate_by_crowding_and_knee.py
Logs Directory: $LogsDir

Configuration Used:
  Provider: $Provider
  Max Capacity: $MaxCapacity
  Filter Threshold: $FilterThreshold
  Crowding Iterations: $CrowdingIterations
  Group Size: $GroupSize

Results:
"@

foreach ($Result in $Results) {
    $SummaryContent += "`n  $Result"
}

$SummaryContent += @"

Output Files Generated:
"@

foreach ($FewShot in $FewShots) {
    $SummaryContent += @"

  ${FewShot}-shot results:
    - final_results/crowded/${FewShot}-shot.txt
    - final_results/knee/${FewShot}-shot.txt
    - final_results/evaluation/${FewShot}-shot-results.txt
"@
}

$SummaryContent | Out-File -FilePath $SummaryFile

# Print final summary to console
Write-Host ""
Write-Host "=== EXECUTION SUMMARY ===" -ForegroundColor Yellow
Write-Host "Completion Time: $(Get-Date)"
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow

foreach ($Result in $Results) {
    if ($Result.Contains("SUCCESS")) {
        Write-Host "  $Result" -ForegroundColor Green
    } else {
        Write-Host "  $Result" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📊 Summary report saved to: $SummaryFile" -ForegroundColor Cyan
Write-Host "📁 All logs available in: $LogsDir" -ForegroundColor Cyan

# Check if all evaluations were successful
$SuccessCount = ($Results | Where-Object { $_ -like "*SUCCESS*" }).Count
$TotalCount = $Results.Count

if ($SuccessCount -eq $TotalCount) {
    Write-Host "🎉 All evaluations completed successfully!" -ForegroundColor Green
    exit 0
} else {
    $FailedCount = $TotalCount - $SuccessCount
    Write-Host "⚠️  $FailedCount out of $TotalCount evaluations failed." -ForegroundColor Yellow
    Write-Host "   Check individual log files for details." -ForegroundColor Yellow
    exit 1
}

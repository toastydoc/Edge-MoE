# Waits for answer generation (all three answer files complete), then judges blind with Gemini.
# Reads the API key from the user environment (setx GEMINI_API_KEY ...); it is never printed or written to disk.
$env:PYTHONIOENCODING = 'utf-8'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$k = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $k) { $k = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User") }
$env:GEMINI_API_KEY = $k
function Lines($f) { if (Test-Path $f) { @(Get-Content $f).Count } else { 0 } }
while ((Lines "results/judge/answers_full.jsonl") -lt 300) { Start-Sleep -Seconds 60 }
Write-Output "=== judging start $(Get-Date -Format HH:mm)"
& $py scripts/judge_answers.py --model gemini-3.1-pro-preview --pairs own:full shared:full
Write-Output "=== judging done $(Get-Date -Format HH:mm)"

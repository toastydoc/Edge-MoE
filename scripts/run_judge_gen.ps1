# Waits for any running project GPU job to finish, then generates the answers for the blind-judged test.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
Add-Type -Namespace Win -Name Power3 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power3]::SetThreadExecutionState([uint32]2147483649)
function Busy { @(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*edge-moe*" }).Count -gt 0 }
$idle = 0
while ($idle -lt 6) { if (Busy) { $idle = 0 } else { $idle++ }; Start-Sleep -Seconds 15 }
Write-Output "=== answers start $(Get-Date -Format HH:mm)"
& $py scripts/gen_judge_answers.py 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects|non_blocking'
Write-Output "=== answers done $(Get-Date -Format HH:mm)"
[void][Win.Power3]::SetThreadExecutionState([uint32]2147483648)

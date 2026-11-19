# Head-to-head against the best model that fits the same device: dense Gemma 4 12B (4-bit, ~6.7 GB of weights)
# answers the same 300 questions; Gemini judges it against the 51 and 64 per-layer packs and against the full 26B.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
Add-Type -Namespace Win -Name Power5 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power5]::SetThreadExecutionState([uint32]2147483649)
Write-Output "=== Gemma 4 12B answers $(Get-Date -Format HH:mm)"
& $py scripts/gen_judge_answers.py --configs gemma4_12b 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects|non_blocking|Loading weights|quantize\('
$k = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $k) { $k = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User") }
$env:GEMINI_API_KEY = $k
Write-Output "=== judging $(Get-Date -Format HH:mm)"
& $py scripts/judge_answers.py --model gemini-3.1-pro-preview --pairs own:gemma4_12b own_n64:gemma4_12b gemma4_12b:full
Write-Output "=== done $(Get-Date -Format HH:mm)"
[void][Win.Power5]::SetThreadExecutionState([uint32]2147483648)

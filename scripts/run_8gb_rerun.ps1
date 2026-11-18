# Rerun of the three store configurations whose pinned-RAM allocation failed in the first pass.
$env:PYTHONIOENCODING = 'utf-8'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
foreach ($sel in @('plan')) {
    Write-Output "=== $sel"
    & $py scripts/eval_policy.py --config 26b-store --select $sel --tasks tasks_v2.jsonl 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects'
    Start-Sleep -Seconds 20
}

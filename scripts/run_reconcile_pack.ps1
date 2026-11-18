# Reconcile-aware packs at 51 experts per layer on the 122 policy tasks.
$env:PYTHONIOENCODING = 'utf-8'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
foreach ($sel in @('mix:regulation+reconcile', 'pack:reconcile')) {
    Write-Output "=== $sel"
    & $py scripts/eval_policy.py --config 26b-store --select $sel --tasks tasks_v2.jsonl 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects'
    Start-Sleep -Seconds 20
}

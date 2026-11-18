# The 8GB comparison on the 122 policy tasks, one configuration at a time (one GPU job at a time).
# Results: results/policy_<config>_tasks_v2.json
$env:PYTHONIOENCODING = 'utf-8'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$runs = @(
    @('--config', 'gemma4-12b'),
    @('--config', '26b-store', '--select', 'pack:law'),
    @('--config', '26b-store', '--select', 'pack:regulation'),
    @('--config', '26b-store', '--select', 'mix:regulation+math'),
    @('--config', '26b-store', '--select', 'joint'),
    @('--config', '26b-store', '--select', 'plan'),
    @('--config', '26b-store', '--select', 'offload'),
    @('--config', '26b', '--install', 'base', 'medicine', 'law', 'math')
)
foreach ($r in $runs) {
    Write-Output "=== $($r -join ' ')"
    & $py scripts/eval_policy.py @r --tasks tasks_v2.jsonl 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects'
}

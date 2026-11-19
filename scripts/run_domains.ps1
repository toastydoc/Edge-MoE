# Ten Wikipedia domains: profile each, then score packs and k-domain mixes on held-out text.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$domains = 'art', 'history', 'biology', 'geology', 'politics', 'computer_science', 'economics', 'chemistry', 'physics', 'philosophy'
foreach ($d in $domains) {
    if (Test-Path "profiles/wiki_$d.npz") { continue }
    Write-Output "=== profile $d"
    & $py scripts/profile_domain.py --prompts "data/prompts/wiki_$d.jsonl" --out "profiles/wiki_$d.npz" --max-tokens 512 --batch-size 48 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects|^batch'
    & $py scripts/export_domain_maps.py 2>&1 | Select-String -NotMatch "Warning"
    Start-Sleep -Seconds 10
}
Write-Output "=== domain loss sweep"
& $py scripts/eval_domain_loss.py 2>&1 | Select-String -NotMatch 'Warning|warn\(|Fetching|RemoteException|Redirects'

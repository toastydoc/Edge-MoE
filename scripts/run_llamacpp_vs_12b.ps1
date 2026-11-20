# Pack vs Gemma 4 12B on a real runtime (stock llama.cpp b12549), at equal file size.
# Exports each Wikipedia domain's 64-per-layer pack as GGUF (Q3_K_M 7.37 GB-class, Q4_K_M 9.2 GB-class),
# answers the 300 judge questions with llama-server, then Gemini judges:
#   pack Q3_K_M (~7.4 GB) vs 12B Q4_K_M (7.38 GB)   and   pack Q4_K_M (~9.2 GB) vs 12B Q6_K (9.79 GB)
# 12B answers: scripts/gen_llamacpp_answers.py --name lcpp_12b_q4km / lcpp_12b_q6k (run first).
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$lc = "$env:USERPROFILE\tools\llama.cpp-b12549"
Add-Type -Namespace Win -Name Power6 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power6]::SetThreadExecutionState([uint32]2147483649)
$domains = 'art','history','biology','geology','politics','computer_science','economics','chemistry','physics','philosophy'
foreach ($d in $domains) {
    $base = "exports\gemma4-26b-$d-64e"
    if ((Test-Path "$base-Q3_K_M.gguf") -and (Test-Path "$base-Q4_K_M.gguf")) { continue }
    Write-Output "=== build $d $(Get-Date -Format HH:mm)"
    & $py scripts/export_pack_checkpoint.py --domain $d --budget 64 --check 2>&1 | Select-String 'router check|done:|Error|error'
    if (-not (Test-Path $base)) { Write-Output "export failed for $d"; exit 1 }
    & $py "$lc\src\convert_hf_to_gguf.py" $base --outtype bf16 --outfile "$base-bf16.gguf" 2>&1 | Select-String 'successfully|Error'
    foreach ($t in 'Q3_K_M', 'Q4_K_M') { & "$lc\cpu\llama-quantize.exe" "$base-bf16.gguf" "$base-$t.gguf" $t 16 2>&1 | Select-String 'quant size' }
    if ((Test-Path "$base-Q3_K_M.gguf") -and (Test-Path "$base-Q4_K_M.gguf")) {
        Copy-Item "$base\pack_manifest.json" "results\pack_manifest_${d}_64e.json"
        Remove-Item -Recurse -Force $base, "$base-bf16.gguf"
    }
}
Write-Output "=== answers $(Get-Date -Format HH:mm)"
& $py scripts/gen_llamacpp_answers.py --name lcpp_pack64_q3km --gguf "exports/gemma4-26b-{domain}-64e-Q3_K_M.gguf"
& $py scripts/gen_llamacpp_answers.py --name lcpp_pack64_q4km --gguf "exports/gemma4-26b-{domain}-64e-Q4_K_M.gguf"
$k = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $k) { $k = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User") }
$env:GEMINI_API_KEY = $k
Write-Output "=== judging $(Get-Date -Format HH:mm)"
& $py scripts/judge_answers.py --model gemini-3.1-pro-preview --pairs lcpp_pack64_q3km:lcpp_12b_q4km lcpp_pack64_q4km:lcpp_12b_q6k
Write-Output "=== done $(Get-Date -Format HH:mm)"
[void][Win.Power6]::SetThreadExecutionState([uint32]2147483648)

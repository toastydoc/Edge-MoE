# Where does a pack overtake Gemma 4 12B on llama.cpp? Larger packs (80 and 96 experts per layer, Q4_K_M)
# against the 12B at Q6_K and at Q8_0 (near lossless), same 300 questions, same blind judge.
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$lc = "$env:USERPROFILE\tools\llama.cpp-b12549"
Add-Type -Namespace Win -Name Power7 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power7]::SetThreadExecutionState([uint32]2147483649)
if (-not (Test-Path exports\gemma4-12b-Q8_0.gguf)) { & "$lc\cpu\llama-quantize.exe" exports\gemma4-12b-bf16.gguf exports\gemma4-12b-Q8_0.gguf Q8_0 16 2>&1 | Select-String 'quant size' }
$domains = 'art','history','biology','geology','politics','computer_science','economics','chemistry','physics','philosophy'
foreach ($n in 96, 80) {
    foreach ($d in $domains) {
        $base = "exports\gemma4-26b-$d-${n}e"
        if (Test-Path "$base-Q4_K_M.gguf") { continue }
        Write-Output "=== build $d $n $(Get-Date -Format HH:mm)"
        & $py scripts/export_pack_checkpoint.py --domain $d --budget $n 2>&1 | Select-String 'done:|Error|error'
        if (-not (Test-Path $base)) { Write-Output "export failed for $d $n"; exit 1 }
        & $py "$lc\src\convert_hf_to_gguf.py" $base --outtype bf16 --outfile "$base-bf16.gguf" 2>&1 | Select-String 'successfully|Error'
        & "$lc\cpu\llama-quantize.exe" "$base-bf16.gguf" "$base-Q4_K_M.gguf" Q4_K_M 16 2>&1 | Select-String 'quant size'
        if (Test-Path "$base-Q4_K_M.gguf") { Remove-Item -Recurse -Force $base, "$base-bf16.gguf" }
    }
}
Write-Output "=== answers $(Get-Date -Format HH:mm)"
& $py scripts/gen_llamacpp_answers.py --name lcpp_12b_q8 --gguf exports/gemma4-12b-Q8_0.gguf
& $py scripts/gen_llamacpp_answers.py --name lcpp_pack96_q4km --gguf "exports/gemma4-26b-{domain}-96e-Q4_K_M.gguf"
& $py scripts/gen_llamacpp_answers.py --name lcpp_pack80_q4km --gguf "exports/gemma4-26b-{domain}-80e-Q4_K_M.gguf"
$k = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $k) { $k = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User") }
$env:GEMINI_API_KEY = $k
Write-Output "=== judging $(Get-Date -Format HH:mm)"
& $py scripts/judge_answers.py --model gemini-3.1-pro-preview --pairs lcpp_pack96_q4km:lcpp_12b_q8 lcpp_pack80_q4km:lcpp_12b_q8 lcpp_pack80_q4km:lcpp_12b_q6k
Write-Output "=== done $(Get-Date -Format HH:mm)"
[void][Win.Power7]::SetThreadExecutionState([uint32]2147483648)

# Judgements still owed after the Gemini spend cap stopped the overnight run (2026-11-20).
# Every pair resumes from its verdict file, so already-judged questions are not sent again.
#   pack96_q4km vs 12b_q8      13.0 vs 12.67 GB   (7 of 10 subjects done)
#   pack96_q4km vs full_q3km   13.0 vs 13.29 GB   drop experts or drop bits, at 13 GB
#   pack64_q4km vs full_q3km   9.21 vs 13.29 GB   (partly done)
#   full_q3ks   vs 12b_q8      12.22 vs 12.67 GB
# Not judged: the full 26B at Q2_K (10.58 GB) collapses into repeated fragments on all 300 answers,
# so there is no working full model at ~10 GB with standard quantisation (answers_lcpp_full_q2k.jsonl).
$env:PYTHONIOENCODING = 'utf-8'
$py = "$env:USERPROFILE\.venvs\edge-moe\Scripts\python.exe"
$k = [Environment]::GetEnvironmentVariable("GEMINI_API_KEY", "User")
if (-not $k) { $k = [Environment]::GetEnvironmentVariable("GOOGLE_API_KEY", "User") }
$env:GEMINI_API_KEY = $k
& $py scripts/judge_answers.py --model gemini-3.1-pro-preview --pairs `
    lcpp_pack96_q4km:lcpp_12b_q8 lcpp_pack96_q4km:lcpp_full_q3km lcpp_pack64_q4km:lcpp_full_q3km `
    lcpp_full_q3ks:lcpp_12b_q8

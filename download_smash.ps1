# Download all smash datasets except black_smash_07
$ErrorActionPreference = 'Continue'
$hf = 'C:\Users\jerry\miniconda3\envs\vlm\Scripts\hf.exe'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_XET_HIGH_PERFORMANCE = '1'   # these repos use Xet storage; hf_transfer is ignored for Xet
# Token is read from the environment; do NOT hardcode it (this file may be committed).
# Set it before running, e.g.:  $env:HF_TOKEN = '<your hf token>'
if (-not $env:HF_TOKEN) { Write-Error 'HF_TOKEN env var is not set. Set it before running.'; exit 1 }

# Latest two per color (black_smash_07 already present locally)
$datasets = @(
  'black_smash_05','black_smash_06',
  'yellow_smash_06','yellow_smash_07',
  'white_smash_06','white_smash_07'
)

$log = 'C:\Intern\download_smash.log'
"START $(Get-Date -Format s)  total=$($datasets.Count)" | Out-File $log -Encoding utf8

$i = 0
foreach ($d in $datasets) {
  $i++
  $repo = "EricChen06/$d"
  $dir  = "C:\Intern\$d"
  "[$i/$($datasets.Count)] $(Get-Date -Format HH:mm:ss) downloading $repo -> $dir" | Tee-Object -FilePath $log -Append
  & $hf download $repo --repo-type dataset --local-dir $dir 2>&1 | Out-File $log -Append -Encoding utf8
  if ($LASTEXITCODE -eq 0) {
    "[$i/$($datasets.Count)] $(Get-Date -Format HH:mm:ss) DONE  $d" | Tee-Object -FilePath $log -Append
  } else {
    "[$i/$($datasets.Count)] $(Get-Date -Format HH:mm:ss) FAILED($LASTEXITCODE) $d" | Tee-Object -FilePath $log -Append
  }
}
"ALL DONE $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append

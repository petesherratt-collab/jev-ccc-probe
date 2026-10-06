<#
.SYNOPSIS
    Run the Jev CCC probe end to end from PowerShell (Windows or Linux).

.DESCRIPTION
    The Python is already cross-platform; this wrapper handles the things that
    actually differ on Windows:

      * pins UTF-8 so prompts containing non-ASCII do not raise
        UnicodeEncodeError on a cp1252 console
      * creates and reuses a virtual environment, so numpy/scipy do not have to
        be installed machine-wide
      * checks for the API key in $env: rather than assuming a shell export
      * refuses to start a real run if the repo path is wrong, instead of
        failing 400 calls in
      * runs determinism BEFORE the main run, and stops if it says the model is
        stochastic while -Repeats is still 1

    Observation files are written as LF-only UTF-8 on every platform, so a run
    on Windows and a run on Linux produce byte-identical raw rows.

.PARAMETER RepoPath
    Path to the harness 'experiments' directory.

.PARAMETER Simulate
    Run offline against the built-in simulator. No API key, no network, no cost.

.PARAMETER Obs
    Observation file to append to. Default _obs_ccc.jsonl

.PARAMETER Repeats
    Calls per cell. Leave at 1 unless the determinism check says otherwise.

.PARAMETER SkipDeterminism
    Skip the determinism check. Only sensible if you have already run it against
    this model version.

.EXAMPLE
    .\Run-JevCcc.ps1 -RepoPath C:\src\harness\experiments -Simulate

.EXAMPLE
    $env:OPENROUTER_API_KEY = 'sk-or-...'
    .\Run-JevCcc.ps1 -RepoPath C:\src\harness\experiments
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $RepoPath,

    [switch] $Simulate,
    [string] $Obs = '_obs_ccc.jsonl',
    [int]    $Repeats = 1,
    [int]    $Workers = 4,
    [int]    $Seed = 305774821,
    [string] $Model = 'typesafe/jev-1.13',
    [string[]] $Domain,
    [string[]] $Encoding,
    [switch] $SkipDeterminism,
    [string] $VenvPath = '.venv',
    [switch] $NoVenv,

    # Simulator only: per-call noise. Use it to prove the stochastic gate below
    # actually fires before you trust it on a real run.
    [double] $SimJitter = 0,

    # An observation file from the OTHER machine. When given, the run finishes
    # with a cross-platform equivalence check and fails if the requests differed.
    [string] $Against,

    # A KEY=VALUE file holding OPENROUTER_API_KEY (.env or .env.txt). Read
    # directly rather than shelled out, so nothing in the file can execute.
    [string] $EnvFile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# UTF-8 everywhere. Windows consoles default to the system code page, which
# turns a non-ASCII character in a prompt into a crash rather than a character.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here

function Write-Step([string] $Message) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor DarkCyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor DarkCyan
}

function Write-Warn([string] $Message) {
    Write-Host "  ! $Message" -ForegroundColor Yellow
}

try {
    # ---------------------------------------------------------------- checks --
    Write-Step 'Preflight'

    $py = $null
    foreach ($candidate in @('python3', 'python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $py = $cmd.Source; break }
    }
    if (-not $py) {
        throw 'No Python found on PATH. Install Python 3.10+ and retry.'
    }
    $pyVersion = & $py -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'
    Write-Host "  python        : $py ($pyVersion)"

    $resolvedRepo = $null
    try { $resolvedRepo = (Resolve-Path -LiteralPath $RepoPath).Path } catch { }
    if (-not $resolvedRepo) {
        throw "RepoPath does not exist: $RepoPath"
    }
    # Fail now, not 400 calls in.
    $required = @('ccc_sql_items.py', 'ccc_code_items.py', 'run_ccc_sql.py')
    $missing = $required | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $resolvedRepo $_))
    }
    if ($missing) {
        throw ("RepoPath does not look like the harness 'experiments' directory " +
               "(missing: $($missing -join ', ')). Got: $resolvedRepo")
    }
    Write-Host "  repo          : $resolvedRepo"

    if (-not $Simulate) {
        if (-not $env:OPENROUTER_API_KEY -and $EnvFile) {
            if (-not (Test-Path -LiteralPath $EnvFile)) {
                throw "EnvFile not found: $EnvFile"
            }
            foreach ($line in Get-Content -LiteralPath $EnvFile) {
                $t = $line.Trim()
                if (-not $t -or $t.StartsWith('#') -or ($t -notmatch '=')) { continue }
                $name, $val = $t -split '=', 2
                $name = ($name -replace '^\s*export\s+', '').Trim()
                if ($name -eq 'OPENROUTER_API_KEY') {
                    $env:OPENROUTER_API_KEY = $val.Trim().Trim('"').Trim("'")
                    Write-Host "  key source    : $EnvFile"
                    break
                }
            }
            if (-not $env:OPENROUTER_API_KEY) {
                throw "$EnvFile does not define OPENROUTER_API_KEY"
            }
        }
        if (-not $env:OPENROUTER_API_KEY) {
            throw ("OPENROUTER_API_KEY is not set. In PowerShell:`n" +
                   "    `$env:OPENROUTER_API_KEY = 'sk-or-...'`n" +
                   "Or pass -EnvFile <path to .env / .env.txt>,`n" +
                   "or -Simulate to run offline against the simulator.")
        }
        Write-Host "  api key       : present ($($env:OPENROUTER_API_KEY.Length) chars)"
    }
    else {
        Write-Host "  mode          : SIMULATED (no key, no network, no cost)" -ForegroundColor Yellow
    }

    # ------------------------------------------------------------------ venv --
    Write-Step 'Environment'

    $isWindows_ = $IsWindows -or ($PSVersionTable.PSEdition -eq 'Desktop')
    $venvPy = if ($isWindows_) {
        Join-Path $VenvPath 'Scripts\python.exe'
    } else {
        Join-Path $VenvPath 'bin/python'
    }

    if ($NoVenv) {
        Write-Host '  using the system Python (-NoVenv)'
        $venvPy = $py
    }
    elseif (Test-Path -LiteralPath $venvPy) {
        Write-Host "  reusing $VenvPath"
    }
    else {
        Write-Host "  creating virtual environment in $VenvPath ..."
        & $py -m venv $VenvPath 2>&1 | Out-Null

        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPy)) {
            # venv defaults to symlinks. A OneDrive-synced folder, a mapped
            # network drive or a container mount may not support them, and the
            # failure looks like "Function not implemented: 'lib'". --copies
            # sidesteps it at the cost of a slightly larger directory.
            Write-Warn 'venv creation failed; retrying with --copies (no symlinks)'
            Remove-Item -LiteralPath $VenvPath -Recurse -Force -ErrorAction SilentlyContinue
            & $py -m venv --copies $VenvPath 2>&1 | Out-Null

            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPy)) {
                Write-Warn 'venv still failed. Falling back to the system Python.'
                Write-Warn 'If numpy/scipy are missing there, install them with:'
                Write-Warn '    python -m pip install --user numpy scipy'
                Write-Warn 'or re-run from a local (non-synced) folder.'
                Remove-Item -LiteralPath $VenvPath -Recurse -Force -ErrorAction SilentlyContinue
                $venvPy = $py
            }
        }
    }

    & $venvPy -c 'import numpy, scipy' 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($venvPy -eq $py) {
            throw ('numpy and scipy are missing from the system Python and ' +
                   'there is no virtual environment to install them into. ' +
                   'Run: python -m pip install --user numpy scipy')
        }
        Write-Host '  installing numpy and scipy ...'
        & $venvPy -m pip install --quiet --upgrade pip
        & $venvPy -m pip install --quiet numpy scipy
        if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
    }
    Write-Host '  numpy, scipy  : ok'

    # ----------------------------------------------------------- determinism --
    if (-not $SkipDeterminism) {
        Write-Step 'Step 1 of 3: determinism check'
        Write-Host '  Deciding whether n means items or calls. Nothing is sized until this answers.'

        if (-not (Test-Path -LiteralPath 'items_demo.jsonl')) {
            & $venvPy 'make_demo_items.py' --out 'items_demo.jsonl' --n 60 | Out-Null
        }

        $detArgs = @('determinism.py', '--items', 'items_demo.jsonl',
                     '--n', '20', '--repeats', '5', '--model', $Model,
                     '--obs', 'determinism_obs.jsonl')
        if ($Simulate) {
            $detArgs += '--simulate'
            if ($SimJitter -gt 0) { $detArgs += @('--sim-jitter', $SimJitter) }
        }

        # Tee-Object without capturing the pipeline output, so the check's own
        # report reaches the console as well as $detLines. Assigning the
        # pipeline to a variable would swallow it and the user would see an
        # empty step.
        # ForEach-Object "$_" flattens stderr ErrorRecords to plain strings.
        # Without it PowerShell renders every progress line the script writes to
        # stderr in red, which reads as a failure when nothing has failed.
        & $venvPy @detArgs 2>&1 |
            ForEach-Object { "$_" } |
            Tee-Object -Variable detLines |
            Out-Host
        $detText = ($detLines | Out-String)
        if ($LASTEXITCODE -ne 0) { throw 'determinism check failed' }

        if ($detText -match 'VERDICT: stochastic' -and $Repeats -le 1) {
            Write-Warn 'The model is stochastic but -Repeats is 1.'
            Write-Warn 'Repeated calls are needed, and within-item variance must'
            Write-Warn 'enter the power calculation. Re-run with -Repeats 3 (or'
            Write-Warn 'more), or pass -SkipDeterminism to override deliberately.'
            throw 'Stopping: run design does not match the determinism result.'
        }
        if ($detText -match 'VERDICT: deterministic' -and $Repeats -gt 1) {
            Write-Warn ("The model is deterministic, so -Repeats $Repeats buys " +
                        'nothing but cost. Continuing anyway.')
        }
    }
    else {
        Write-Step 'Step 1 of 3: determinism check SKIPPED'
        Write-Warn 'Only safe if you have already run it against this model version.'
    }

    # ------------------------------------------------------------------- run --
    Write-Step 'Step 2 of 3: the CCC factorial'

    $runArgs = @('ccc_jev_run.py', '--repo', $resolvedRepo, '--obs', $Obs,
                 '--repeats', $Repeats, '--workers', $Workers,
                 '--seed', $Seed, '--model', $Model)
    if ($Simulate) { $runArgs += '--simulate' }
    foreach ($d in $Domain)   { $runArgs += @('--domain', $d) }
    foreach ($e in $Encoding) { $runArgs += @('--encoding', $e) }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $venvPy @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'run failed' }
    $sw.Stop()
    Write-Host ("  elapsed: {0:n1}s" -f $sw.Elapsed.TotalSeconds)

    # -------------------------------------------------------------- analysis --
    Write-Step 'Step 3 of 3: both experiments'

    & $venvPy 'analyse_ccc.py' --obs $Obs
    if ($LASTEXITCODE -ne 0) { throw 'analysis failed' }

    $hasBoth = (-not $Encoding) -or ($Encoding.Count -gt 1)
    if ($hasBoth) {
        Write-Step 'Secondary: the structured encoding'
        & $venvPy 'analyse_ccc.py' --obs $Obs --encoding 'structured'
    }

    # --------------------------------------------------------- fingerprint --
    Write-Step 'Run fingerprint'
    $chkArgs = @('check_platform.py', '--obs', $Obs)
    if ($Against) { $chkArgs += @('--against', $Against) }
    & $venvPy @chkArgs
    $chkExit = $LASTEXITCODE
    if ($Against -and $chkExit -ne 0) {
        throw ('Cross-platform check FAILED: the two runs sent different ' +
               'requests. Do not pool them.')
    }

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
    Write-Host "  raw rows : $Obs"
    Write-Host '  re-analyse at any time without re-running:'
    Write-Host "      $venvPy analyse_ccc.py --obs $Obs"
}
catch {
    Write-Host ''
    Write-Host "FAILED: $_" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}

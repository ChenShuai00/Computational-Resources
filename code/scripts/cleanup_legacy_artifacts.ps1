param()

$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot

$targets = @(
    (Join-Path $root "meta_data"),
    (Join-Path $root "logs"),
    (Join-Path $root ".pytest_cache"),
    (Join-Path $root "pytest-cache-files-5c70yx53"),
    (Join-Path $root "pytest-cache-files-5xmglduw"),
    (Join-Path $root "pytest-cache-files-booq2ey0"),
    (Join-Path $root "pytest-cache-files-more4mx9"),
    (Join-Path $root "pytest-cache-files-swwb060u")
)

foreach ($target in $targets) {
    if (-not (Test-Path $target)) {
        Write-Host "[SKIP] Missing: $target"
        continue
    }

    try {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
        if (Test-Path $target) {
            Write-Warning "Target still exists after delete attempt: $target"
        }
        else {
            Write-Host "[OK] Removed: $target"
        }
    }
    catch {
        Write-Warning "Failed to remove $target"
        Write-Warning $_.Exception.Message
    }
}

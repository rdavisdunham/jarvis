$ErrorActionPreference = "Stop"
$JarvisDocker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$JarvisRoot = "\\wsl.localhost\Ubuntu\home\davin\jarvis"
$JarvisLog = "C:\Users\davin\.jarvis\startup.log"
try {
    & wsl.exe -d Ubuntu -- true
    $ErrorActionPreference = "Continue" # Docker writes normal progress to stderr. Exit codes are checked below.
    & $JarvisDocker desktop start --detach *> $JarvisLog
    $JarvisReady = $false
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        & $JarvisDocker info --format "{{.ServerVersion}}" *>> $JarvisLog
        if ($LASTEXITCODE -eq 0) { $JarvisReady = $true; break }
        Start-Sleep -Seconds 5
    }
    if (-not $JarvisReady) { throw "Docker did not become ready within five minutes." }
    & $JarvisDocker compose --env-file (Join-Path $JarvisRoot ".env.upgrade") -f (Join-Path $JarvisRoot "compose.upgrade.yml") up -d *>> $JarvisLog
    if ($LASTEXITCODE -ne 0) { throw "Jarvis startup failed. Review startup.log." }
} catch {
    $_.Exception.Message | Add-Content -LiteralPath $JarvisLog
    exit 1
}

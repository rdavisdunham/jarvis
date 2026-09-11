param(
    [ValidateSet("start", "status", "restart", "logs", "backup", "stop")]
    [string]$Action = "start"
)
$ErrorActionPreference = "Stop"
$JarvisRoot = Split-Path -Parent $PSScriptRoot
$DockerExe = (Get-Command docker.exe -ErrorAction SilentlyContinue).Source
if (-not $DockerExe) { $DockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe" }
$ComposeArguments = @("compose", "--env-file", (Join-Path $JarvisRoot ".env.upgrade"), "-f", (Join-Path $JarvisRoot "compose.upgrade.yml"))
switch ($Action) {
    "start" { & $DockerExe @ComposeArguments up -d --build }
    "status" { & $DockerExe @ComposeArguments ps }
    "restart" { & $DockerExe @ComposeArguments restart api worker }
    "logs" { & $DockerExe @ComposeArguments logs --tail 80 api worker backup }
    "backup" { & $DockerExe @ComposeArguments run --rm --no-deps backup once }
    "stop" { & $DockerExe @ComposeArguments stop }
}
exit $LASTEXITCODE

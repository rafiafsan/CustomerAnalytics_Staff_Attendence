# Run once in an elevated PowerShell on the edge PC:
#   powershell -ExecutionPolicy Bypass -File deploy\install_windows_task.ps1
$root = Split-Path -Parent $PSScriptRoot
$bat  = Join-Path $root "run_service.bat"

$action   = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName "RetailAI_Service" -Action $action -Trigger $trigger `
    -Settings $settings -User "SYSTEM" -RunLevel Highest -Force
Write-Host "Installed. Start now with: Start-ScheduledTask -TaskName RetailAI_Service"

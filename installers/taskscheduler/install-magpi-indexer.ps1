# install-magpi-indexer.ps1
# Register a Windows Scheduled Task to run `magpi index` hourly.
# Run from any PowerShell session (no admin required when using LogonType=Interactive).
#
#   powershell -ExecutionPolicy Bypass -File install-magpi-indexer.ps1
#
$action = New-ScheduledTaskAction -Execute "magpi" -Argument "index"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName "MagpieIndexer" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Magpi — incremental Claude Code transcript indexing" -Force
Write-Host "Registered task 'MagpieIndexer'. It will run hourly."

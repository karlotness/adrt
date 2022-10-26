$license_file = $args[0]
$patch_file = Join-Path -Path $PSScriptRoot -ChildPath "LICENSE_windows.txt"

Add-Content -Path $license_file -Value ("`r`n" + ("-" * 70) + "`r`n")
Get-content -Path $patch_file | Add-Content -Path $license_file

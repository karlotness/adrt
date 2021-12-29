$base_dir = $args[0]
$license_file = Join-Path -Path $base_dir -ChildPath "LICENSE.txt"
$patch_file = Join-Path -Path $base_dir -ChildPath "tools\LICENSE_windows.txt"

Add-Content -Path $license_file -Value ("`r`n" + ("-" * 70) + "`r`n")
Get-content -Path $patch_file | Add-Content -Path $license_file

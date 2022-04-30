$repo_base = $args[0]
$output_base = $args[1]
$license_file = Join-Path -Path $output_base -ChildPath "LICENSE.txt"
$patch_file = Join-Path -Path $repo_base -ChildPath "tools\LICENSE_windows.txt"

Add-Content -Path $license_file -Value ("`r`n" + ("-" * 70) + "`r`n")
Get-content -Path $patch_file | Add-Content -Path $license_file

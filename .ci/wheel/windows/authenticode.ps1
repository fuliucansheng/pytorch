$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
if (-not [IO.Path]::IsPathRooted($request.path) -or -not [IO.File]::Exists($request.path)) {
    throw 'Expected an existing absolute native-file path'
}
if (-not [IO.Path]::IsPathRooted($request.signtool) -or -not [IO.File]::Exists($request.signtool)) {
    throw 'An explicit installed SignTool executable is required'
}
$signature = Get-AuthenticodeSignature -LiteralPath $request.path -ErrorAction Stop
$signer = $null
if ($null -ne $signature.SignerCertificate) {
    $usage = @($signature.SignerCertificate.Extensions | Where-Object {$_.Oid.Value -eq '2.5.29.37'})
    $signer = [ordered]@{
        subject=$signature.SignerCertificate.Subject
        issuer=$signature.SignerCertificate.Issuer
        ekus=@($usage | ForEach-Object {$_.EnhancedKeyUsages} | ForEach-Object {$_.Value})
    }
}
$timestamp = $null
if ($null -ne $signature.TimeStamperCertificate) {
    $timestamp = [ordered]@{
        subject=$signature.TimeStamperCertificate.Subject
        issuer=$signature.TimeStamperCertificate.Issuer
        not_before=$signature.TimeStamperCertificate.NotBefore.ToUniversalTime().ToString('o')
        not_after=$signature.TimeStamperCertificate.NotAfter.ToUniversalTime().ToString('o')
    }
}
$ErrorActionPreference = 'Continue'
$verification = @(& $request.signtool verify /pa /all /v /tw $request.path 2>&1)
$verifyExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
[ordered]@{
    status=[string]$signature.Status
    signature_type=[string]$signature.SignatureType
    signer=$signer
    timestamp=$timestamp
    signtool_exit_code=$verifyExit
    signtool_output=($verification -join "`n").Replace($request.path, '<native-file>')
} | ConvertTo-Json -Depth 6 -Compress

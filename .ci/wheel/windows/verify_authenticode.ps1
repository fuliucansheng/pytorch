param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Assign the JSON array directly: PowerShell 5.1 otherwise nests it in @().
$items = Get-Content -LiteralPath $Manifest -Encoding UTF8 -Raw | ConvertFrom-Json
$hasher = [Security.Cryptography.SHA256]::Create()
try {
    $rows = @(
        foreach ($item in $items) {
            $signature = Get-AuthenticodeSignature -LiteralPath $item.file
            $certHash = $null
            $subject = $null
            if ($null -ne $signature.SignerCertificate) {
                $certHash = [BitConverter]::ToString(
                    $hasher.ComputeHash([byte[]]$signature.SignerCertificate.RawData)
                ).Replace('-', '').ToLowerInvariant()
                $subject = $signature.SignerCertificate.Subject
            }
            [ordered]@{
                path = $item.path
                status = [string]$signature.Status
                signature_type = [string]$signature.SignatureType
                signer_sha256 = $certHash
                subject = $subject
                timestamped = ($null -ne $signature.TimeStamperCertificate)
            }
        }
    )
    $json = ConvertTo-Json -InputObject $rows -Depth 5
    [IO.File]::WriteAllText($Output, $json, [Text.UTF8Encoding]::new($false))
} finally {
    $hasher.Dispose()
}

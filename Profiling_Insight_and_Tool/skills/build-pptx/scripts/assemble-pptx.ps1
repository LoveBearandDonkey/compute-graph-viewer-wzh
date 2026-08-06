param(
  [Parameter(Mandatory = $true)]
  [string]$SvgDir,
  [Parameter(Mandatory = $true)]
  [string]$OutputPath,
  [string]$RenderDir = "",
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$resolvedSvgDir = (Resolve-Path -LiteralPath $SvgDir).Path
$slides = @(Get-ChildItem -LiteralPath $resolvedSvgDir -Filter 'slide-*.svg' -File | Sort-Object Name)
if ($slides.Count -lt 1) { throw "No slide-*.svg files found in $resolvedSvgDir" }

$manifestPath = Join-Path $resolvedSvgDir 'deck-manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  if ([int]$manifest.slideCount -ne $slides.Count) {
    throw "Manifest expects $($manifest.slideCount) slides, found $($slides.Count)"
  }
}

$outputParent = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
if (-not (Test-Path -LiteralPath $outputParent)) {
  New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
$resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $resolvedOutput) {
  if (-not $Force) { throw "Output already exists. Pass -Force to replace exactly: $resolvedOutput" }
  Remove-Item -LiteralPath $resolvedOutput -Force
}

$resolvedRender = $null
if ($RenderDir) {
  $resolvedRender = [IO.Path]::GetFullPath($RenderDir)
  if (Test-Path -LiteralPath $resolvedRender) {
    $existing = @(Get-ChildItem -LiteralPath $resolvedRender -Force)
    if ($existing.Count -gt 0 -and -not $Force) {
      throw "Render directory is not empty. Pass -Force to replace generated PNG files: $resolvedRender"
    }
    if ($Force) {
      Get-ChildItem -LiteralPath $resolvedRender -Filter '*.PNG' -File | Remove-Item -Force
    }
  } else {
    New-Item -ItemType Directory -Path $resolvedRender -Force | Out-Null
  }
}

$app = $null
$presentation = $null
try {
  $app = New-Object -ComObject PowerPoint.Application
  $presentation = $app.Presentations.Add()
  $presentation.PageSetup.SlideWidth = 960
  $presentation.PageSetup.SlideHeight = 540

  foreach ($svg in $slides) {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $null = $slide.Shapes.AddPicture($svg.FullName, 0, -1, 0, 0, 960, 540)
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($slide)
  }

  $presentation.SaveAs($resolvedOutput, 24)
  if ($resolvedRender) {
    $presentation.Export($resolvedRender, 'PNG', 1600, 900)
  }
  $count = $presentation.Slides.Count
  $presentation.Close()
  $app.Quit()

  $rendered = if ($resolvedRender) { @(Get-ChildItem -LiteralPath $resolvedRender -Filter '*.PNG' -File).Count } else { 0 }
  if ($count -ne $slides.Count) { throw "PowerPoint saved $count slides, expected $($slides.Count)" }
  if ($resolvedRender -and $rendered -ne $slides.Count) { throw "PowerPoint rendered $rendered PNG files, expected $($slides.Count)" }

  [pscustomobject]@{
    Output = $resolvedOutput
    Slides = $count
    Rendered = $rendered
    Bytes = (Get-Item -LiteralPath $resolvedOutput).Length
  }
}
finally {
  if ($presentation) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
  if ($app) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) }
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}


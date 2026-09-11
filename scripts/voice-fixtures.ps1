# Synthetic inputs for the opt-in live Realtime test; never records a microphone.
$ErrorActionPreference = 'Stop'
$jarvisFixtureRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\.runtime'))
[IO.Directory]::CreateDirectory($jarvisFixtureRoot) | Out-Null
Add-Type -AssemblyName System.Speech
$jarvisFixtures = @{
    'voice-task.wav' = 'Please create a task called acceptance voice check silver meadow.'
    'voice-thanks.wav' = 'Thanks.'
}
foreach ($jarvisFixture in $jarvisFixtures.GetEnumerator()) {
    $jarvisSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $jarvisAudioFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
            48000,
            [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
            [System.Speech.AudioFormat.AudioChannel]::Mono
        )
        $jarvisSynth.SetOutputToWaveFile((Join-Path $jarvisFixtureRoot $jarvisFixture.Key), $jarvisAudioFormat)
        $jarvisSynth.Speak($jarvisFixture.Value)
    } finally {
        $jarvisSynth.Dispose()
    }
}
Write-Output 'Synthetic voice fixtures are ready in .runtime.'

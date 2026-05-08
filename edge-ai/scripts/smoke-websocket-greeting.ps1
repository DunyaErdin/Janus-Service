param(
    [string]$WsUrl = "ws://127.0.0.1:8080/ws/device",
    [string]$DeviceId = "janus-smoke-client",
    [string]$InteractionId = "wake-smoke-1",
    [string]$Text = "Size nasil yardimci olabilirim?",
    [string]$OutputPath = "janus-greeting-smoke.wav",
    [string]$DeviceToken = "",
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"

function Send-JsonMessage {
    param(
        [System.Net.WebSockets.ClientWebSocket]$Socket,
        [hashtable]$Message
    )

    $json = $Message | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $segment = [ArraySegment[byte]]::new($bytes)
    $Socket.SendAsync(
        $segment,
        [System.Net.WebSockets.WebSocketMessageType]::Text,
        $true,
        [Threading.CancellationToken]::None
    ).GetAwaiter().GetResult()
}

function Receive-JsonMessage {
    param(
        [System.Net.WebSockets.ClientWebSocket]$Socket,
        [int]$TimeoutSeconds
    )

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
    $buffer = [byte[]]::new(8192)
    $chunks = New-Object System.Collections.Generic.List[byte]

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $remaining = $deadline - [DateTimeOffset]::UtcNow
        $cts = [Threading.CancellationTokenSource]::new($remaining)
        try {
            do {
                $segment = [ArraySegment[byte]]::new($buffer)
                $result = $Socket.ReceiveAsync($segment, $cts.Token).GetAwaiter().GetResult()
                if ($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
                    throw "WebSocket closed by server."
                }
                if ($result.Count -gt 0) {
                    for ($index = 0; $index -lt $result.Count; $index++) {
                        $chunks.Add($buffer[$index])
                    }
                }
            } until ($result.EndOfMessage)

            $json = [System.Text.Encoding]::UTF8.GetString($chunks.ToArray())
            return $json | ConvertFrom-Json
        }
        finally {
            $cts.Dispose()
        }
    }

    throw "Timed out waiting for WebSocket message."
}

function Write-Pcm16MonoWav {
    param(
        [string]$Path,
        [byte[]]$PcmBytes,
        [int]$SampleRateHz,
        [int]$Channels
    )

    $writer = [System.IO.BinaryWriter]::new([System.IO.File]::Open($Path, [System.IO.FileMode]::Create))
    try {
        $byteRate = $SampleRateHz * $Channels * 2
        $blockAlign = $Channels * 2
        $dataSize = $PcmBytes.Length
        $riffSize = 36 + $dataSize

        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
        $writer.Write([uint32]$riffSize)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
        $writer.Write([uint32]16)
        $writer.Write([uint16]1)
        $writer.Write([uint16]$Channels)
        $writer.Write([uint32]$SampleRateHz)
        $writer.Write([uint32]$byteRate)
        $writer.Write([uint16]$blockAlign)
        $writer.Write([uint16]16)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $writer.Write([uint32]$dataSize)
        $writer.Write($PcmBytes)
    }
    finally {
        $writer.Dispose()
    }
}

$socket = [System.Net.WebSockets.ClientWebSocket]::new()
if ($DeviceToken) {
    $socket.Options.SetRequestHeader("x-janus-device-token", $DeviceToken)
}

try {
    $socket.ConnectAsync([Uri]$WsUrl, [Threading.CancellationToken]::None).GetAwaiter().GetResult()

    Send-JsonMessage -Socket $socket -Message @{
        message_type = "hello"
        device_id = $DeviceId
        protocol_version = "1.1"
        firmware_version = "smoke"
        capabilities = @("audio_in", "audio_out", "debug_client")
    }

    $helloAck = Receive-JsonMessage -Socket $socket -TimeoutSeconds $TimeoutSeconds
    Write-Host "received: $($helloAck.message_type) $($helloAck.detail)"

    Send-JsonMessage -Socket $socket -Message @{
        message_type = "greeting_request"
        device_id = $DeviceId
        interaction_id = $InteractionId
        text = $Text
        encoding = "pcm16"
        sample_rate_hz = 24000
        channels = 1
    }

    $audioBase64 = New-Object System.Text.StringBuilder
    $sampleRateHz = 24000
    $channels = 1

    while ($true) {
        $message = Receive-JsonMessage -Socket $socket -TimeoutSeconds $TimeoutSeconds
        Write-Host "received: $($message.message_type)"

        if ($message.message_type -eq "error") {
            throw "Server error $($message.code): $($message.message)"
        }

        if ($message.message_type -eq "audio_output_chunk") {
            [void]$audioBase64.Append($message.data_base64)
            $sampleRateHz = [int]$message.sample_rate_hz
            $channels = [int]$message.channels
        }

        if ($message.message_type -eq "audio_output_end") {
            break
        }
    }

    if ($audioBase64.Length -eq 0) {
        throw "No audio_output_chunk payload was received."
    }

    $pcm = [Convert]::FromBase64String($audioBase64.ToString())
    Write-Pcm16MonoWav -Path $OutputPath -PcmBytes $pcm -SampleRateHz $sampleRateHz -Channels $channels
    Write-Host "Wrote $OutputPath ($($pcm.Length) PCM bytes at ${sampleRateHz}Hz, channels=$channels)"
}
finally {
    if ($socket.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
        $socket.CloseAsync(
            [System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure,
            "smoke_done",
            [Threading.CancellationToken]::None
        ).GetAwaiter().GetResult()
    }
    $socket.Dispose()
}

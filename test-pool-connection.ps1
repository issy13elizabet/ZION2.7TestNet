# Test pool connection manually
Write-Host "Testing ZION pool connection on port 3333..." -ForegroundColor Green

try {
    $client = New-Object System.Net.Sockets.TcpClient("127.0.0.1", 3333)
    Write-Host "✅ Connected to pool successfully" -ForegroundColor Green
    
    $stream = $client.GetStream()
    $writer = New-Object System.IO.StreamWriter($stream)
    $reader = New-Object System.IO.StreamReader($stream)
    
    # Send login request
    $loginRequest = '{"id":1,"method":"login","params":{"login":"Z321nirFfsdGcAE8Loe1vzcZS9ztUvsTCYsYKYsncA63QqjqQyMnLiedvZSnniUsfE93Zdvu5tpkvC2qNVpDf4ot9q1UJUBMap","pass":"test","agent":"TestScript/1.0"}}'
    
    Write-Host "Sending login request..." -ForegroundColor Yellow
    $writer.WriteLine($loginRequest)
    $writer.Flush()
    
    # Read response
    $response = $reader.ReadLine()
    Write-Host "Pool response: $response" -ForegroundColor Cyan
    
    # Keep connection for few seconds
    Start-Sleep -Seconds 3
    
    $client.Close()
    Write-Host "✅ Test completed" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}
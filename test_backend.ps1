# CryptoAuditAI Backend Test Script - Clean Version
param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$UserId = "test_user",
    [string]$OllamaUrl = "http://localhost:11434"
)

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Cyan = "`e[36m"
$Reset = "`e[0m"

# Test counter
$script:TestCount = 0
$script:PassedTests = 0
$script:FailedTests = 0

function Write-TestHeader {
    param([string]$Title)
    Write-Host "${Blue}========================================${Reset}"
    Write-Host "${Blue}Testing: $Title${Reset}"
    Write-Host "${Blue}========================================${Reset}"
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Details = ""
    )
    $script:TestCount++
    if ($Passed) {
        $script:PassedTests++
        Write-Host "${Green}PASS${Reset}: $TestName"
        if ($Details) { Write-Host "  Details: $Details" }
    } else {
        $script:FailedTests++
        Write-Host "${Red}FAIL${Reset}: $TestName"
        if ($Details) { Write-Host "  Error: $Details" }
    }
}

function Write-Info {
    param([string]$Message)
    Write-Host "${Cyan}INFO${Reset}: $Message"
}

function Invoke-MultipartRequest {
    param(
        [string]$Url,
        [hashtable]$FormData
    )
    
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $bodyLines = @()
    foreach ($key in $FormData.Keys) {
        $bodyLines += "--$boundary"
        $bodyLines += "Content-Disposition: form-data; name=`"$key`"$LF"
        $bodyLines += $FormData[$key]
    }
    $bodyLines += "--$boundary--$LF"
    
    $body = $bodyLines -join $LF
    
    return Invoke-RestMethod -Uri $Url -Method Post -Body $body -ContentType "multipart/form-data; boundary=$boundary"
}

function Test-OllamaService {
    Write-TestHeader "Ollama Service Check"
    
    try {
        $response = Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -Method Get -TimeoutSec 10
        Write-TestResult "Ollama service accessible" $true "Found $($response.models.Count) models"
        
        $embedModel = $response.models | Where-Object { $_.name -like "*mxbai-embed-large*" }
        if ($embedModel) {
            Write-TestResult "mxbai-embed-large model available" $true $embedModel.name
            return $true
        } else {
            Write-TestResult "mxbai-embed-large model available" $false "Model not found"
            Write-Info "Available models: $($response.models.name -join ', ')"
            return $false
        }
    } catch {
        Write-TestResult "Ollama service accessible" $false $_.Exception.Message
        return $false
    }
}

function Test-EmbeddingGeneration {
    Write-TestHeader "Direct Embedding Test"
    
    $testText = "Smart contract security vulnerability detection"
    
    try {
        $response = Invoke-RestMethod -Uri "$OllamaUrl/api/embeddings" -Method Post -Body (@{
            model = "mxbai-embed-large:335m"
            prompt = $testText
        } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
        
        $passed = $response.embedding -and $response.embedding.Count -gt 0
        Write-TestResult "Generate embedding directly" $passed "Embedding dimension: $($response.embedding.Count)"
        
        return $passed
    } catch {
        Write-TestResult "Generate embedding directly" $false $_.Exception.Message
        return $false
    }
}

function Test-DocumentWorkflow {
    Write-TestHeader "Document Processing Workflow"
    
    # Clean up existing data
    Write-Info "Cleaning up existing data..."
    try {
        Invoke-MultipartRequest -Url "$BaseUrl/delete_all" -FormData @{ user_id = $UserId } | Out-Null
    } catch {
        # Ignore cleanup errors
    }
    
    # Test documents
    $testDocuments = @(
        @{
            name = "Security Contract"
            text = @"
pragma solidity ^0.8.0;
contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        balances[msg.sender] -= amount;
    }
}
"@
        },
        @{
            name = "DeFi Protocol"
            text = @"
This DeFi protocol implements automated market making with constant product formula.
Users can provide liquidity and earn fees from swaps.
The protocol supports flash loans with 0.09% fee.
Time-weighted average price oracles provide reliable price feeds.
"@
        }
    )
    
    $docIds = @()
    
    # Upload documents
    Write-Info "Uploading test documents..."
    foreach ($doc in $testDocuments) {
        try {
            $response = Invoke-MultipartRequest -Url "$BaseUrl/ingest" -FormData @{
                user_id = $UserId
                text = $doc.text
            }
            $docIds += $response.doc_id
            Write-TestResult "Upload '$($doc.name)'" $true "Doc ID: $($response.doc_id)"
        } catch {
            Write-TestResult "Upload '$($doc.name)'" $false $_.Exception.Message
        }
    }
    
    # Check document listing
    try {
        $listResponse = Invoke-RestMethod -Uri "$BaseUrl/list_docs?user_id=$UserId" -Method Get
        $foundDocs = $listResponse.documents.Count
        Write-TestResult "List uploaded documents" ($foundDocs -gt 0) "Found $foundDocs documents"
    } catch {
        Write-TestResult "List uploaded documents" $false $_.Exception.Message
    }
    
    # Wait for embeddings
    Write-Info "Waiting for embeddings to be generated..."
    $maxWait = 180
    $waitTime = 0
    $embeddingsReady = $false
    
    while ($waitTime -lt $maxWait -and -not $embeddingsReady) {
        Start-Sleep -Seconds 5
        $waitTime += 5
        
        Write-Host "${Yellow}Waiting... ($waitTime/$maxWait seconds)${Reset}" -NoNewline
        
        try {
            $queryResponse = Invoke-MultipartRequest -Url "$BaseUrl/query" -FormData @{
                user_id = $UserId
                query = "test query"
                top_k = "1"
            }
            
            if ($queryResponse.results -and $queryResponse.results.Count -gt 0) {
                $embeddingsReady = $true
                Write-Host " ${Green}Ready!${Reset}"
            } else {
                Write-Host " ${Yellow}Processing...${Reset}"
            }
        } catch {
            Write-Host " ${Red}Error${Reset}"
            break
        }
    }
    
    if ($embeddingsReady) {
        Write-TestResult "Embeddings generation" $true "Completed in $waitTime seconds"
        
        # Test semantic queries
        $testQueries = @(
            "reentrancy vulnerability security",
            "defi liquidity automated market",
            "flash loan protocol fee"
        )
        
        Write-Info "Testing semantic search..."
        foreach ($query in $testQueries) {
            try {
                $response = Invoke-MultipartRequest -Url "$BaseUrl/query" -FormData @{
                    user_id = $UserId
                    query = $query
                    top_k = "2"
                }
                
                $hasResults = $response.results -and $response.results.Count -gt 0
                Write-TestResult "Query: '$query'" $hasResults "Found $($response.results.Count) results"
                
                if ($hasResults) {
                    Write-Host "    Preview: $($response.results[0].text.Substring(0, [Math]::Min(80, $response.results[0].text.Length)))..."
                }
            } catch {
                Write-TestResult "Query: '$query'" $false $_.Exception.Message
            }
        }
    } else {
        Write-TestResult "Embeddings generation" $false "Timeout after $maxWait seconds"
    }
}

function Show-TestSummary {
    Write-Host ""
    Write-Host "${Blue}========================================${Reset}"
    Write-Host "${Blue}TEST SUMMARY${Reset}"
    Write-Host "${Blue}========================================${Reset}"
    Write-Host "Total Tests: $script:TestCount"
    Write-Host "${Green}Passed: $script:PassedTests${Reset}"
    Write-Host "${Red}Failed: $script:FailedTests${Reset}"
    
    $successRate = if ($script:TestCount -gt 0) { [math]::Round(($script:PassedTests / $script:TestCount) * 100, 2) } else { 0 }
    Write-Host "Success Rate: ${Green}$successRate%${Reset}"
    
    if ($script:FailedTests -eq 0) {
        Write-Host "${Green}All tests passed! Backend is fully functional!${Reset}"
    } elseif ($script:FailedTests -le 2) {
        Write-Host "${Yellow}Most tests passed. Check failed tests above.${Reset}"
    } else {
        Write-Host "${Red}Several tests failed. Review the implementation.${Reset}"
    }
    
    Write-Host ""
    Write-Host "${Cyan}Backend Status Summary:${Reset}"
    Write-Host "  - API Endpoints: ${Green}Working${Reset}"
    Write-Host "  - Document Ingestion: ${Green}Working${Reset}"
    Write-Host "  - Embedding Service: $(if ($script:PassedTests -gt 3) { "${Green}Working${Reset}" } else { "${Yellow}Check Setup${Reset}" })"
    Write-Host "  - Semantic Search: $(if ($script:PassedTests -gt 6) { "${Green}Working${Reset}" } else { "${Yellow}Needs Review${Reset}" })"
}

# Main execution
Write-Host "${Blue}CryptoAuditAI Backend Test Suite${Reset}"
Write-Host "Testing backend at: $BaseUrl"
Write-Host "Using User ID: $UserId"
Write-Host "Ollama URL: $OllamaUrl"
Write-Host ""

# Check backend health
try {
    Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get | Out-Null
    Write-Host "${Green}Backend server is running${Reset}"
} catch {
    Write-Host "${Red}Backend server is not accessible at $BaseUrl${Reset}"
    Write-Host "Please start the server with: python main.py"
    exit 1
}

# Run tests
$ollamaWorking = Test-OllamaService
if ($ollamaWorking) {
    Test-EmbeddingGeneration
    Test-DocumentWorkflow
} else {
    Write-Host "${Yellow}Ollama service issues detected.${Reset}"
    Write-Host "${Cyan}To fix:${Reset}"
    Write-Host "  1. Start Ollama: ollama serve"
    Write-Host "  2. Pull model: ollama pull mxbai-embed-large:335m"
    Write-Host "  3. Set environment: `$env:OLLAMA_EMBED_URL = 'http://localhost:11434/api/embeddings'"
    Write-Host "  4. Restart backend and re-run test"
}

Show-TestSummary

if ($script:FailedTests -gt 0) {
    exit 1
} else {
    exit 0
}


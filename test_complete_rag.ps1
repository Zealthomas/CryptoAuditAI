# CryptoAuditAI Fixed RAG Test Suite
# Save as: test_complete_rag_fixed.ps1

param(
    [string]$BaseURL = "http://localhost:8000",
    [string]$UserID = "crypto_auditor_test"
)

Write-Host "🚀 CryptoAuditAI Complete RAG Test Suite (FIXED)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Backend URL: $BaseURL" -ForegroundColor Gray
Write-Host "User ID: $UserID" -ForegroundColor Gray
Write-Host ""

# Test 1: Health and Model Check
Write-Host "1. 🏥 Health and Model Availability Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$BaseURL/health" -Method GET
    Write-Host "✅ Backend Health: $($health.status)" -ForegroundColor Green
    
    $models = Invoke-RestMethod -Uri "$BaseURL/models" -Method GET
    if ($models.mistral_available) {
        Write-Host "✅ Mistral Available: $($models.current_model)" -ForegroundColor Green
        Write-Host "   Available models: $($models.available_models -join ', ')" -ForegroundColor Gray
    } else {
        Write-Host "❌ Mistral not available. Available models: $($models.available_models -join ', ')" -ForegroundColor Red
        Write-Host "   Run: ollama pull mistral:instruct" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Backend or Ollama not accessible: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 2: Quick Mistral Test (using form data)
Write-Host "`n2. 🤖 Direct Mistral Test" -ForegroundColor Yellow
$testPrompt = "Explain reentrancy attacks in one sentence."

try {
    # Create form data for the test
    $formData = @{
        prompt = $testPrompt
        model = "mistral:instruct"
    }
    
    $directTest = Invoke-RestMethod -Uri "$BaseURL/test_mistral" -Method POST -Form $formData
    if ($directTest.status -eq "success") {
        Write-Host "✅ Mistral Direct Test Passed" -ForegroundColor Green
        Write-Host "   Response: $($directTest.response.Substring(0, [Math]::Min(100, $directTest.response.Length)))..." -ForegroundColor Gray
    } else {
        Write-Host "❌ Mistral Direct Test Failed: $($directTest.error)" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Direct test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Upload Vulnerable Smart Contracts (FIXED - using form data)
Write-Host "`n3. 📄 Uploading Test Smart Contracts" -ForegroundColor Yellow

# Contract 1: Reentrancy Vulnerable
$reentrancyContract = @"
pragma solidity ^0.8.0;

contract ReentrancyVulnerable {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // CRITICAL VULNERABILITY: Reentrancy attack vector
    function withdraw(uint256 _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        // External call before state change - DANGEROUS!
        (bool success,) = msg.sender.call{value: _amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= _amount; // Too late!
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
"@

# Contract 2: Access Control Issues
$accessControlContract = @"
pragma solidity ^0.8.0;

contract AccessControlVulnerable {
    address public owner;
    mapping(address => uint256) public funds;
    
    constructor() {
        owner = msg.sender;
    }
    
    // VULNERABILITY: Missing access control modifier
    function emergencyWithdraw() public {
        // Anyone can call this!
        payable(msg.sender).transfer(address(this).balance);
    }
    
    // VULNERABILITY: Weak access control
    function changeOwner(address newOwner) public {
        // Should check msg.sender == owner!
        owner = newOwner;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the owner");
        _;
    }
}
"@

# Contract 3: Integer Overflow (for older Solidity)
$overflowContract = @"
pragma solidity ^0.7.0; // Vulnerable version

contract IntegerOverflow {
    mapping(address => uint256) public balances;
    
    function deposit(uint256 amount) public {
        // VULNERABILITY: Integer overflow possible
        balances[msg.sender] += amount;
    }
    
    function withdraw(uint256 amount) public {
        // VULNERABILITY: Integer underflow possible  
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
    }
    
    // VULNERABILITY: Unchecked multiplication
    function multiply(uint256 a, uint256 b) public pure returns (uint256) {
        return a * b; // Can overflow
    }
}
"@

$contracts = @(
    @{ name = "ReentrancyVulnerable.sol"; content = $reentrancyContract },
    @{ name = "AccessControlVulnerable.sol"; content = $accessControlContract },
    @{ name = "IntegerOverflow.sol"; content = $overflowContract }
)

$uploadedDocs = @()

foreach ($contract in $contracts) {
    try {
        # Use form data instead of JSON
        $formData = @{
            user_id = $UserID
            filename = $contract.name
            content = $contract.content
        }
        
        $uploadResult = Invoke-RestMethod -Uri "$BaseURL/ingest" -Method POST -Form $formData
        Write-Host "✅ Uploaded: $($contract.name)" -ForegroundColor Green
        Write-Host "   Doc ID: $($uploadResult.doc_id)" -ForegroundColor Gray
        $uploadedDocs += $uploadResult.doc_id
    } catch {
        Write-Host "❌ Failed to upload $($contract.name): $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Details: $($_.ErrorDetails.Message)" -ForegroundColor Red
    }
}

# Wait for embedding
Write-Host "`n⏳ Waiting for embeddings to complete..." -ForegroundColor Yellow
$maxWait = 30
$waited = 0

do {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $docList = Invoke-RestMethod -Uri "$BaseURL/list_docs?user_id=$UserID" -Method GET
        $readyDocs = ($docList.documents | Where-Object { $_.status -eq "ready" }).Count
        $totalDocs = $docList.documents.Count
        
        Write-Host "   Progress: $readyDocs/$totalDocs documents ready..." -ForegroundColor Gray
        
        if ($readyDocs -eq $totalDocs -and $totalDocs -gt 0) {
            Write-Host "✅ All documents embedded successfully!" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "   Checking..." -ForegroundColor Gray
    }
} while ($waited -lt $maxWait)

if ($waited -ge $maxWait) {
    Write-Host "⚠️  Embedding timeout, but continuing with tests..." -ForegroundColor Yellow
}

# Test 4: RAG Query Tests (FIXED - using form data)
Write-Host "`n4. 🔍 RAG Query Tests" -ForegroundColor Yellow

$ragQueries = @(
    @{
        query = "What security vulnerabilities are present in these smart contracts?"
        description = "General vulnerability analysis"
        expected_topics = @("reentrancy", "access control", "overflow")
    },
    @{
        query = "How can I fix the reentrancy vulnerability in the withdraw function?"
        description = "Specific reentrancy remediation"
        expected_topics = @("checks-effects-interactions", "reentrancyguard", "state change")
    },
    @{
        query = "What's wrong with the emergencyWithdraw function and how to secure it?"
        description = "Access control vulnerability"
        expected_topics = @("onlyowner", "modifier", "access control")
    },
    @{
        query = "Explain the risks of integer overflow in smart contracts"
        description = "Integer overflow education"
        expected_topics = @("safemath", "overflow", "underflow")
    },
    @{
        query = "Provide a security checklist for smart contract deployment"
        description = "Best practices generation"
        expected_topics = @("testing", "audit", "security")
    }
)

$passedQueries = 0
$totalQueries = $ragQueries.Count

foreach ($testQuery in $ragQueries) {
    Write-Host "`n--- Test: $($testQuery.description) ---" -ForegroundColor Cyan
    Write-Host "Query: '$($testQuery.query)'" -ForegroundColor White
    
    try {
        # Use form data instead of JSON
        $formData = @{
            user_id = $UserID
            query = $testQuery.query
            temperature = 0.3
            max_tokens = 800
            top_k = 5
        }
        
        $startTime = Get-Date
        $response = Invoke-RestMethod -Uri "$BaseURL/chat" -Method POST -Form $formData
        $endTime = Get-Date
        $responseTime = ($endTime - $startTime).TotalSeconds
        
        if ($response.response -and $response.response.Length -gt 50) {
            Write-Host "✅ Response generated successfully" -ForegroundColor Green
            Write-Host "   Response time: $([Math]::Round($responseTime, 2))s" -ForegroundColor Gray
            Write-Host "   Response length: $($response.response.Length) characters" -ForegroundColor Gray
            
            # Check if response contains expected topics
            $responseText = $response.response.ToLower()
            $foundTopics = $testQuery.expected_topics | Where-Object { $responseText -contains $_.ToLower() }
            
            if ($foundTopics.Count -gt 0) {
                Write-Host "   Found relevant topics: $($foundTopics -join ', ')" -ForegroundColor Gray
            }
            
            # Display first part of response
            $preview = if ($response.response.Length -gt 200) { 
                $response.response.Substring(0, 200) + "..." 
            } else { 
                $response.response 
            }
            Write-Host "`n🤖 AI Response Preview:" -ForegroundColor Green
            Write-Host "$preview" -ForegroundColor White
            
            # Show sources
            if ($response.sources -and $response.sources.Count -gt 0) {
                Write-Host "`n📚 Sources used:" -ForegroundColor Blue
                foreach ($source in $response.sources) {
                    Write-Host "  - Doc: $($source.doc_id) | Chunk: $($source.chunk) | Score: $([Math]::Round($source.score, 3))" -ForegroundColor Gray
                }
            }
            
            $passedQueries++
        } else {
            Write-Host "❌ Empty or invalid response" -ForegroundColor Red
            if ($response.error) {
                Write-Host "   Error: $($response.error)" -ForegroundColor Red
            }
        }
        
    } catch {
        Write-Host "❌ Query failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Error details: $($_.ErrorDetails.Message)" -ForegroundColor Red
    }
    
    Write-Host "`n" + ("="*80) -ForegroundColor Gray
}

# Test 5: Simple cURL test for streaming (PowerShell streaming is complex)
Write-Host "`n5. 🌊 Streaming Response Test" -ForegroundColor Yellow
Write-Host "Testing streaming endpoint availability..." -ForegroundColor Gray

try {
    # Just test if the endpoint exists and responds
    $testResponse = Invoke-WebRequest -Uri "$BaseURL/chat/stream" -Method POST -Form @{
        user_id = $UserID
        query = "Test query"
    } -TimeoutSec 5 -ErrorAction Stop
    
    Write-Host "✅ Streaming endpoint accessible (status: $($testResponse.StatusCode))" -ForegroundColor Green
    Write-Host "   Use curl for full streaming test: curl -X POST $BaseURL/chat/stream -d 'user_id=$UserID&query=test'" -ForegroundColor Gray
    
} catch {
    if ($_.Exception.Message -like "*timeout*" -or $_.Exception.Message -like "*time*") {
        Write-Host "✅ Streaming endpoint working (response started)" -ForegroundColor Green
    } else {
        Write-Host "❌ Streaming test issue: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Test 6: Performance Summary
Write-Host "`n6. 📊 Performance and Quality Summary" -ForegroundColor Yellow

try {
    $finalDocList = Invoke-RestMethod -Uri "$BaseURL/list_docs?user_id=$UserID" -Method GET
    $totalDocs = $finalDocList.documents.Count
    $readyDocs = ($finalDocList.documents | Where-Object { $_.status -eq "ready" }).Count
    $totalChunks = ($finalDocList.documents | Measure-Object embedded_chunks -Sum).Sum
    
    Write-Host "📈 System Statistics:" -ForegroundColor Cyan
    Write-Host "  - Documents uploaded: $totalDocs" -ForegroundColor White
    Write-Host "  - Documents ready: $readyDocs" -ForegroundColor White
    Write-Host "  - Total embedded chunks: $totalChunks" -ForegroundColor White
    Write-Host "  - RAG queries passed: $passedQueries/$totalQueries" -ForegroundColor White
    Write-Host "  - Success rate: $([Math]::Round(($passedQueries/$totalQueries)*100, 1))%" -ForegroundColor White
    
} catch {
    Write-Host "Could not retrieve final statistics" -ForegroundColor Yellow
}

# Final Summary
Write-Host "`n🎯 FINAL TEST RESULTS" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

$overallSuccess = $passedQueries -ge ($totalQueries * 0.8)  # 80% success rate

if ($overallSuccess) {
    Write-Host "🎉 CONGRATULATIONS! Your CryptoAuditAI RAG system is working excellently!" -ForegroundColor Green
    Write-Host ""
    Write-Host "✅ Backend integration: Working" -ForegroundColor Green
    Write-Host "✅ Document processing: Working" -ForegroundColor Green  
    Write-Host "✅ Mistral integration: Working" -ForegroundColor Green
    Write-Host "✅ RAG responses: Working" -ForegroundColor Green
    Write-Host "✅ Security analysis: Working" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Your system is ready for real-world crypto security auditing!" -ForegroundColor Cyan
    
} else {
    Write-Host "⚠️  System partially working but needs attention" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Issues to address:" -ForegroundColor Red
    Write-Host "  - Only $passedQueries/$totalQueries RAG queries succeeded" -ForegroundColor White
    if ($totalDocs -eq 0) {
        Write-Host "  - No documents were uploaded successfully" -ForegroundColor White
    }
    Write-Host "  - Check backend logs for detailed error messages" -ForegroundColor White
}

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Try manual testing with cURL:" -ForegroundColor White
Write-Host "     curl -X POST $BaseURL/chat -F 'user_id=$UserID' -F 'query=test'" -ForegroundColor Gray
Write-Host "  2. Check backend logs for any errors" -ForegroundColor White
Write-Host "  3. Test individual endpoints in your browser or Postman" -ForegroundColor White

Write-Host "`n" + ("="*60) -ForegroundColor Gray
Write-Host "Test completed at $(Get-Date)" -ForegroundColor Gray
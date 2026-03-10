# Auto-Sync Script - Watches for changes and pushes to GitHub automatically

$projectPath = "C:\mini_project-main"
$checkInterval = 10

Write-Host "Auto-Sync Started - Watching for changes..." -ForegroundColor Green
Write-Host "Project: $projectPath" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Configure git if not already configured
Set-Location $projectPath
$gitUserName = git config user.name
$gitUserEmail = git config user.email

if (-not $gitUserName) {
    Write-Host "Configuring Git..." -ForegroundColor Yellow
    git config user.name "Auto-Sync Bot"
}

if (-not $gitUserEmail) {
    git config user.email "autosync@smartcrop.local"
}

Write-Host "Git configured for user: $(git config user.name)" -ForegroundColor Green
Write-Host ""

while ($true) {
    try {
        Set-Location $projectPath
        
        $gitStatus = git status --porcelain
        
        if ($gitStatus) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Changes detected!" -ForegroundColor Yellow
            
            git add -A
            Write-Host "Files staged" -ForegroundColor Green
            
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $commitMessage = "Auto-commit: Changes saved at $timestamp"
            
            git commit -m $commitMessage
            Write-Host "Committed: $commitMessage" -ForegroundColor Green
            
            $pushOutput = git push origin main 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Pushed to GitHub (main branch)" -ForegroundColor Cyan
            } else {
                Write-Host "Push failed:" -ForegroundColor Red
                Write-Host $pushOutput
                Write-Host "Run: git config --global credential.helper store" -ForegroundColor Yellow
            }
            Write-Host ""
        }
    }
    catch {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Error: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds $checkInterval
}

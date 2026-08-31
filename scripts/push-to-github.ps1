# Push this project to GitHub from your terminal (commits use YOUR git identity).
#
# First-time setup (repo already exists on GitHub with an initial commit):
#   .\scripts\push-to-github.ps1 -RemoteUrl "https://github.com/DMadan54/Monte-Carlo-NBA-Season-Simulator-ML.git" -Message "Add data pipeline and ML model scaffolding"
#
# Later pushes:
#   .\scripts\push-to-github.ps1 -Message "Describe what changed"

param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [string]$RemoteUrl = "",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Project: $ProjectRoot"
Write-Host "Commit author: $(git config user.name) <$(git config user.email)>"
Write-Host ""

if (-not (git config user.name) -or -not (git config user.email)) {
    Write-Error "Git user.name / user.email not set. Run: git config --global user.name `"Your Name`""
}

if (-not (Test-Path ".git")) {
    Write-Host "Initializing local git repo..."
    git init
    git branch -M $Branch
}

$remotes = git remote
if (-not $remotes) {
    if (-not $RemoteUrl) {
        Write-Error "No git remote configured. Pass -RemoteUrl on first run."
    }
    git remote add origin $RemoteUrl
    Write-Host "Added remote: origin -> $RemoteUrl"
}

Write-Host "Staging files (data/*.parquet excluded by .gitignore)..."
git add .

$status = git status --porcelain
if (-not $status) {
    Write-Host "Nothing to commit - working tree clean."
    exit 0
}

Write-Host ""
git status --short
Write-Host ""
Write-Host "Committing with message: $Message"
git commit -m $Message

$hasUpstream = git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
if (-not $hasUpstream) {
    $remoteHasCommits = git ls-remote --heads origin $Branch 2>$null
    if ($remoteHasCommits) {
        Write-Host "Pulling existing GitHub history (first sync)..."
        git pull origin $Branch --allow-unrelated-histories --no-edit
    }
    Write-Host "Pushing to origin/$Branch..."
    git push -u origin $Branch
} else {
    Write-Host "Pushing to origin/$Branch..."
    git push
}

Write-Host ""
Write-Host "Done."

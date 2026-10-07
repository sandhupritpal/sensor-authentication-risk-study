@echo off
setlocal enabledelayedexpansion
title Upload Sensor Project to GitHub
color 0A

echo ======================================================
echo    Sensor-Based Authentication Risk Study Uploader
echo ======================================================
echo.

set "PATH=C:\Program Files\Git\cmd;%PATH%"

echo Checking Git installation...
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git was not found in C:\Program Files\Git\cmd.
    echo Please make sure Git is installed.
    pause
    exit /b 1
)

echo.
echo Step 1: Open GitHub in your browser and make sure your repo is created:
echo   Name: sensor-authentication-risk-study
echo.
set /p GH_USER="Please enter your GitHub Username (e.g. pritpalsandhu): "

if "%GH_USER%"=="" (
    echo [ERROR] Username cannot be blank!
    pause
    exit /b 1
)

echo.
echo Setting up GitHub remote...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/%GH_USER%/sensor-authentication-risk-study.git

echo.
echo Uploading project to https://github.com/%GH_USER%/sensor-authentication-risk-study.git ...
echo.
echo (A browser popup may appear asking you to Sign In to GitHub. Just click "Sign in with your browser")
echo.

git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================
    echo   SUCCESS! Your project is now live on GitHub!
    echo   View it at: https://github.com/%GH_USER%/sensor-authentication-risk-study
    echo ======================================================
) else (
    echo.
    echo [!] If you got an authentication error, please verify that:
    echo     1. Your username is spelled correctly.
    echo     2. The repository 'sensor-authentication-risk-study' exists on GitHub.
)

echo.
pause

@echo off
cd /d C:\Windows
powershell -ExecutionPolicy Bypass -File "\\host.lan\Data\setup.ps1" >> "\\host.lan\Data\setup_log.txt" 2>&1

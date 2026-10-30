@echo off
rem Phase 0 of the repository migration (29 Oct 2026): commit everything git never saw, push, tag.
rem Double-click from the project root. It stops at the first error.
cd /d "%~dp0"
echo === Before ===
git status --short
git add .gitignore planet_layer scripts tests docs tools requirements-tested.txt "*.cmd" || goto fail
git commit -F outputs\test-tmp\pre-split-commit-msg.txt || goto fail
git push origin main || goto fail
git tag -a pre-split-2026-10-29 -F outputs\test-tmp\pre-split-tag-msg.txt || goto fail
git push origin pre-split-2026-10-29 || goto fail
echo.
echo === Done ===
git log -1 --format="%%h %%s"
echo Files in the commit:
git show --name-only --format= HEAD | find /c /v ""
git status --short
echo Phase 0 finished: commit and tag pre-split-2026-10-29 are on GitHub.
pause
exit /b 0
:fail
echo.
echo *** Stopped: the step above failed. Nothing after it ran. Copy the message above to Claude. ***
pause
exit /b 1

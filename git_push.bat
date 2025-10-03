@echo off
cd /d e:\
git add -A
git commit -m "Fix hybrid mining: Xmrig 10 threads, SRBMiner stderr monitoring, debug output"
git push
echo Git operations completed!
pause
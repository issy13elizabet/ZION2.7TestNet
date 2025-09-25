# Session: SSH ports & keys
- Time: 2025-09-23T
- Changes:
  - scripts/ssh-redeploy-pool.ps1: add -KeyPath support
  - scripts/ssh-patch-ports.ps1: new script to publish 443/80 and restart uzi-pool
- Next:
  - Run ssh-patch-ports.ps1 with repo key
  - Test-NetConnection 443/80

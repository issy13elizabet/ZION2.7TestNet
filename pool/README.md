Zion Stratum Stub (Node.js)

Dočasný lehký Stratum-like server, který otevře port 3333 a umožní klientům (např. XMRig) navázat spojení, provést subscribe/authorize a dostat dummy joby. Slouží k ověření síťové konektivity a k odblokování portu 3333, než bude integrován plnohodnotný pool.

Spuštění lokálně:
- npm install
- npm start

Proměnné:
- POOL_PORT (default 3333)
- POOL_BIND (default 0.0.0.0)

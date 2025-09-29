#!/bin/bash

# ZION Wallet Security Setup
# Generuje bezpečné heslo pro wallet service

echo "🔐 ZION Wallet Security Setup"
echo "=============================="

# Generuj silné heslo
if command -v openssl &> /dev/null; then
    NEW_PASSWORD=$(openssl rand -base64 48 | tr -d '\n')
    echo "✅ Nové silné heslo vygenerováno pomocí OpenSSL"
elif command -v pwgen &> /dev/null; then
    NEW_PASSWORD=$(pwgen -s 64 1)
    echo "✅ Nové silné heslo vygenerováno pomocí pwgen"
else
    # Fallback pro systémy bez openssl/pwgen
    NEW_PASSWORD="Zion$(date +%s%N | sha256sum | cut -c1-32)Bootstrap!$(shuf -i 1000-9999 -n 1)"
    echo "⚠️  Použit fallback generátor (doporučuje se nainstalovat OpenSSL)"
fi

# Updatuj .env soubor
if [ -f ".env" ]; then
    # Backup original
    cp .env .env.backup
    
    # Update password
    sed -i "s/WALLET_PASSWORD=.*/WALLET_PASSWORD=${NEW_PASSWORD}/" .env
    echo "✅ Heslo uloženo do .env souboru"
    echo "💾 Backup původního .env vytvořen jako .env.backup"
else
    # Vytvoř nový .env
    echo "WALLET_PASSWORD=${NEW_PASSWORD}" > .env
    echo "✅ Nový .env soubor vytvořen"
fi

echo ""
echo "🔒 Bezpečnostní doporučení:"
echo "   - .env soubor není commitnutý do Git (.gitignore)"
echo "   - V produkci použij ještě silnější heslo"
echo "   - Pravidelně rotuj hesla"
echo "   - Neukládej hesla do plaintext souborů"
echo ""
echo "📋 Pro restart wallet service:"
echo "   docker-compose -p zion-bootstrap -f docker-compose-bootstrap.yml restart wallet-service"
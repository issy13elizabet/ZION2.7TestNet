#!/usr/bin/env python3
"""
ZION 2.7.4 - Security Audit Tool
Kontrola všech souborů na citlivé informace před veřejným releasem
"""

import os
import re
import sys
import hashlib
from pathlib import Path

class SecurityAudit:
    """Bezpečnostní audit nástroj"""
    
    def __init__(self):
        self.sensitive_patterns = [
            # Private keys patterns
            r'[0-9a-fA-F]{64}',  # 64 hex chars (private key)
            r'[0-9a-fA-F]{32}',  # 32 hex chars (shorter key)
            
            # Real addresses patterns  
            r'ZION[a-zA-Z0-9]{30,}',  # ZION addresses
            r'Z[0-9a-zA-Z]{50,}',     # Long Z addresses
            r'MAITREYA_BUDDHA_NETWORK_ADMINISTRATOR_[0-9]{4}',
            
            # Wallet patterns
            r'"private_key":\s*"[^"]*[0-9a-fA-F]{20,}',
            r'"mnemonic":\s*"[^"]*seed[^"]*"',
            r'"address":\s*"[A-Za-z0-9]{30,}"',
            
            # Configuration secrets
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'secret\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{16,}["\']',
            
            # Database connections
            r'mongodb://[^"\'\\s]+',
            r'mysql://[^"\'\\s]+',
            r'postgresql://[^"\'\\s]+',
        ]
        
        self.whitelist_patterns = [
            r'\[REDACTED.*\]',
            r'\[EXTERNAL_BACKUP_ONLY\]',
            r'placeholder',
            r'example',
            r'demo',
            r'test'
        ]
        
        self.suspicious_files = []
        self.clean_files = []
        
    def is_whitelisted(self, text):
        """Kontrola, zda je text v whitelist"""
        for pattern in self.whitelist_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
        
    def scan_file(self, filepath):
        """Skenování jednoho souboru"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            findings = []
            
            for pattern in self.sensitive_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip if whitelisted
                    if self.is_whitelisted(match.group()):
                        continue
                        
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append({
                        'pattern': pattern,
                        'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group(),
                        'line': line_num,
                        'context': self.get_context(content, match.start())
                    })
            
            if findings:
                self.suspicious_files.append({
                    'file': str(filepath),
                    'findings': findings
                })
            else:
                self.clean_files.append(str(filepath))
                
        except Exception as e:
            print(f"❌ Chyba při skenování {filepath}: {e}")
    
    def get_context(self, content, position):
        """Získání kontextu kolem nálezu"""
        lines = content.split('\n')
        line_num = content[:position].count('\n')
        
        start = max(0, line_num - 2)
        end = min(len(lines), line_num + 3)
        
        context_lines = []
        for i in range(start, end):
            marker = '>>>' if i == line_num else '   '
            context_lines.append(f"{marker} {i+1:3d}: {lines[i][:80]}")
            
        return '\n'.join(context_lines)
    
    def scan_directory(self, directory, extensions=None):
        """Skenování celého adresáře"""
        if extensions is None:
            extensions = ['.py', '.js', '.json', '.md', '.txt', '.yml', '.yaml', '.sh', '.env']
            
        directory_path = Path(directory)
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                # Skip binary files and specific directories
                if any(skip in str(file_path) for skip in ['.git', 'node_modules', '__pycache__', '.next']):
                    continue
                    
                if file_path.suffix.lower() in extensions or file_path.name.startswith('.env'):
                    self.scan_file(file_path)
    
    def generate_report(self):
        """Generování bezpečnostní zprávy"""
        print("🔍 ZION 2.7.4 - BEZPEČNOSTNÍ AUDIT REPORT")
        print("=" * 60)
        
        if self.suspicious_files:
            print(f"⚠️  NALEZENO {len(self.suspicious_files)} PODEZŘELÝCH SOUBORŮ:")
            print()
            
            for file_info in self.suspicious_files:
                print(f"📁 {file_info['file']}")
                for finding in file_info['findings']:
                    print(f"   🚨 Pattern: {finding['pattern']}")
                    print(f"   📍 Line {finding['line']}: {finding['match']}")
                    print(f"   📄 Context:")
                    for line in finding['context'].split('\n'):
                        print(f"      {line}")
                    print()
                print("-" * 40)
                
            print(f"❌ AUDIT FAILED - {len(self.suspicious_files)} souborů vyžaduje úpravu!")
            return False
        else:
            print("✅ AUDIT PASSED - Žádné citlivé informace nenalezeny!")
            print(f"📊 Zkontrolováno {len(self.clean_files)} souborů")
            return True
    
    def fix_suggestions(self):
        """Návrhy na opravu"""
        if self.suspicious_files:
            print("\n🔧 NÁVHRY NA OPRAVU:")
            print("1. Nahraďte skutečné adresy za [REDACTED_*] placeholders")
            print("2. Přesuňte private keys do externí zálohy")  
            print("3. Použijte environment variables pro konfigurace")
            print("4. Zkontrolujte .gitignore pravidla")
            print("5. Implementujte runtime key generation")

def main():
    """Hlavní funkce"""
    if len(sys.argv) > 1:
        scan_path = sys.argv[1]
    else:
        scan_path = "."
        
    print(f"🔍 Spouštím bezpečnostní audit: {scan_path}")
    print()
    
    auditor = SecurityAudit()
    auditor.scan_directory(scan_path)
    
    success = auditor.generate_report()
    auditor.fix_suggestions()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🔄 ZION 2.7 BLOCKCHAIN ADAPTER 🔄
Adapter pro přechod ze starého na optimalizované úložiště
Zajišťuje backward compatibility během migrace
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sqlite3

class BlockchainStorageAdapter:
    def __init__(self, data_dir: str = "2.7/data"):
        self.data_dir = Path(data_dir)
        self.blocks_dir = self.data_dir / "blocks"
        self.optimized_dir = self.data_dir / "optimized"
        self.db_path = self.data_dir / "blockchain.db"
        
        # Detekce dostupných úložišť
        self.has_legacy = self.blocks_dir.exists() and list(self.blocks_dir.glob("*.json"))
        self.has_optimized = self.optimized_dir.exists() and self.db_path.exists()
        
        print(f"💾 Storage Adapter initialized:")
        print(f"   Legacy storage: {'✅' if self.has_legacy else '❌'}")
        print(f"   Optimized storage: {'✅' if self.has_optimized else '❌'}")
    
    def get_block(self, height: int = None, block_hash: str = None) -> Optional[Dict[str, Any]]:
        """Univerzální metoda pro načtení bloku z jakéhokoliv úložiště"""
        
        # Pokus o optimalizované úložiště nejdříve
        if self.has_optimized:
            block = self._get_from_optimized(height, block_hash)
            if block:
                return block
        
        # Fallback na legacy úložiště
        if self.has_legacy:
            block = self._get_from_legacy(height, block_hash)
            if block:
                return block
        
        return None
    
    def _get_from_optimized(self, height: int = None, block_hash: str = None) -> Optional[Dict[str, Any]]:
        """Načte blok z optimalizovaného úložiště"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if height is not None:
                    cursor = conn.execute(
                        "SELECT file_path, file_offset FROM blocks WHERE height = ?", 
                        (height,)
                    )
                elif block_hash:
                    cursor = conn.execute(
                        "SELECT file_path, file_offset FROM blocks WHERE hash = ?", 
                        (block_hash,)
                    )
                else:
                    return None
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                file_path, offset = result
                
                # Načti batch soubor
                with open(file_path, 'r') as f:
                    batch_data = json.load(f)
                
                # Najdi blok v batchi
                for block in batch_data['blocks']:
                    if height is not None and block['height'] == height:
                        return block
                    elif block_hash and block['hash'] == block_hash:
                        return block
                
        except Exception as e:
            print(f"⚠️ Error reading from optimized storage: {e}")
            return None
    
    def _get_from_legacy(self, height: int = None, block_hash: str = None) -> Optional[Dict[str, Any]]:
        """Načte blok ze starého úložiště"""
        try:
            if height is not None:
                # Zkus hlavní soubor
                file_path = self.blocks_dir / f"{height:08d}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        return json.load(f)
                
                # Zkus hledat podle patternu
                pattern = f"{height:08d}_*.json"
                files = list(self.blocks_dir.glob(pattern))
                if files:
                    with open(files[0], 'r') as f:
                        return json.load(f)
            
            elif block_hash:
                # Hledání podle hash - musíme projít všechny soubory
                for file_path in self.blocks_dir.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            block = json.load(f)
                            if block.get('hash') == block_hash:
                                return block
                    except:
                        continue
        
        except Exception as e:
            print(f"⚠️ Error reading from legacy storage: {e}")
            return None
    
    def list_blocks(self, start_height: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Vrací seznam bloků (univerzálně)"""
        blocks = []
        
        # Optimalizované úložiště
        if self.has_optimized:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT height, hash, file_path, file_offset 
                        FROM blocks 
                        WHERE height >= ? 
                        ORDER BY height 
                        LIMIT ?
                    """, (start_height, limit))
                    
                    batch_cache = {}
                    
                    for row in cursor.fetchall():
                        height, block_hash, file_path, offset = row
                        
                        # Cache batch files
                        if file_path not in batch_cache:
                            with open(file_path, 'r') as f:
                                batch_cache[file_path] = json.load(f)
                        
                        # Najdi blok v batchi
                        for block in batch_cache[file_path]['blocks']:
                            if block['height'] == height:
                                blocks.append(block)
                                break
                
                return blocks
            
            except Exception as e:
                print(f"⚠️ Error listing from optimized storage: {e}")
        
        # Fallback na legacy
        if self.has_legacy and not blocks:
            try:
                for i in range(start_height, start_height + limit):
                    block = self._get_from_legacy(height=i)
                    if block:
                        blocks.append(block)
                    else:
                        break  # Žádné další bloky
            except Exception as e:
                print(f"⚠️ Error listing from legacy storage: {e}")
        
        return blocks
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Vrací informace o blockchainu"""
        info = {
            'storage_type': 'hybrid',
            'legacy_available': self.has_legacy,
            'optimized_available': self.has_optimized,
            'height': 0,
            'total_blocks': 0,
            'latest_hash': None,
            'latest_timestamp': None
        }
        
        # Načti nejnovější blok
        if self.has_optimized:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT height, hash, timestamp, COUNT(*) as total
                        FROM blocks 
                        ORDER BY height DESC 
                        LIMIT 1
                    """)
                    row = cursor.fetchone()
                    if row:
                        height, block_hash, timestamp, total = row
                        info.update({
                            'height': height,
                            'total_blocks': total,
                            'latest_hash': block_hash,
                            'latest_timestamp': timestamp
                        })
            except Exception as e:
                print(f"⚠️ Error getting info from optimized storage: {e}")
        
        elif self.has_legacy:
            # Najdi nejvyšší blok v legacy storage
            max_height = -1
            for file_path in self.blocks_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        block = json.load(f)
                        if 'height' in block and block['height'] > max_height:
                            max_height = block['height']
                            info.update({
                                'height': block['height'],
                                'latest_hash': block['hash'],
                                'latest_timestamp': block['timestamp']
                            })
                except:
                    continue
            
            info['total_blocks'] = max_height + 1 if max_height >= 0 else 0
        
        return info


def test_adapter():
    """Test adaptéru úložiště"""
    print("🧪 TESTING BLOCKCHAIN STORAGE ADAPTER")
    print("=" * 50)
    
    adapter = BlockchainStorageAdapter()
    
    # Test základních informací
    info = adapter.get_blockchain_info()
    print(f"\n📊 Blockchain Info:")
    print(f"   Height: {info['height']}")
    print(f"   Total blocks: {info['total_blocks']}")
    print(f"   Latest hash: {info['latest_hash'][:16] if info['latest_hash'] else 'None'}...")
    print(f"   Storage type: {info['storage_type']}")
    
    # Test načtení konkrétních bloků
    print(f"\n🔍 Block Tests:")
    
    # Genesis blok
    block0 = adapter.get_block(height=0)
    if block0:
        print(f"   ✅ Block 0: {block0['hash'][:16]}...")
    else:
        print(f"   ❌ Block 0: Not found")
    
    # Nejnovější blok
    latest_block = adapter.get_block(height=info['height'])
    if latest_block:
        print(f"   ✅ Block {info['height']}: {latest_block['hash'][:16]}...")
        print(f"      Difficulty: {latest_block['difficulty']:,}")
    else:
        print(f"   ❌ Block {info['height']}: Not found")
    
    # Test podle hash
    if info['latest_hash']:
        block_by_hash = adapter.get_block(block_hash=info['latest_hash'])
        if block_by_hash:
            print(f"   ✅ By hash: Block {block_by_hash['height']}")
        else:
            print(f"   ❌ By hash: Not found")
    
    # Test seznamu bloků
    blocks = adapter.list_blocks(0, 5)
    print(f"\n📋 Block List (first 5):")
    for block in blocks[:5]:
        print(f"   Block {block['height']}: {block['hash'][:16]}...")
    
    print(f"\n🎉 ADAPTER TEST COMPLETE!")


if __name__ == "__main__":
    test_adapter()
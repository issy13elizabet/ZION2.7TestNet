#!/usr/bin/env python3
"""
PDF Text Extractor for ZION V2 Legacy Books
Extrahuje text ze všech PDF knih do textových souborů
"""

import PyPDF2
import os
import sys

def extract_text_from_pdf(pdf_path, output_path):
    """Extrahuje text z PDF a uloží do textového souboru"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📖 Extracting from: {os.path.basename(pdf_path)}")
            print(f"   Pages: {len(pdf_reader.pages)}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n\n=== PAGE {page_num + 1} ===\n\n"
                    text += page_text
                except Exception as e:
                    print(f"   Error extracting page {page_num + 1}: {e}")
                    continue
            
            # Uložit text
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            
            print(f"✅ Extracted to: {os.path.basename(output_path)}")
            print(f"   Text length: {len(text)} characters\n")
            
            return True
            
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return False

def main():
    """Hlavní funkce - extrahuje všechny PDF knihy"""
    
    # Cesty
    cz_folder = "/Users/yose/Zion/V2/src/cz"
    output_folder = "/Users/yose/Zion/V2/src/text_extracts"
    
    # Vytvořit výstupní složku
    os.makedirs(output_folder, exist_ok=True)
    
    # Seznam PDF souborů
    pdf_files = [
        "CosmicEgg.pdf",
        "Dohrmanovo-proroctvi.pdf", 
        "OmnityOneLove CZ.pdf",
        "SmaragdoveDesky.pdf",
        "Starobyly_sip.pdf",
        "Tajemství amenti.PDF"
    ]
    
    print("🌟 ZION V2 LEGACY BOOKS - PDF TEXT EXTRACTION")
    print("=" * 50)
    
    success_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(cz_folder, pdf_file)
        txt_file = pdf_file.replace('.pdf', '.txt').replace('.PDF', '.txt')
        txt_path = os.path.join(output_folder, txt_file)
        
        if os.path.exists(pdf_path):
            if extract_text_from_pdf(pdf_path, txt_path):
                success_count += 1
        else:
            print(f"❌ File not found: {pdf_file}")
    
    print("=" * 50)
    print(f"🎉 EXTRACTION COMPLETE: {success_count}/{len(pdf_files)} books extracted!")
    print(f"📁 Text files saved in: {output_folder}")

if __name__ == "__main__":
    main()
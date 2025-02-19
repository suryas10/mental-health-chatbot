from pathlib import Path
from PyPDF2 import PdfReader

def verify_pdfs():
    database_dir = Path("database")
    
    if not database_dir.exists():
        print("Database directory not found!")
        return
        
    pdf_files = list(database_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in database directory!")
        return
        
    print(f"Found {len(pdf_files)} PDF files:")
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            print(f"\n- {pdf_path.name}")
            print(f"  Pages: {num_pages}")
            print(f"  First page preview: {reader.pages[0].extract_text()[:100]}...")
        except Exception as e:
            print(f"\nError reading {pdf_path.name}: {e}")

if __name__ == "__main__":
    verify_pdfs()
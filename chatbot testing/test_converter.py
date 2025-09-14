from Converter import PDFParser

parser = PDFParser()
doc = parser.parse_pdf("C:/Users/aksha/OneDrive/Desktop/sample.pdf", source_url="local-test")


print("Parsed Document ID:", doc["_id"])
print("Language Detected:", doc["lang"])
print("Chunks stored:", len(doc["chunks"]))
print("Tables found:", len(doc["tables"]))

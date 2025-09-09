from docling.document_converter import DocumentConverter
# from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

result = converter.convert("https://www.notta.ai/en/blog/roblox-statistics")
document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

print(markdown_output)

result2 = converter.convert(r"D:\market research ai\2024_Newzoo_Global_Games_Market_Report.pdf")
result3 = converter.convert(r"D:\market research ai\2025 Africa Games Industry Report.pdf")
result4 = converter.convert(r"D:\market research ai\2025-Africa-Games-Industry-Report-1.pdf")

doc1 = result2.document
doc2 = result3.document
doc3 = result4.document

mark2 = doc1.export_to_markdown()
mark3= doc2.export_to_markdown()
mark4 = doc3.export_to_markdown()

json_out1 = doc1.export_to_dict()
json_out2 = doc2.export_to_dict()
json_out3 = doc3.export_to_dict()

# print(mark2)
# print(mark3)
# print(mark4)

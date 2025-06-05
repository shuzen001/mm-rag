from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
import os


def extract_pdf_elements(pdf_dir, fname, output_dir=None):
    """從 PDF 檔案中提取圖片、表格與分段文字。

    Parameters
    ----------
    pdf_dir : str
        PDF 檔案所在資料夾路徑。
    fname : str
        PDF 檔名。
    output_dir : str, optional
        圖片輸出資料夾路徑，預設與 ``pdf_dir`` 相同。

    Returns
    -------
    list
        unstructured 解析後的元素列表（圖片、表格、文字等）。
    """

    if output_dir is None:
        output_dir = pdf_dir

    return partition_pdf(
        filename=os.path.join(pdf_dir, fname),
        extract_images_in_pdf=True,
        extract_tables=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_dir,
    )

# 目的：方便後續針對不同型態資料進行摘要與檢索。
def categorize_elements(raw_pdf_elements):
    """
    將 PDF 提取的元素分類為表格與文字。
    raw_pdf_elements: unstructured.documents.elements 的列表
    回傳：texts（文字列表）、tables（表格列表）
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


if __name__ == "__main__":
    print(
        "Wrong file execution. Please use the build_vector_db.py to build up the vector database. Then use the main.py to run the mm-RAG."
    )

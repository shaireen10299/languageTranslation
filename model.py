import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PyPDF2 import PdfReader
from fpdf import FPDF
import os
import re
import difflib
from collections import Counter

# Try importing sacrebleu for optional NLP evaluation metrics; 
# if not installed, the code will seamlessly fall back to custom math metrics.
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

# Optional Semantic metrics (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer, util
    # Loads a tiny, fast, offline-compatible semantic model
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

# --- Configuration & Paths ---
# MODEL_FORWARD = "ai4bharat/indictrans2-en-indic-dist-200M"
# MODEL_REVERSE = "ai4bharat/indictrans2-indic-en-dist-200M"
# --- Configuration & Paths ---
MODEL_FORWARD = "./local_models/en-indic"
MODEL_REVERSE = "./local_models/indic-en"

INPUT_PDF = "/home/sdc-delhi/PaperTranslation/SSC QUESTION PAPER/ENGLISH/SSC-CGL-12-Sep-2025-S1-English.pdf"
LANGUAGES_FILE = "languages.json"
FONTS_DIR = "fonts"
BATCH_SIZE = 16

# Ensure required output directories exist so file saving never fails
os.makedirs("Output", exist_ok=True)
os.makedirs("Evaluation", exist_ok=True)

# Maps the 4-letter script suffix of IndicTrans2 codes to the corresponding TrueType font file
SCRIPT_TO_FONT = {
    "Deva": "NotoSansDevanagari.ttf",
    "Beng": "NotoSansBengali.ttf",
    "Gujr": "NotoSansGujarati.ttf",
    "Guru": "NotoSansGurmukhi.ttf",
    "Knda": "NotoSansKannada.ttf",
    "Mlym": "NotoSansMalayalam.ttf",
    "Orya": "NotoSansOriya.ttf",
    "Taml": "NotoSansTamil.ttf",
    "Telu": "NotoSansTelugu.ttf",
    "Arab": "NotoSansArabic.ttf",
    "Mtei": "NotoSansMeeteiMayek.ttf",
    "Olck": "NotoSansOlChiki.ttf",
}


def get_font_path(target_code):
    """Automatically picks the correct .ttf font file based on the script suffix of the selected language code."""
    script_code = target_code.split("_")[-1] if "_" in target_code else "Deva"
    font_filename = SCRIPT_TO_FONT.get(script_code, "NotoSansDevanagari.ttf")

    # 1. Look inside the dedicated ./fonts/ directory
    path_in_fonts_dir = os.path.join(FONTS_DIR, font_filename)
    if os.path.exists(path_in_fonts_dir):
        return path_in_fonts_dir

    # 2. Look in the base project directory
    if os.path.exists(font_filename):
        return font_filename

    # 3. Fallback warning if the specific font is missing
    if os.path.exists("NotoSansDevanagari.ttf"):
        print(f"⚠️ Warning: '{font_filename}' not found! Falling back to 'NotoSansDevanagari.ttf'.")

    raise FileNotFoundError(
        f"Font file '{font_filename}' not found in './{FONTS_DIR}/' or base directory. "
        f"Please download it first."
    )


def load_languages(filepath):
    """Loads language name and code mappings from the JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_target_language(languages_dict):
    """Displays a numbered list in the terminal so the user can pick a target translation language."""
    print("\n--- Available Languages ---")
    lang_names = list(languages_dict.keys())
    for i, lang in enumerate(lang_names, 1):
        print(f"{i}. {lang}")

    while True:
        try:
            choice = int(input("\nEnter the number of your target language: "))
            if 1 <= choice <= len(lang_names):
                selected_lang_name = lang_names[choice - 1]
                selected_code = languages_dict[selected_lang_name]
                print(f"\n✅ Selected: {selected_lang_name} (Code: {selected_code})")
                return selected_code, selected_lang_name
            else:
                print("Invalid choice. Please select a valid number from the list.")
        except ValueError:
            print("Please enter a valid number.")


def clean_raw_line(line):
    """Strips out common test portal watermarks and unsupported decorative symbols from PDF text."""
    line = re.sub(r'\b(Adda247|Adda\s*247|Auda\s*247|A Google Play)\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'[★▼←•]', '', line)
    return line.strip()


def is_standalone_series_or_code(line):
    """Identifies standalone reasoning patterns like 'COLD: FRPE :: WARM:?' or uppercase alphanumeric lists."""
    if "::" in line:
        return True
    if not re.search(r'[a-z]', line) and len(line.split()) <= 10:
        return True
    return False


def structure_mcq_page(page_text):
    """Parses raw PDF lines into structured MCQ elements (keeping questions, options, and answers separate)."""
    raw_lines = page_text.split('\n')
    structured_lines = []
    current_line = ""

    new_block_pattern = re.compile(
        r'^(Q\d+|Ans\s*\.|\([a-dA-D1-4]\)|[1-4]\.\s|Assertion\s*\(|Reason\s*\(|Read the following|Category)',
        re.IGNORECASE
    )

    for raw in raw_lines:
        line = clean_raw_line(raw)
        if not line or line in ["A", "-247", "247"]:
            continue

        if new_block_pattern.match(line) or is_standalone_series_or_code(line):
            if current_line:
                structured_lines.append(current_line.strip())
            current_line = line
        else:
            if current_line and (
                re.match(r'^(Ans\s*\.|\([a-dA-D1-4]\))', current_line)
                or is_standalone_series_or_code(current_line)
            ):
                structured_lines.append(current_line.strip())
                current_line = line
            else:
                current_line = f"{current_line} {line}" if current_line else line

    if current_line:
        structured_lines.append(current_line.strip())

    return structured_lines


def should_skip_translation(text):
    """Determines whether a line contains numbers, math symbols, or letter codes that shouldn't be translated."""
    cleaned = text.strip()
    if not cleaned:
        return True
    if re.fullmatch(r'[\d\s.,:?/+\-*=%@#$()&\-–—]+', cleaned):
        return True
    if not re.search(r'[a-z]', cleaned) and len(cleaned.split()) <= 10:
        return True
    return False


def split_long_text(text, max_words=150):
    """Breaks long paragraphs (such as reading comprehension passages) into manageable sentence blocks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current.split()) + len(s.split()) > max_words:
            if current.strip():
                chunks.append(current.strip())
            current = s + " "
        else:
            current += s + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def clean_translated_output(text):
    """Normalizes model output artifacts, turning placeholder symbols into neat underlines."""
    text = re.sub(r'(_\s*)?(@\s*)+', ' ______ ', text)
    return re.sub(r'\s+', ' ', text).strip()


def translate_batch(texts, src_lang, tgt_lang, model, tokenizer):
    """Feeds batches of text into the transformer model for efficient translation."""
    if not texts:
        return []

    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        formatted_batch = [f"{src_lang} {tgt_lang} {t}" for t in batch]

        inputs = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend([clean_translated_output(d) for d in decoded])

    return results


def translate_mcq_page(lines, src_lang, tgt_lang, model, tokenizer):
    """Extracts line prefixes (like Q1 or option labels), handles batch translation, and rebuilds the page."""
    prefix_regex = re.compile(r'^(Q\d+\.*|\([a-dA-D1-4]\)|[1-4]\.)\s*(.*)$')

    parsed_items = []
    texts_to_translate = []

    for line in lines:
        if re.match(r'^Ans\s*\.', line, re.IGNORECASE):
            parsed_items.append({"prefix": line, "chunks": []})
            continue

        match = prefix_regex.match(line)
        if match:
            prefix = match.group(1) + " "
            body = match.group(2).strip()
        else:
            prefix = ""
            body = line.strip()

        if should_skip_translation(body):
            parsed_items.append({"prefix": f"{prefix}{body}".strip(), "chunks": []})
        else:
            chunks = split_long_text(body)
            chunk_indices = []
            for ch in chunks:
                chunk_indices.append(len(texts_to_translate))
                texts_to_translate.append(ch)
            parsed_items.append({"prefix": prefix, "chunks": chunk_indices})

    translated_strings = translate_batch(texts_to_translate, src_lang, tgt_lang, model, tokenizer)

    final_lines = []
    for item in parsed_items:
        if not item["chunks"]:
            final_lines.append(item["prefix"])
        else:
            trans_body = " ".join(translated_strings[idx] for idx in item["chunks"])
            final_lines.append(f"{item['prefix']}{trans_body}")

    return "\n".join(final_lines)


def extract_text_from_pdf(pdf_path):
    """Reads all pages from the target PDF file while skipping promotional ads."""
    print("Extracting text from original PDF...")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text()
        if txt and (
            "Test Prime - built only for mock tests" in txt
            or "Turn PDFs into real exam experience" in txt
        ):
            print(f"Skipping promotional ad on page {i+1}...")
            continue
        if txt and txt.strip():
            pages.append(txt)
    return pages


def create_translated_pdf(translated_pages, output_path, font_path):
    """Builds and saves the final translated PDF document with proper margins and shaping."""
    print(f"\nCreating translated PDF at {output_path} using font '{font_path}'...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    try:
        pdf.set_text_shaping(True)
    except AttributeError:
        print("Note: Upgrade fpdf2 and install uharfbuzz for best Indic script rendering.")

    pdf.add_font("IndicFont", "", font_path)
    pdf.set_font("IndicFont", size=11)

    effective_width = pdf.w - 2 * pdf.l_margin

    for page_text in translated_pages:
        pdf.add_page()
        for line in page_text.split('\n'):
            clean_line = line.strip()
            if not clean_line:
                continue

            if re.match(r'^Q\d+', clean_line):
                pdf.ln(3)

            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(effective_width, 7, txt=clean_line)

    pdf.output(output_path)
    print(f"🎉 Success! Translated PDF saved as '{output_path}'")


# --- Accuracy Evaluation Formulas ---
def calculate_accuracy(orig_text, back_text):
    """
    Computes Word F1 vocabulary overlap, character sequence similarity, 
    and AI-powered semantic meaning similarity (if available).
    """
    # Normalize texts (lowercase, strip extra spaces)
    orig_norm = re.sub(r'\s+', ' ', orig_text.lower()).strip()
    back_norm = re.sub(r'\s+', ' ', back_text.lower()).strip()

    # 1. Sequence Matcher Ratio (Character-level order similarity)
    seq_accuracy = difflib.SequenceMatcher(None, orig_norm, back_norm).ratio() * 100

    # 2. Word-level F1 / Dice (Vocabulary preservation)
    orig_words = re.findall(r'\w+', orig_norm)
    back_words = re.findall(r'\w+', back_norm)

    if not orig_words and not back_words:
        word_f1 = 100.0
    elif not orig_words or not back_words:
        word_f1 = 0.0
    else:
        common = Counter(orig_words) & Counter(back_words)
        overlap_count = sum(common.values())
        word_f1 = (2.0 * overlap_count) / (len(orig_words) + len(back_words)) * 100

    # 3. Semantic Similarity (Meaning preservation via AI embeddings)
    semantic_acc = 0.0
    if HAS_SEMANTIC and orig_norm and back_norm:
        # Convert sentences to mathematical vectors and check cosine similarity
        emb1 = SEMANTIC_MODEL.encode(orig_norm, convert_to_tensor=True)
        emb2 = SEMANTIC_MODEL.encode(back_norm, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        semantic_acc = max(0.0, similarity * 100)

    # Average the available metrics for a combined page score
    if HAS_SEMANTIC:
        combined_pct = (seq_accuracy + word_f1 + semantic_acc) / 3.0
    else:
        combined_pct = (seq_accuracy + word_f1) / 2.0
        
    return word_f1, seq_accuracy, semantic_acc, combined_pct


def create_evaluation_file(original_pages, back_translated_pages, target_name):
    """Generates a detailed accuracy report comparing the original text with the back-translated text."""
    eval_file = f"./Evaluation/evaluation_{target_name.replace(' ', '_')}.txt"

    total_word_f1, total_seq_acc, total_sem_acc, total_combined = [], [], [], []
    page_metrics = []

    # Calculate metrics for each page
    for orig, back_trans in zip(original_pages, back_translated_pages):
        w_f1, s_acc, sem_acc, comb = calculate_accuracy(orig, back_trans)
        page_metrics.append((w_f1, s_acc, sem_acc, comb))
        total_word_f1.append(w_f1)
        total_seq_acc.append(s_acc)
        total_sem_acc.append(sem_acc)
        total_combined.append(comb)

    # Calculate document-wide averages
    avg_word_f1 = sum(total_word_f1) / len(total_word_f1) if total_word_f1 else 0.0
    avg_seq_acc = sum(total_seq_acc) / len(total_seq_acc) if total_seq_acc else 0.0
    avg_sem_acc = sum(total_sem_acc) / len(total_sem_acc) if total_sem_acc else 0.0
    overall_accuracy = sum(total_combined) / len(total_combined) if total_combined else 0.0

    # Optional sacrebleu metrics if installed
    bleu_str, chrf_str = "", ""
    if HAS_SACREBLEU and original_pages:
        bleu = sacrebleu.corpus_bleu(back_translated_pages, [original_pages])
        chrf = sacrebleu.corpus_chrf(back_translated_pages, [original_pages])
        bleu_str = f"NLP BLEU Score                  : {bleu.score:.2f}%\n"
        chrf_str = f"NLP chrF Score (Char n-gram F1) : {chrf.score:.2f}%\n"
        
        # Factor chrF into the overall accuracy score
        metric_count = 3 if HAS_SEMANTIC else 2
        overall_accuracy = ((overall_accuracy * metric_count) + chrf.score) / (metric_count + 1)

    # Write report to file
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("           COMPREHENSIVE TRANSLATION ACCURACY REPORT        \n")
        f.write("============================================================\n")
        f.write(f"Target Language Tested          : {target_name}\n")
        f.write(f"Word Overlap F1 Accuracy        : {avg_word_f1:.2f}%\n")
        f.write(f"Character Sequence Similarity   : {avg_seq_acc:.2f}%\n")
        if HAS_SEMANTIC:
            f.write(f"AI Semantic Meaning Similarity  : {avg_sem_acc:.2f}%\n")
        if HAS_SACREBLEU:
            f.write(chrf_str)
            f.write(bleu_str)
        f.write("------------------------------------------------------------\n")
        f.write(f"⭐ OVERALL TRANSLATION ACCURACY : {overall_accuracy:.2f}%\n")
        f.write("============================================================\n\n")

        for i, ((orig, back_trans), (w_f1, s_acc, sem_acc, comb)) in enumerate(
            zip(zip(original_pages, back_translated_pages), page_metrics)
        ):
            # Format the page header dynamically based on whether semantic checking is enabled
            sem_str = f" | Semantic: {sem_acc:.2f}%" if HAS_SEMANTIC else ""
            f.write(f"--- PAGE {i+1} (Page Accuracy: {comb:.2f}% | Word F1: {w_f1:.2f}% | Seq: {s_acc:.2f}%{sem_str}) ---\n")
            
            f.write(f"ORIGINAL ENGLISH (STRUCTURED):\n{orig}\n\n")
            f.write(f"BACK-TRANSLATED ENGLISH:\n{back_trans}\n\n")
            f.write("=" * 60 + "\n\n")

    # Print summary to terminal
    print("\n============================================================")
    print("           📊 COMPREHENSIVE ACCURACY SUMMARY                ")
    print("============================================================")
    print(f"   • Word Overlap F1 Accuracy      : {avg_word_f1:.2f}%")
    print(f"   • Character Sequence Similarity : {avg_seq_acc:.2f}%")
    if HAS_SEMANTIC:
        print(f"   • AI Semantic Meaning Similarity: {avg_sem_acc:.2f}%")
    if HAS_SACREBLEU:
        print(f"   • NLP chrF Score                : {chrf.score:.2f}%")
        print(f"   • NLP BLEU Score                : {bleu.score:.2f}%")
    print("------------------------------------------------------------")
    print(f"   ⭐ OVERALL TRANSLATION ACCURACY : {overall_accuracy:.2f}%")
    print("============================================================")
    print(f"📋 Full page-by-page accuracy report saved as '{eval_file}'")


def translate_document(input_pdf_path, output_pdf_path, target_code, target_name):
    """Main execution pipeline callable by the GUI."""
    if not os.path.exists(LANGUAGES_FILE):
        raise FileNotFoundError(f"{LANGUAGES_FILE} not found. Please ensure it exists.")

    font_path = get_font_path(target_code)
    print(f"🔤 Using Font: {font_path}")

    # 1. Extract and structure text blocks from the source PDF
    raw_pages = extract_text_from_pdf(input_pdf_path)
    structured_pages_lines = [structure_mcq_page(p) for p in raw_pages]
    structured_pages_text = ["\n".join(lines) for lines in structured_pages_lines]

    translated_pages = []
    back_translated_pages = []

    # 2. Perform Forward Translation (English to Target Language)
    print("\nLoading Forward Model (English -> Indic)...")
    tokenizer_fwd = AutoTokenizer.from_pretrained(MODEL_FORWARD, trust_remote_code=True)
    model_fwd = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FORWARD, trust_remote_code=True)

    print("\nTranslating to Target Language...")
    for i, page_lines in enumerate(structured_pages_lines):
        print(f"Forward translating page {i+1}/{len(structured_pages_lines)}...")
        translated_page = translate_mcq_page(page_lines, "eng_Latn", target_code, model_fwd, tokenizer_fwd)
        translated_pages.append(translated_page)

    del model_fwd
    del tokenizer_fwd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Perform Back-Translation (Target Language back to English) for Accuracy Verification
    print("\nLoading Reverse Model (Indic -> English) for Accuracy Check...")
    tokenizer_rev = AutoTokenizer.from_pretrained(MODEL_REVERSE, trust_remote_code=True)
    model_rev = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REVERSE, trust_remote_code=True)

    print("\nTranslating back to English for evaluation...")
    for i, trans_page in enumerate(translated_pages):
        print(f"Back-translating page {i+1}/{len(translated_pages)}...")
        trans_lines = trans_page.split('\n')
        back_translated = translate_mcq_page(trans_lines, target_code, "eng_Latn", model_rev, tokenizer_rev)
        back_translated_pages.append(back_translated)
        
    del model_rev
    del tokenizer_rev
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Output final translated document and mathematical accuracy evaluation report
    create_translated_pdf(translated_pages, output_pdf_path, font_path)
    create_evaluation_file(structured_pages_text, back_translated_pages, target_name)


# Allow app.py to still be run from the terminal for testing/debugging
if __name__ == "__main__":
    languages = load_languages(LANGUAGES_FILE)
    tgt_code, tgt_name = select_target_language(languages)
    out_pdf = f"./Output/translated_{tgt_name.replace(' ', '_')}.pdf"
    translate_document(INPUT_PDF, out_pdf, tgt_code, tgt_name)
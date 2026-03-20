import streamlit as st
import pdfplumber
import anthropic
import pandas as pd
import io
import os
import json
import re

import base64

import fitz  # PyMuPDF
from PIL import Image


COLUMN_LABELS = {
    "filename": "File",
    "project_name": "Project Name",
    "applicant_developer_name": "Applicant / Developer",
    "county": "County",
    "state": "State",
    "coordinates": "Coordinates",
    "capacity_mw": "Capacity (MW)",
    "technology_type": "Technology",
    "filing_or_permit_date": "Filing / Permit Date",
    "approval_status": "Approval Status",
    "key_conditions_or_modifications": "Key Conditions / Modifications",
    "cost_figures": "Cost Figures",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """Return API key from Streamlit secrets, env var, or session state."""
    if "ANTHROPIC_API_KEY" in st.secrets:
        return st.secrets["ANTHROPIC_API_KEY"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    return st.session_state.get("api_key")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


MAX_IMAGE_DIM = 1400   # max width or height in pixels before resizing
MAX_IMAGES = 15        # max images to extract per document


def resize_image(img_bytes: bytes) -> bytes:
    """Resize an image to fit within MAX_IMAGE_DIM, converting CMYK to RGB if needed."""
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode in ("CMYK", "P"):
        img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    return buf.getvalue()


def extract_images_from_pdf(file_bytes: bytes) -> list[bytes]:
    """Extract images from a PDF using PyMuPDF, resizing large ones."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    seen_xrefs = set()
    for page in doc:
        if len(images) >= MAX_IMAGES:
            break
        for img in page.get_images(full=True):
            if len(images) >= MAX_IMAGES:
                break
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            # Skip tiny images (logos, icons, line art)
            if len(img_bytes) < 10_000:
                continue
            try:
                images.append(resize_image(img_bytes))
            except Exception:
                continue  # skip unreadable images
    return images


def is_map_image(img_bytes: bytes, client: anthropic.Anthropic) -> bool:
    """Use Claude Haiku vision to check if an image looks like a map, aerial photo, or site plan."""
    img_b64 = base64.standard_b64encode(img_bytes).decode()
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": "Does this image look like a map, aerial photo, site plan, or project location diagram? Reply YES or NO only.",
                    },
                ],
            }],
        )
        return response.content[0].text.strip().upper().startswith("Y")
    except Exception:
        return False  # skip on any error


def extract_fields_with_claude(
    text: str, filename: str, client: anthropic.Anthropic
) -> dict:
    """Send extracted PDF text to Claude and return structured fields."""
    truncated = text[:60_000]
    truncation_note = (
        "\n\n[Note: document was truncated to fit context limits]"
        if len(text) > 60_000
        else ""
    )

    prompt = f"""You are analyzing a renewable energy project document.
Extract the fields listed below from the document text. If a field is not present, use null.

Document filename: {filename}{truncation_note}

--- DOCUMENT TEXT ---
{truncated}
--- END DOCUMENT ---

Return ONLY a valid JSON object with exactly these keys (no markdown, no explanation):
{{
  "project_name": "...",
  "applicant_developer_name": "...",
  "county": "...",
  "state": "...",
  "coordinates": "...",
  "capacity_mw": "...",
  "technology_type": "...",
  "filing_or_permit_date": "...",
  "approval_status": "...",
  "key_conditions_or_modifications": "...",
  "cost_figures": "...",
  "summary": "..."
}}

Field guidance:
- project_name: Name of the solar/energy project
- applicant_developer_name: Company or person who filed / is developing the project
- county: County where the project is located
- state: US state where the project is located
- coordinates: GPS / lat-long coordinates if mentioned
- capacity_mw: Nameplate or AC capacity in megawatts, include units (e.g. "200 MW AC")
- technology_type: solar PV, battery storage, solar+storage, wind, etc.
- filing_or_permit_date: Date the permit or application was filed
- approval_status: approved, conditionally approved, pending, denied, etc.
- key_conditions_or_modifications: brief summary of key conditions or requirements
- cost_figures: any cost amounts mentioned (interconnection costs, project cost, fees, etc.)
- summary: 2-3 sentence plain-language summary of the project for a non-technical audience
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude adds them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    result["filename"] = filename
    return result


def results_to_df(results: list[dict]) -> pd.DataFrame:
    """Convert list of result dicts into a display-ready DataFrame (excludes summary and images)."""
    table_keys = list(COLUMN_LABELS.keys())
    rows = [{k: r.get(k) for k in table_keys} for r in results]
    df = pd.DataFrame(rows)
    cols = ["filename"] + [c for c in COLUMN_LABELS if c != "filename" and c in df.columns]
    df = df[cols]
    df = df.rename(columns=COLUMN_LABELS)
    df = df.fillna("—")
    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚡ Energy Document Intelligence")
st.caption(
    "Upload solar permits, interconnection filings, or project modification documents "
    "to extract key structured fields using Claude AI."
)

# API key input (shown only when not set via secrets/env)
api_key = get_api_key()
if not api_key:
    with st.sidebar:
        st.header("Configuration")
        entered_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get your key at console.anthropic.com",
        )
        if entered_key:
            st.session_state["api_key"] = entered_key
            api_key = entered_key
        if not api_key:
            st.warning("Enter your API key to get started.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select multiple PDF files at once.",
)

if uploaded_files and not api_key:
    st.error("Please enter your Anthropic API key in the sidebar before processing.")
    st.stop()

if uploaded_files and api_key:
    if st.button("Extract Fields", type="primary", use_container_width=True):
        client = anthropic.Anthropic(api_key=api_key)
        results = []
        errors = []

        progress_bar = st.progress(0, text="Starting extraction…")
        status_area = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_area.info(f"Processing **{filename}** ({i + 1}/{len(uploaded_files)})…")

            try:
                file_bytes = uploaded_file.read()

                # Step 1: extract text
                pdf_text = extract_text_from_pdf(file_bytes)
                if not pdf_text.strip():
                    errors.append((filename, "No extractable text found — the PDF may be scanned/image-only."))
                    continue

                # Step 2: extract images, then filter to maps only
                all_images = extract_images_from_pdf(file_bytes)
                status_area.info(f"Classifying images in **{filename}**…")
                images = [img for img in all_images if is_map_image(img, client)]

                # Step 3: call Claude for structured fields
                fields = extract_fields_with_claude(pdf_text, filename, client)
                fields["_images"] = images
                results.append(fields)

            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check your key and try again.")
                st.stop()
            except anthropic.RateLimitError:
                errors.append((filename, "Rate limit hit — wait a moment and try again."))
            except Exception as e:
                errors.append((filename, str(e)))

            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {i + 1}/{len(uploaded_files)} files")

        status_area.empty()
        progress_bar.empty()

        # Show any errors
        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) had errors", expanded=True):
                for fname, msg in errors:
                    st.error(f"**{fname}**: {msg}")

        # Show results
        if results:
            st.success(f"Extracted fields from {len(results)} document(s).")
            df = results_to_df(results)

            # Tab names: "All Projects" + one per project
            def short_name(r):
                name = r.get("project_name") or r["filename"]
                return name[:30] + "…" if len(name) > 30 else name

            tab_names = ["All Projects"] + [short_name(r) for r in results]
            tabs = st.tabs(tab_names)

            # ── All Projects tab ──────────────────────────────────────────────
            with tabs[0]:
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Download CSV",
                    data=df_to_csv_bytes(df),
                    file_name="energy_document_extractions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # ── Per-project tabs ──────────────────────────────────────────────
            for i, result in enumerate(results):
                with tabs[i + 1]:
                    # Summary
                    summary = result.get("summary")
                    if summary:
                        st.info(summary)

                    # Key fields in two columns
                    col1, col2 = st.columns(2)
                    field_pairs = [
                        ("Project Name", result.get("project_name")),
                        ("Applicant / Developer", result.get("applicant_developer_name")),
                        ("County", result.get("county")),
                        ("State", result.get("state")),
                        ("Capacity (MW)", result.get("capacity_mw")),
                        ("Technology", result.get("technology_type")),
                        ("Filing / Permit Date", result.get("filing_or_permit_date")),
                        ("Approval Status", result.get("approval_status")),
                        ("Coordinates", result.get("coordinates")),
                        ("Cost Figures", result.get("cost_figures")),
                    ]
                    for j, (label, value) in enumerate(field_pairs):
                        target_col = col1 if j % 2 == 0 else col2
                        target_col.metric(label, value or "—")

                    # Key conditions (can be long — full width)
                    conditions = result.get("key_conditions_or_modifications")
                    if conditions and conditions != "null":
                        st.markdown("**Key Conditions / Modifications**")
                        st.write(conditions)

                    # Images
                    images = result.get("_images", [])
                    if images:
                        st.markdown(f"**Maps & Images** ({len(images)} extracted)")
                        for img_bytes in images:
                            st.image(img_bytes, use_container_width=True)
                    else:
                        st.caption("No images found in this document.")

        elif not errors:
            st.warning("No results were extracted.")

elif not uploaded_files:
    st.info("Upload one or more PDF files above to get started.")

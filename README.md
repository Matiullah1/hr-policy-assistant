# HR Policy Assistant using RAG

A Streamlit app that lets users upload an HR Policy PDF and ask questions about it. The app extracts text with PyMuPDF, creates embeddings with Sentence Transformers, searches relevant passages using FAISS, and produces a grounded answer with Groq's `openai/gpt-oss-20b` model.

## Features

- Upload one HR policy PDF
- Page-aware text chunking
- Semantic retrieval with `all-MiniLM-L6-v2` and FAISS
- Answers grounded only in retrieved policy text
- Inline page citations
- Expandable retrieved passages for transparency
- Clear fallback when the answer is absent
- Groq API key kept in Streamlit secrets

## Project files

```text
hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Deploy using only GitHub and Streamlit Cloud

### 1. Get a Groq API key

1. Open [GroqCloud Console](https://console.groq.com/keys).
2. Sign in and create an API key.
3. Copy it and keep it private. Never put it in `app.py` or GitHub.

### 2. Create the GitHub repository

1. Sign in at [GitHub](https://github.com/).
2. Click **New repository**.
3. Name it `hr-policy-assistant`.
4. Select **Public** (the easiest option for Streamlit Community Cloud).
5. Click **Create repository**.

### 3. Upload these files in the browser

1. On the empty repository page, click **uploading an existing file**. If the repository is not empty, click **Add file → Upload files**.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Click **Commit changes**.
4. Confirm that all four files are at the repository's top level, not inside another folder.

### 4. Deploy on Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Click **Create app**.
3. Choose the repository `hr-policy-assistant` and the branch `main`.
4. Set **Main file path** to `app.py`.
5. Open **Advanced settings** and select Python 3.11 if a Python-version option is shown.
6. In **Secrets**, paste the following, replacing the placeholder with your real key:

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```

7. Click **Deploy**. The first build can take several minutes because the embedding model and PyTorch dependencies must be installed.

### 5. Test the deployed app

1. Upload a text-based HR Policy PDF.
2. Ask a question whose answer is clearly stated in the PDF.
3. Check the cited page and open **Retrieved policy passages**.
4. Ask a question not answered by the policy and confirm that the assistant says the information was not found.

## Updating the app without a terminal

1. Open a file in GitHub.
2. Click the pencil icon (**Edit this file**).
3. Make the change and click **Commit changes**.
4. Streamlit Cloud normally detects the commit and redeploys automatically.

To replace a complete file, use **Add file → Upload files** and upload a file with the same name.

## Troubleshooting

- **Missing API key:** Open the deployed app in Streamlit Cloud, choose **Manage app → Settings → Secrets**, add the TOML secret, save, and reboot the app.
- **No text found:** The PDF is probably scanned. Apply OCR using a PDF/OCR service, then upload the searchable PDF.
- **Build error:** Check that `requirements.txt` has that exact name and is in the same top-level folder as `app.py`.
- **Slow first start:** Loading Sentence Transformers and its model can take time on the first run. Later reruns use Streamlit's resource cache.
- **Private repository unavailable:** Reauthorize Streamlit's GitHub access or allow the Streamlit GitHub app to access that repository.

## Privacy and production notes

The app processes the uploaded PDF in memory for the current Streamlit session and does not intentionally save it to disk. Relevant text passages are sent to Groq to generate answers. For confidential company policies, confirm that this processing matches your organization's privacy, retention, and access-control requirements. A public deployment does not provide employee authentication or document-level authorization by itself.

## Limitations

- Scanned PDFs need OCR before upload.
- The in-memory FAISS index disappears when the Streamlit session ends.
- Retrieval quality depends on the document layout, extracted text, and wording of the question.
- This is a policy-information assistant, not legal advice or an access-control system.

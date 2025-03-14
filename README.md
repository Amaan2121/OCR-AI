# OCR AI - Text Extraction Tool

A powerful OCR (Optical Character Recognition) tool that uses multiple AI models to extract text from images and PDFs.

## Features

- Support for multiple AI models:
  - ChatGPT 4o (OpenAI)
  - Gemini 2.0 Flash (Google)
  - Mistral AI
  - Qwen 2.5 VL Max (Alibaba)
- File format support: PDF, PNG, JPG, JPEG, GIF
- Performance tracking and analytics
- User-friendly web interface
- Real-time processing status updates

## Prerequisites

- Python 3.8 or higher
- Poppler (for PDF processing)
  - Windows: Download and install from [poppler releases](http://blog.alivate.com.au/poppler-windows/)
  - Linux: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`

## Local Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd OCR-AI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
QWEN_API_KEY=your_qwen_api_key
```

## Running Locally

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## Cloud Deployment (Render.com)

1. Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Deploy on Render.com:
   - Go to [Render.com](https://render.com) and sign up/login with your GitHub account
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Configure the deployment:
     - Name: `ocr-ai`
     - Environment: `Python`
     - Build Command: `./build.sh`
     - Start Command: `gunicorn app:app`
     - Plan: Free

3. Set Environment Variables:
   In the Render dashboard, add these environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   QWEN_API_KEY=your_qwen_api_key
   ```

4. Deploy:
   - Click "Create Web Service"
   - Wait for the deployment to complete
   - Your app will be available at: `https://ocr-ai.onrender.com` (or similar URL)

## Usage

1. Upload a PDF or image file (max 16MB)
2. Select an AI model for processing
3. Click "Extract Text" to start processing
4. View the extracted text and performance metrics
5. Access the performance dashboard for analytics

## Project Structure

```
OCR-AI/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # API keys (create this)
├── static/            # Static files (charts, results)
├── templates/         # HTML templates
└── uploads/           # Temporary file uploads
```

## Notes

- The application runs in debug mode by default
- Uploaded files are temporarily stored in the `uploads/` directory
- Performance metrics are stored in `performance_log.json`
- Charts are generated in `static/charts/`

## Troubleshooting

1. If you get a "Poppler not found" error:
   - Ensure Poppler is installed and in your system PATH
   - For Windows, add the Poppler bin directory to your PATH

2. If API calls fail:
   - Check your API keys in the `.env` file
   - Ensure you have sufficient API credits
   - Check your internet connection

3. If the application crashes:
   - Check the console output for error messages
   - Ensure all dependencies are installed correctly
   - Try running with a smaller file size 
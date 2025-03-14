import os
import base64
import imghdr
import time
import json
import datetime
import re
import traceback
from io import BytesIO
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, send_file, session
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image
import openai
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=10)

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

# API keys - print for debugging
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
QWEN_API_KEY = os.getenv('QWEN_API_KEY')

print(f"API Keys loaded: OpenAI: {'✓' if OPENAI_API_KEY else '✗'}, "
      f"Gemini: {'✓' if GEMINI_API_KEY else '✗'}, "
      f"Mistral: {'✓' if MISTRAL_API_KEY else '✗'}, "
      f"Qwen: {'✓' if QWEN_API_KEY else '✗'}")

# Configure API clients
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if MISTRAL_API_KEY:
    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

if QWEN_API_KEY:
    dashscope.api_key = QWEN_API_KEY

# Performance tracking
PERFORMANCE_LOG_FILE = 'performance_log.json'

def load_performance_data():
    if os.path.exists(PERFORMANCE_LOG_FILE):
        with open(PERFORMANCE_LOG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_performance_data(data):
    with open(PERFORMANCE_LOG_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def count_lines(text):
    """Count the number of lines in the text."""
    return len(text.split('\n'))

def count_paragraphs(text):
    """Count the number of paragraphs in the text."""
    # A paragraph is defined as text separated by one or more blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    return len([p for p in paragraphs if p.strip()])

def calculate_text_density(character_count, word_count):
    """Calculate the average word length (text density)."""
    if word_count == 0:
        return 0
    return character_count / word_count

def log_performance(model, file_type, file_size, processing_time, character_count, word_count, filename, ocr_text=None):
    performance_data = load_performance_data()
    
    # Calculate additional metrics
    if ocr_text:
        line_count = count_lines(ocr_text)
        paragraph_count = count_paragraphs(ocr_text)
    else:
        line_count = 0
        paragraph_count = 0
        
    text_density = calculate_text_density(character_count, word_count)
    chars_per_second = character_count / processing_time if processing_time > 0 else 0
    words_per_second = word_count / processing_time if processing_time > 0 else 0
    
    # Add new entry
    performance_data.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'model': model,
        'file_type': file_type,
        'file_size_kb': file_size / 1024,  # Convert to KB
        'processing_time_sec': processing_time,
        'character_count': character_count,
        'word_count': word_count,
        'line_count': line_count,
        'paragraph_count': paragraph_count,
        'text_density': text_density,
        'chars_per_second': chars_per_second,
        'words_per_second': words_per_second,
        'filename': filename
    })
    
    save_performance_data(performance_data)

def generate_performance_charts():
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        return None
    
    # Load performance data
    df = pd.read_json(PERFORMANCE_LOG_FILE)
    
    if df.empty:
        return None
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add missing columns with default values if they don't exist
    if 'chars_per_second' not in df.columns:
        df['chars_per_second'] = df.apply(lambda row: row['character_count'] / row['processing_time_sec'] 
                                         if row['processing_time_sec'] > 0 else 0, axis=1)
    
    if 'words_per_second' not in df.columns:
        df['words_per_second'] = df.apply(lambda row: row['word_count'] / row['processing_time_sec'] 
                                         if row['processing_time_sec'] > 0 else 0, axis=1)
    
    if 'text_density' not in df.columns:
        df['text_density'] = df.apply(lambda row: row['character_count'] / row['word_count'] 
                                     if row['word_count'] > 0 else 0, axis=1)
    
    if 'line_count' not in df.columns:
        df['line_count'] = 0
    
    if 'paragraph_count' not in df.columns:
        df['paragraph_count'] = 0
    
    # Create charts directory
    charts_dir = os.path.join('static', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Generate charts
    chart_files = []
    
    try:
        # 1. Average processing time by model
        plt.figure(figsize=(10, 6))
        avg_time = df.groupby('model')['processing_time_sec'].mean().sort_values(ascending=False)
        avg_time.plot(kind='bar', color='skyblue')
        plt.title('Average Processing Time by Model')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.tight_layout()
        time_chart = os.path.join(charts_dir, 'avg_time_by_model.png')
        plt.savefig(time_chart)
        plt.close()
        chart_files.append(time_chart)
        
        # 2. Processing speed by model (chars per second)
        plt.figure(figsize=(10, 6))
        avg_speed = df.groupby('model')['chars_per_second'].mean().sort_values(ascending=False)
        avg_speed.plot(kind='bar', color='lightgreen')
        plt.title('Average Processing Speed by Model')
        plt.ylabel('Characters per Second')
        plt.xlabel('Model')
        plt.tight_layout()
        speed_chart = os.path.join(charts_dir, 'avg_speed_by_model.png')
        plt.savefig(speed_chart)
        plt.close()
        chart_files.append(speed_chart)
        
        # 3. Processing time by file type
        if len(df['file_type'].unique()) > 1:  # Only if we have multiple file types
            plt.figure(figsize=(10, 6))
            avg_time_by_type = df.groupby(['model', 'file_type'])['processing_time_sec'].mean().unstack()
            if not avg_time_by_type.empty:
                avg_time_by_type.plot(kind='bar')
                plt.title('Average Processing Time by Model and File Type')
                plt.ylabel('Time (seconds)')
                plt.xlabel('Model')
                plt.tight_layout()
                type_chart = os.path.join(charts_dir, 'avg_time_by_type.png')
                plt.savefig(type_chart)
                plt.close()
                chart_files.append(type_chart)
        
        # 4. Text density by model
        plt.figure(figsize=(10, 6))
        avg_density = df.groupby('model')['text_density'].mean().sort_values(ascending=False)
        avg_density.plot(kind='bar', color='salmon')
        plt.title('Average Text Density by Model (Chars per Word)')
        plt.ylabel('Characters per Word')
        plt.xlabel('Model')
        plt.tight_layout()
        density_chart = os.path.join(charts_dir, 'text_density_by_model.png')
        plt.savefig(density_chart)
        plt.close()
        chart_files.append(density_chart)
        
        # 5. File size vs processing time scatter plot
        plt.figure(figsize=(10, 6))
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.scatter(model_data['file_size_kb'], model_data['processing_time_sec'], label=model, alpha=0.7)
        plt.title('File Size vs Processing Time')
        plt.xlabel('File Size (KB)')
        plt.ylabel('Processing Time (seconds)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        scatter_chart = os.path.join(charts_dir, 'file_size_vs_time.png')
        plt.savefig(scatter_chart)
        plt.close()
        chart_files.append(scatter_chart)
        
        # 6. Performance over time
        if len(df) > 1:  # Only if we have multiple entries
            plt.figure(figsize=(12, 6))
            for model in df['model'].unique():
                model_data = df[df['model'] == model].sort_values('timestamp')
                if len(model_data) > 0:
                    plt.plot(model_data['timestamp'], model_data['chars_per_second'], label=model, marker='o')
            plt.title('OCR Performance Over Time')
            plt.xlabel('Date')
            plt.ylabel('Characters per Second')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            time_series_chart = os.path.join(charts_dir, 'performance_over_time.png')
            plt.savefig(time_series_chart)
            plt.close()
            chart_files.append(time_series_chart)
    
    except Exception as e:
        print(f"Error generating charts: {str(e)}")
    
    return [f.replace('static/', '') for f in chart_files]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def is_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def is_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_ocr_text_chatgpt(file_path):
    """Process file with ChatGPT 4o"""
    start_time = time.time()
    
    try:
        if is_pdf(file_path):
            # Convert PDF to images
            images = convert_from_path(file_path)
            all_text = []
            
            for i, image in enumerate(images):
                # Save each page as temporary image
                temp_img_path = f"{file_path}_page_{i}.jpg"
                image.save(temp_img_path, "JPEG")
                
                # Process image with ChatGPT
                base64_image = encode_image_to_base64(temp_img_path)
                
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an OCR assistant. Extract all text from the image accurately."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract all the text from this image."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=1500,
                        timeout=60  # 60 second timeout
                    )
                    
                    all_text.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"Error processing page {i+1} with ChatGPT: {str(e)}")
                    all_text.append(f"Error processing page {i+1}: {str(e)}")
                
                # Clean up temporary image
                os.remove(temp_img_path)
                
            result_text = "\n\n--- Page Break ---\n\n".join(all_text)
        
        elif is_image(file_path):
            # Process single image
            base64_image = encode_image_to_base64(file_path)
            
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an OCR assistant. Extract all text from the image accurately."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract all the text from this image."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1500,
                    timeout=60  # 60 second timeout
                )
                
                result_text = response.choices[0].message.content
            except Exception as e:
                print(f"Error processing image with ChatGPT: {str(e)}")
                traceback.print_exc()
                result_text = f"Error processing image: {str(e)}"
        
        else:
            return "Unsupported file format", 0
        
        processing_time = time.time() - start_time
        
        return result_text, processing_time
    
    except Exception as e:
        print(f"Error in ChatGPT processing: {str(e)}")
        traceback.print_exc()
        return f"Error processing with ChatGPT: {str(e)}", time.time() - start_time

def get_ocr_text_gemini(file_path):
    """Process file with Gemini 2.0 Flash"""
    start_time = time.time()
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if is_pdf(file_path):
            # For PDFs, we'll process each page as an image
            images = convert_from_path(file_path)
            all_text = []
            
            for i, image in enumerate(images):
                # Convert PIL Image to bytes
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Process with Gemini
                try:
                    response = model.generate_content(
                        [
                            "Extract all the text from this image. Return only the extracted text without any additional commentary.",
                            {"mime_type": "image/jpeg", "data": img_byte_arr}
                        ],
                        # Set a timeout for the request
                        generation_config=genai.types.GenerationConfig(
                            temperature=0,
                            top_p=1,
                            top_k=1,
                            max_output_tokens=1500,
                        )
                    )
                    
                    all_text.append(response.text)
                except Exception as e:
                    print(f"Error processing page {i+1} with Gemini: {str(e)}")
                    all_text.append(f"Error processing page {i+1}: {str(e)}")
            
            result_text = "\n\n--- Page Break ---\n\n".join(all_text)
        
        elif is_image(file_path):
            # Process single image
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            # Determine MIME type
            mime_type = f"image/{file_path.split('.')[-1].lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"
            
            try:
                response = model.generate_content(
                    [
                        "Extract all the text from this image. Return only the extracted text without any additional commentary.",
                        {"mime_type": mime_type, "data": image_data}
                    ],
                    # Set a timeout for the request
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        top_p=1,
                        top_k=1,
                        max_output_tokens=1500,
                    )
                )
                
                result_text = response.text
            except Exception as e:
                print(f"Error processing image with Gemini: {str(e)}")
                traceback.print_exc()
                result_text = f"Error processing image: {str(e)}"
        
        else:
            return "Unsupported file format", 0
        
        processing_time = time.time() - start_time
        
        return result_text, processing_time
    
    except Exception as e:
        print(f"Error in Gemini processing: {str(e)}")
        traceback.print_exc()
        return f"Error processing with Gemini: {str(e)}", time.time() - start_time

def get_ocr_text_mistral(file_path):
    """Process file with Mistral AI"""
    start_time = time.time()
    
    try:
        if is_pdf(file_path):
            # For PDFs, we need to convert to images first
            images = convert_from_path(file_path)
            all_text = []
            
            for i, image in enumerate(images):
                # Save each page as temporary image
                temp_img_path = f"{file_path}_page_{i}.jpg"
                image.save(temp_img_path, "JPEG")
                
                # Convert image to base64 for description
                with open(temp_img_path, 'rb') as f:
                    image_data = f.read()
                
                # Create a text-only prompt that describes what we're trying to do
                # Since Mistral doesn't directly support image input in the API
                messages = [
                    ChatMessage(role="system", content="You are an OCR assistant. Your task is to extract text from documents."),
                    ChatMessage(role="user", content=f"I'm trying to extract text from page {i+1} of a PDF document. Please provide instructions on how I can best extract text from PDF documents.")
                ]
                
                # Call Mistral API
                response = mistral_client.chat(
                    model="mistral-large-latest",
                    messages=messages
                )
                
                # For demonstration, we'll include a placeholder message
                all_text.append(f"[Page {i+1} text would appear here if Mistral supported direct image processing]")
                
                # Clean up temporary image
                os.remove(temp_img_path)
            
            # Since Mistral doesn't directly support OCR, we'll return a helpful message
            result_text = "\n\n".join(all_text)
            result_text += "\n\nNote: The Mistral model doesn't currently support direct OCR processing. Please try another model like ChatGPT 4o, Gemini, or Qwen for OCR capabilities."
            
        elif is_image(file_path):
            # Create a text-only prompt that describes what we're trying to do
            messages = [
                ChatMessage(role="system", content="You are an OCR assistant. Your task is to extract text from images."),
                ChatMessage(role="user", content="I'm trying to extract text from an image. Please provide instructions on how I can best extract text from images.")
            ]
            
            # Call Mistral API
            response = mistral_client.chat(
                model="mistral-large-latest",
                messages=messages
            )
            
            # Since Mistral doesn't directly support OCR, we'll return a helpful message
            result_text = "The Mistral model doesn't currently support direct OCR processing. Please try another model like ChatGPT 4o, Gemini, or Qwen for OCR capabilities."
            
        else:
            return "Unsupported file format", 0
        
        processing_time = time.time() - start_time
        
        return result_text, processing_time
    
    except Exception as e:
        print(f"Error in Mistral processing: {str(e)}")
        traceback.print_exc()
        return f"Error processing with Mistral: {str(e)}", time.time() - start_time

def get_ocr_text_qwen(file_path):
    """Process file with Qwen 2.5 VL Max"""
    start_time = time.time()
    
    try:
        if is_pdf(file_path):
            # Convert PDF to images
            images = convert_from_path(file_path)
            all_text = []
            
            for i, image in enumerate(images):
                # Save each page as temporary image
                temp_img_path = f"{file_path}_page_{i}.jpg"
                image.save(temp_img_path, "JPEG")
                
                # Process image with Qwen
                with open(temp_img_path, 'rb') as f:
                    image_data = f.read()
                
                # Convert image to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Create the message content properly
                response = dashscope.MultiModalConversation.call(
                    model='qwen2.5-vl-max',
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are an OCR assistant. Extract all text from the image accurately.'
                        },
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'Extract all the text from this image. Return only the extracted text without any additional commentary.'
                                },
                                {
                                    'type': 'image',
                                    'image_url': {
                                        'url': f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                )
                
                if response.status_code == 200:
                    # Handle the response content properly
                    try:
                        # Check if the response has the expected structure
                        if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                            message_content = response.output.choices[0].message.content
                            # Check if content is a list or a string
                            if isinstance(message_content, list) and len(message_content) > 0:
                                all_text.append(message_content[0].text)
                            elif isinstance(message_content, str):
                                all_text.append(message_content)
                            else:
                                all_text.append("Error: Unexpected response format")
                        else:
                            all_text.append("Error: No content in response")
                    except Exception as e:
                        all_text.append(f"Error processing page {i+1} content: {str(e)}")
                else:
                    all_text.append(f"Error processing page {i+1}: {response.code}")
                
                # Clean up temporary image
                os.remove(temp_img_path)
                
            result_text = "\n\n--- Page Break ---\n\n".join(all_text)
        
        elif is_image(file_path):
            # Process single image
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create the message content properly
            response = dashscope.MultiModalConversation.call(
                model='qwen2.5-vl-max',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an OCR assistant. Extract all text from the image accurately.'
                    },
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Extract all the text from this image. Return only the extracted text without any additional commentary.'
                            },
                            {
                                'type': 'image',
                                'image_url': {
                                    'url': f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            if response.status_code == 200:
                # Handle the response content properly
                try:
                    # Check if the response has the expected structure
                    if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                        message_content = response.output.choices[0].message.content
                        # Check if content is a list or a string
                        if isinstance(message_content, list) and len(message_content) > 0:
                            result_text = message_content[0].text
                        elif isinstance(message_content, str):
                            result_text = message_content
                        else:
                            result_text = "Error: Unexpected response format"
                    else:
                        result_text = "Error: No content in response"
                except Exception as e:
                    result_text = f"Error processing image content: {str(e)}"
            else:
                result_text = f"Error processing image: {response.code}"
        
        else:
            return "Unsupported file format", 0
        
        processing_time = time.time() - start_time
        
        return result_text, processing_time
    
    except Exception as e:
        print(f"Error in Qwen processing: {str(e)}")
        traceback.print_exc()
        return f"Error processing with Qwen: {str(e)}", time.time() - start_time

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        model = request.form.get('model')
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Determine file type
            file_type = 'pdf' if is_pdf(file_path) else 'image'
            
            try:
                # Warn user about Mistral's limitations for OCR
                if model == 'mistral':
                    flash('Note: Mistral has limited OCR capabilities. For best results, try ChatGPT 4o, Gemini, or Qwen.')
                
                # Process file based on selected model
                if model == 'chatgpt':
                    if not OPENAI_API_KEY:
                        flash('OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.')
                        return redirect(request.url)
                    ocr_text, processing_time = get_ocr_text_chatgpt(file_path)
                elif model == 'gemini':
                    if not GEMINI_API_KEY:
                        flash('Gemini API key not set. Please set the GEMINI_API_KEY environment variable.')
                        return redirect(request.url)
                    ocr_text, processing_time = get_ocr_text_gemini(file_path)
                elif model == 'mistral':
                    if not MISTRAL_API_KEY:
                        flash('Mistral API key not set. Please set the MISTRAL_API_KEY environment variable.')
                        return redirect(request.url)
                    ocr_text, processing_time = get_ocr_text_mistral(file_path)
                elif model == 'qwen':
                    if not QWEN_API_KEY:
                        flash('Qwen API key not set. Please set the QWEN_API_KEY environment variable.')
                        return redirect(request.url)
                    ocr_text, processing_time = get_ocr_text_qwen(file_path)
                else:
                    flash('Invalid model selection')
                    return redirect(request.url)
                
                # Check if there was an error in the OCR process
                if ocr_text and ocr_text.startswith("Error processing"):
                    flash(ocr_text)
                    return redirect(request.url)
                
                # Check if the response indicates the model can't process the file
                if ocr_text and ("unable to process" in ocr_text.lower() or "cannot process" in ocr_text.lower()):
                    flash(f"The {model} model was unable to process this file. Please try another model.")
                    return redirect(request.url)
                
                # Calculate metrics
                character_count = len(ocr_text) if ocr_text else 0
                word_count = len(ocr_text.split()) if ocr_text else 0
                
                # Only calculate these metrics if we have actual text content
                if character_count > 0:
                    line_count = count_lines(ocr_text)
                    paragraph_count = count_paragraphs(ocr_text)
                    text_density = calculate_text_density(character_count, word_count)
                    chars_per_second = character_count/processing_time if processing_time > 0 else 0
                    words_per_second = word_count/processing_time if processing_time > 0 else 0
                else:
                    line_count = 0
                    paragraph_count = 0
                    text_density = 0
                    chars_per_second = 0
                    words_per_second = 0
                
                # Log performance
                log_performance(
                    model=model,
                    file_type=file_type,
                    file_size=file_size,
                    processing_time=processing_time,
                    character_count=character_count,
                    word_count=word_count,
                    filename=filename,
                    ocr_text=ocr_text
                )
                
                # Generate performance metrics
                charts = generate_performance_charts()
                
                return render_template(
                    'result.html',
                    ocr_text=ocr_text,
                    filename=filename,
                    model=model,
                    processing_time=processing_time,
                    character_count=character_count,
                    word_count=word_count,
                    line_count=line_count,
                    paragraph_count=paragraph_count,
                    text_density=text_density,
                    chars_per_second=chars_per_second,
                    words_per_second=words_per_second,
                    charts=charts
                )
            
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                print(traceback.format_exc())
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
            finally:
                # Clean up - optionally remove the file after processing
                # os.remove(file_path)
                pass
        else:
            flash('Invalid file type. Allowed file types: PDF, PNG, JPG, JPEG, GIF')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/performance', methods=['GET'])
def performance_dashboard():
    # Generate performance charts
    charts = generate_performance_charts()
    
    # Load performance data for table
    performance_data = load_performance_data()
    
    return render_template('performance.html', charts=charts, performance_data=performance_data)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Use 0.0.0.0 to make the server publicly available
    app.run(host='0.0.0.0', port=port, debug=False) 
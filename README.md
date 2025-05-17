# Jianpu Classifier API 🎵

An API microservice that classifies staff notation or numbered notation (jianpu) music sheet images

- Staff notation
- Jianpu (numbered notation)

## 🔧 Features

- Uses OpenCV + Tesseract OCR
- Accepts image uploads via `/classify`
- Returns detected type and confidence scores
- Deployable with Docker

## 🚀 Quick Start

### Run with Docker

```bash
docker build -t music-sheet-api .
docker run -p 8000:8000 music-sheet-api
```

## 🧪 Test Locally

### 1. Swagger UI (Browser)

After running the service with Docker, visit:

```bash
Visit: http://localhost:8000/docs
```

### 2. CURL (Command Line)

```bash
curl -X POST http://localhost:8000/classify \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/your_sheet.png"
```

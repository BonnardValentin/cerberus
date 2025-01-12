# Cerberus

Cerberus is a modular, containerized AI solution offering APIs for sentiment analysis, text summarization, and text generation. Built on top of fine-tuned and quantized Hugging Face models, Cerberus is designed for scalability, efficiency, and easy integration into workflows.

---

## Features

- **Sentiment Analysis**: Quickly classify text sentiment as positive, negative, or neutral.
- **Summarization**: Generate concise and meaningful summaries for long text.
- **Text Generation**: Produce creative or structured text based on input prompts.
- **Scalable Architecture**: Deployed using containerized services for easy scaling.
- **Optimized Models**: Uses fine-tuned and quantized Hugging Face models for speed and accuracy.
- **OpenAPI Documentation**: Built-in API documentation for developers.
- **Extensible**: Easily add new AI models and services.

---

## Technologies Used

- **Programming Language**: Python 3.10+
- **Frameworks**: FastAPI for API development
- **AI Models**: Hugging Face (Transformers, Optimum, Accelerate)
- **Machine Learning Backend**: PyTorch with MPS support for macOS (Apple Silicon)
- **Data Handling**: Datasets library by Hugging Face
- **Deployment**:
  - Docker for containerization
  - Docker Compose for service orchestration

---

## Table of Contents

- [Cerberus](#cerberus)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Endpoints](#endpoints)
    - [Example Requests](#example-requests)
  - [API Documentation](#api-documentation)
  - [Development](#development)
    - [Run Locally Without Docker](#run-locally-without-docker)
    - [Test the Models](#test-the-models)
  - [Contributing](#contributing)
  - [License](#license)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  
- Docker and Docker Compose installed: [Docker Installation Guide](https://docs.docker.com/get-docker/)

### Installation

1. Clone the Repository:

   ```bash
   git clone https://github.com/BonnardValentin/cerberus
   cd cerberus
   ```

2. Install Dependencies with Poetry:

   ```bash
   poetry install
   ```

3. Set Up Environment Variables: Create a `.env` file in the project root with the following:

   ```plaintext
   MODEL_PATH="./models/quantized"
   API_KEY="your-secret-key"
   ```

4. Start the Application: Using Docker Compose:

   ```bash
   docker-compose up --build
   ```

5. Test the Application: Access the API at [http://localhost:8000](http://localhost:8000).

---

## Usage

### Endpoints

- **Sentiment Analysis**: `/analyze-sentiment/`
- **Summarization**: `/summarize/`
- **Text Generation**: `/generate-text/`

### Example Requests

**Sentiment Analysis**

```bash
curl -X POST http://localhost:8000/analyze-sentiment/ \
-H "Content-Type: application/json" \
-d '{"text": "I love Cerberus!"}'
```

**Summarization**

```bash
curl -X POST http://localhost:8000/summarize/ \
-H "Content-Type: application/json" \
-d '{"text": "Docker simplifies containerization by providing a platform to develop, ship, and run applications in isolated environments."}'
```

**Text Generation**

```bash
curl -X POST http://localhost:8000/generate-text/ \
-H "Content-Type: application/json" \
-d '{"text": "Once upon a time, in a faraway land,"}'
```

---

## API Documentation

Cerberus provides an interactive OpenAPI UI:

- Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.
- Visit [http://localhost:8000/redoc](http://localhost:8000/redoc) for ReDoc.

---

## Development

### Run Locally Without Docker

1. Activate the Poetry environment:

   ```bash
   poetry env activate
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

### Test the Models

Run a Python script to verify that models are loaded correctly:

```python
from transformers import pipeline

model = pipeline("sentiment-analysis", model="./models/quantized")
print(model("I love open-source!"))
```

---

## Contributing

We welcome contributions to Cerberus! To contribute:

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add new feature"
   ```

4. Push to your fork:

   ```bash
   git push origin feature-name
   ```

5. Open a pull request.

Please read our Contributing Guide for more details.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

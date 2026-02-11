# News-Spectrum

> A BERT-powered web application for analyzing the emotional tone of news articles to empower readers in identifying potential bias.

![Project Status](https://img.shields.io/badge/status-stable-success.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/flask-2.0+-green.svg)

## Overview

![App-Overview.png](https://github.com/jason-one-cpu/XJCO2913-Team-Project/blob/main/docs/imgs/App-Overview.png?raw=true)

**News-Spectrum** is a tool designed to bring transparency to news consumption. Instead of using LLMs to arbitrarily label news as "Fake" or "Biased," we use a Distil-BERT sentiment analysis model to identify the emotional coloring of the text.

We believe in Human-in-the-loop AI: our tool visualizes the data, but the final judgment remains with the user.

## Key Features

Unlike standard sentiment tools that just give you a single score, News-Spectrum offers a granular look at how news is written.

* **Sentence-Level "Spectrum"**: We break down the article and analyze every single sentence individually. This helps you spot exactly where the emotional language is hiding.
* **3-Class Detection**: Powered by FinBERT, our model accurately distinguishes between Positive, Negative, and Neutral tones. It won't flag objective facts as "biased."
* **Visual Confidence**: We use dynamic opacity to show the model's confidence.
    * Dark Green/Red: The model is 90%+ sure this is emotional.
    * Light Green/Red: Mild sentiment.
    * Transparent: Purely objective/neutral information.
* **Interactive Tooltips**: Hover over any part of the text to see the precise classification and confidence score in real-time.
* **Privacy-First Design**: Everything runs locally on your machine. No data is sent to external cloud servers.

## Tech Stack

* **Backend**: Python, Flask
* **AI Model**: PyTorch, Hugging Face Transformers
* **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript

## Quick Start

Follow these steps to set up the project locally.

#### 1. Clone the Repository
```bash
git clone https://github.com/jason-one-cpu/XJCO2913-Team-Project.git
cd XJCO2913-Team-Project
```

#### 2. Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# macOS/Linux
python3 -m venv news-spectrum
source news-spectrum/bin/activate

# Windows
python -m venv news-spectrum
news-spectrum\Scripts\activate
```

Alternatively, you can also use conda to create a virtual environment.

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Application

```bash
python run.py
```

You will see logs indicating the BERT model is initializing. Once ready, access the app at: `http://127.0.0.1:5000/`


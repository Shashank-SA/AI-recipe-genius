# 🧠🍲 Smart AI Recipe Genius

Smart AI Recipe Genius is a next-generation web application that blends **Artificial Intelligence**, **Computer Vision**, and **Natural Language Processing** to transform the way we cook. It detects vegetables from images, videos, or live camera feeds and generates personalized recipe suggestions based on the identified ingredients.

## 🚀 Features

- 🥦 **Real-Time Vegetable Detection** using YOLOv8
- 📸 Supports **image upload**, **video input**, and **live webcam feed**
- 🍽️ **Recipe generation** using GPT-4 based on selected cuisine and detected ingredients
- 🖥️ Built with **Streamlit (frontend)** and **Flask (backend)**
- 🌱 Promotes sustainability by reducing food waste through smart meal planning

## 🛠️ Tech Stack

- **Python**
- **YOLOv8** – Object Detection (Vegetable Classification)
- **OpenCV** – Image/Video Processing
- **Flask** – Backend API
- **Streamlit** – Frontend UI
- **OpenAI GPT-4 API** – Recipe Generation
- **HTML/CSS/JavaScript** (optional for custom UI)

## 📌 How It Works

1. User uploads or captures an image/video of ingredients.
2. YOLOv8 detects vegetables and passes the list to the backend.
3. User selects a preferred cuisine (e.g., Indian, Chinese, Italian).
4. GPT-4 generates a step-by-step recipe tailored to the ingredients and cuisine.
5. Recipe is displayed in a clean, readable format.

## 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-recipe-genius.git
   cd ai-recipe-genius

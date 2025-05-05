# 🏥 MediAssist - AI Healthcare Assistant

## 📌 Project Overview
MediAssist is an AI-powered healthcare assistant that provides:
- Symptom analysis and possible condition predictions
- Medical report and scan image analysis using AI
- Medication interaction and side-effect checking

This project uses **Google Gemini AI** to analyze medical reports, symptoms, and medications efficiently. It supports PDF medical reports (extracting images and text) and direct image uploads for medical scan analysis.

## ✨ Features
- ✅ **Symptom Checker**: Predicts possible conditions based on symptoms
- ✅ **Medical Report & Scan Analyzer**: Extracts and analyzes images and text from PDFs
- ✅ **Medication Analyzer**: Identifies drug interactions, side effects, and guidelines
- ✅ **AI-powered**: Uses **Google Gemini AI** for advanced medical analysis

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Diagnostic_Assistant.git
cd MediAssist
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Google Gemini API Key
Create a `.env` file and add your API key:
```plaintext
GEMINI_API_KEY=your_api_key_here
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🔧 Code Structure
```
MediAssist/
│── app.py                # Main Streamlit application
│── healthcare_agent.py    # AI processing logic
│── requirements.txt       # Python dependencies
│── .env                   # API Key storage (not included in repo)
└── README.md              # Project documentation
```



---

## 📸 **Screenshots**
## 🔍 **Symptom Checker UI**
![Symptom Checker](https://github.com/user-attachments/assets/d5a35363-9d44-4a57-91c1-861d40d93e1c)
![Symptom Checker](https://github.com/user-attachments/assets/47b2c6e7-f56e-4155-aaa3-5d338b5589c6)

## 📄 **Medical Report Analysis UI**
![Medical Report](https://github.com/user-attachments/assets/14fa702d-abd0-4532-a276-6bcb640b77f2)
![Medical Report](https://github.com/user-attachments/assets/c06530e9-7019-4fcb-ab6d-016e8f0766c3)
![Medical Report](https://github.com/user-attachments/assets/14af5671-ce04-4620-9355-c079800094dc)

## 💊 **Medication Analysis UI**
![Medication Analysis](https://github.com/user-attachments/assets/af35fdac-2f52-4e49-9b5a-32d61ffe747d)

---
## 🎥 **Experience MediAssist in Action!**  
[![Watch the Demo](https://img.shields.io/badge/📽️%20Watch%20Demo-Click%20Here-blue?style=for-the-badge)](https://drive.google.com/file/d/1K7xVughKGuXprT5dOArAo-OtTVFy0sdu/view?usp=drivesdk)


---

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.

## 📞 Contact
For questions or collaboration, reach out via:
- ✉ Email: krishnamadhumitadutta@gmail.com

---

_"MediAssist: AI-powered healthcare at your fingertips!"_ 🚀

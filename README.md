# 🔠 LetterLoopsSolver

LetterLoopsSolver is an advanced Python script designed to **solve letter-based puzzle games** by detecting letters from an image, applying intelligent filtering based on known connections, and suggesting the best possible words.

It utilizes **computer vision (OpenCV)** and **OCR (EasyOCR)** to extract letters, applies **graph-based heuristics** to determine valid letter connections, and provides **interactive filtering** for refining results.

---

## 🚀 Features
- 🖼️ **Image-based letter detection** (using OpenCV + EasyOCR)
- 🔎 **Automated candidate word generation** from a dictionary
- 🔢 **Filters candidates using known letter connections**
- ⌨️ **Interactive filtering**: Type known letter pairs to refine results
- 📋 **Auto-copies the best words** to the clipboard for easy input
- 🤖 **Corrects OCR misreads** (e.g., distinguishing "T" from "I")
- 🏆 **Ranks words based on frequency and relevance**

---

## 🛠 Installation

### **1️⃣ Clone this repository**
```sh
git clone https://github.com/yourusername/LetterLoopsSolver.git
cd LetterLoopsSolver
```

### **2️⃣ Install dependencies**
Ensure you have Python 3.x installed, then run:

```sh
pip install -r requirements.txt
```

> **Note:** You may need to install additional dependencies for OpenCV and EasyOCR if they are missing.

---

## 🖥️ Usage

### **🔍 Step 1: Copy an image of the puzzle**
Take a screenshot or copy an image containing **circular letter nodes**.

### **🛠 Step 2: Run the script**
```sh
python LetterLoopsSolver.py
```

### **🎯 Step 3: Refine the search interactively**
- Type **two letters** and press `SPACE` to confirm a **known connection**.
- Keep entering **pairs** to further refine the options.
- If you type **three letters in a row** (without a space), it auto-copies the top candidates.
- If filtering reduces results to **less than 6 options**, it auto-copies them.
- Press **a digit (1-9)** to select a word directly.

### **📋 Step 4: Paste the solution**
The best words will be **automatically copied to your clipboard** for easy pasting into the puzzle.

---

## 🔢 Logic & Functionality

### **1️⃣ Detecting Letters**
- Uses **EasyOCR** to extract letters from **Hough Circle-detected nodes** in an image.
- Corrects **common OCR mistakes** (e.g., `"0"` → `"O"`, `"1"` → `"I"`, `"6"` → `"G"`, `"8"` → `"B"`).

### **2️⃣ Fixing "T" and "I" Confusion**
- If the system **detects a second "T"**, it **must be an "I"** (and vice versa).
- If OCR detects **nothing (`""`)**, it assumes `"I"` (if missing) or `"T"` otherwise.

### **3️⃣ Generating Candidate Words**
- Filters words **containing only detected letters**.
- Uses **neighbor pairs** (confirmed adjacent letters) to **refine the list**.
- Scores words based on **connections and language frequency**.

### **4️⃣ Interactive Filtering**
- Allows **dynamic filtering** by typing letter connections.
- Auto-selects and **copies** words if the list **shrinks to 6 or fewer**.
- Provides **manual selection** (pressing a number to copy that word).

---

## 🛠️ Requirements
- **Python 3.x**
- **OpenCV** (`opencv-python`)
- **EasyOCR** (`easyocr`)
- **Pillow** (`PIL`)
- **NumPy**
- **PyAutoGUI** (`pyautogui`)
- **Pyperclip** (`pyperclip`)
- **NLTK** (`nltk`)

Install them all using:
```sh
pip install -r requirements.txt
```

---

## 📌 Known Issues & Improvements
✅ **Fix OCR misreadings** (T/I, 0/O, etc.)  
✅ **Enhance filtering logic** for better word selection  
⚡ **Optimize performance** for faster candidate ranking  
📖 **Add support for multiple dictionary sources**  

---

## 🤝 Contributing
Feel free to open **issues, feature requests, or pull requests**!

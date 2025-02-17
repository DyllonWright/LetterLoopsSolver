import time
import io
import os
import hashlib
import cv2
import numpy as np
import pyperclip
from PIL import ImageGrab, Image
import easyocr
from collections import Counter, defaultdict
from nltk.corpus import words
from itertools import permutations, combinations
import pyautogui

# -------------------- Global Setup --------------------
# Initialize EasyOCR reader (set gpu=True if you have CUDA support)
reader = easyocr.Reader(['en'], gpu=False)
USE_ANAGRAM_MODE = True  # using candidate mode

# -------------------- Dictionary / Candidate Helper Functions --------------------
def load_dictionary(filename='dictionary.txt'):
    """
    Loads a dictionary file (one word per line) and returns a list of uppercase words.
    If the file is not found, a small built-in list is returned.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            words_list = [line.strip().upper() for line in f if line.strip()]
        return words_list
    else:
        print("Dictionary file not found. Using a small built-in dictionary.")
        return ["HELLO", "WORLD", "TEST", "PUZZLE", "LETTER", "ANAGRAM", "EXAMPLE", "SOLUTION", "LOOPS", "PYTHON", "PROGRAM", "DEVELOP"]

dictionary_words = load_dictionary()

def find_candidates(letters, min_length, neighbor_pairs=None, farthest_pairs=None):
    """
    Given a set of detected letters and a minimum word length, return dictionary words that:
      - Contain only the detected letters,
      - (If neighbor_pairs is provided) contain at least one of those pairs.
    Then score each candidate word:
      - For every neighbor (true) pair found as a substring, add +1.
      - For every farthest (negative) pair found as a substring, subtract -1.
    If the final score is negative, omit that candidate.
    """
    MAX_LENGTH_PUZZLE = 18
    allowed_set = {letter.upper() for letter in letters}
    valid_candidates = []

    for word in dictionary_words:
        if len(word) < MAX_LENGTH_PUZZLE and len(word) > min_length:
            word_upper = word.upper()
            # Only allow words using the detected letters.
            if set(word_upper).issubset(allowed_set):
                # If neighbor_pairs is provided, only keep words that contain at least one.
                if neighbor_pairs and not any(pair in word_upper for pair in neighbor_pairs):
                    continue
                valid_candidates.append(word_upper)

    if neighbor_pairs and farthest_pairs:
        scored_candidates = []
        for word in valid_candidates:
            score = 0
            # Increase score for each neighbor (true) pair found.
            for pair in neighbor_pairs:
                if pair in word:
                    score += 1
            # Decrease score for each farthest (negative) pair found.
            for pair in farthest_pairs:
                if pair in word:
                    score -= 1
            # Only include candidate if final score is nonnegative.
            if score < 0:
                continue
            scored_candidates.append((word, score))
        # Sort by descending score.
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        valid_candidates = [word for word, score in scored_candidates]
    else:
        valid_candidates = sorted(valid_candidates)

    return valid_candidates

# -------------------- New Heuristics: Neighbor and Farthest Pairs --------------------
def get_neighbor_pairs(detected_nodes):
    """
    For each detected node (x, y, letter), determine its nearest neighbor (by Euclidean distance)
    and add both orders (e.g. if A's closest is L then add "AL" and "LA").
    """
    neighbor_pairs = set()
    n = len(detected_nodes)
    for i in range(n):
        x1, y1, letter1 = detected_nodes[i]
        best_dist = float('inf')
        best_neighbor = None
        for j in range(n):
            if i == j:
                continue
            x2, y2, letter2 = detected_nodes[j]
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_neighbor = letter2
        if best_neighbor:
            neighbor_pairs.add(letter1 + best_neighbor)
            neighbor_pairs.add(best_neighbor + letter1)
    return neighbor_pairs

def get_farthest_pairs(detected_nodes):
    """
    For each detected node (x, y, letter), find the farthest node (by Euclidean distance)
    and add both orders (e.g. if A's farthest is X then add "AX" and "XA").
    """
    farthest_pairs = set()
    n = len(detected_nodes)
    for i in range(n):
        x1, y1, letter1 = detected_nodes[i]
        max_dist = -1
        farthest_letter = None
        for j in range(n):
            if i == j:
                continue
            x2, y2, letter2 = detected_nodes[j]
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if dist > max_dist:
                max_dist = dist
                farthest_letter = letter2
        if farthest_letter:
            farthest_pairs.add(letter1 + farthest_letter)
            farthest_pairs.add(farthest_letter + letter1)
    return farthest_pairs

# -------------------- Ranking Helper --------------------
# Rank words based on frequency in the language
word_frequencies = Counter(dictionary_words)
def rank_words(candidates):
    """Ranks words based on general usage frequency."""
    return sorted(candidates, key=lambda word: word_frequencies[word.lower()], reverse=True)

# -------------------- Clipboard / Image Helper Functions --------------------
def get_image_hash(pil_image):
    """Return an MD5 hash for a PIL image (saved as PNG)."""
    with io.BytesIO() as output:
        pil_image.save(output, format="PNG")
        data = output.getvalue()
    return hashlib.md5(data).hexdigest()

def pil_to_cv2(pil_image):
    """Convert a PIL image to an OpenCV image (BGR)."""
    cv2_img = np.array(pil_image)
    if cv2_img.ndim == 3:
        if cv2_img.shape[2] == 4:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGBA2BGRA)
        else:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img

def composite_image_on_white(cv2_img):
    """
    If the image has an alpha channel, composite it on a white background.
    This helps if most of the image is transparent.
    """
    if cv2_img.shape[2] == 4:
        b, g, r, a = cv2.split(cv2_img)
        alpha = a.astype(float) / 255.0
        white_bg = np.ones(cv2_img.shape[:2], dtype=float) * 255
        b = cv2.multiply(alpha, b.astype(float)) + cv2.multiply(1 - alpha, white_bg)
        g = cv2.multiply(alpha, g.astype(float)) + cv2.multiply(1 - alpha, white_bg)
        r = cv2.multiply(alpha, r.astype(float)) + cv2.multiply(1 - alpha, white_bg)
        composed = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
        return composed
    else:
        return cv2_img

# -------------------- Image Processing Functions --------------------
def detect_letter_circles(cv2_img):
    """
    Use the Hough Circle Transform to detect letter nodes.
    Adjust parameters as needed.
    """
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=40,
                               param1=50,
                               param2=30,
                               minRadius=15,
                               maxRadius=50)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

def extract_letter_from_circle(cv2_img, circle):
    """
    Crop a region from the circle where the letter is expected.
    Crops a 30×30 region, scales it up, and runs OCR.
    """
    x, y, r = circle
    crop_size = 30  # fixed crop size
    half_crop = crop_size // 2
    x1 = max(x - half_crop, 0)
    y1 = max(y - half_crop + 2, 0)
    x2 = x1 + crop_size
    y2 = y1 + crop_size - 2
    crop = cv2_img[y1:y2, x1:x2].copy()

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
    
    results = reader.readtext(scaled_rgb, detail=1, paragraph=False)
    best_letter = ""
    best_conf = 0
    for bbox, text, conf in results:
        for char in text:
            if char == '0':
                return 'O'  # Manually fix the O problem
            if char == '1':
                return 'I'  # Manually fix the I problem
            if char == '6':
                return 'G' # Mangually fix the G problem
            if char == '8':
                return 'B' # Manually fix the B problem
            if char == 'T' and conf <0.6:
                char = 'I'
            if char.isalpha() and conf > best_conf:
                best_letter = char.upper()
                best_conf = conf
    return best_letter

def process_letterloops_cv2_image(cv2_img, use_anagram_mode=True):
    """
    Process the image by compositing, detecting circles, extracting letters,
    and generating candidate words. Also computes:
      - The neighbor pairs (true connections) and
      - The farthest pairs (negative connections).
    """
    composed = composite_image_on_white(cv2_img)
    circles = detect_letter_circles(composed)
    if circles is None:
        print("No circles detected.")
        return None, None, None, None

    # Collect (x, y, letter) for each detected circle.
    detected_nodes = []
    scanned_letters = []
    for circle in circles:
        letter = extract_letter_from_circle(composed, circle)
        if letter in {'T', 'I'}:
            if 'T' in scanned_letters and letter == 'T':
                letter = 'I'  # Convert duplicate "T" to "I"
            elif 'I' in scanned_letters and letter == 'I':
                letter = 'T'  # Convert duplicate "I" to "T"

        elif letter == "":  # If OCR detects nothing
            if 'I' not in scanned_letters:
                letter = 'I'  # Assume "I" if it's missing
            else:
                letter = 'T'  # Otherwise, assume "T"
        if letter and letter.isalpha():
            detected_nodes.append((circle[0], circle[1], letter))
            print(f"Detected letter at {(circle[0], circle[1])}: '{letter}'")
        scanned_letters.append(letter)
    letters = [node[2] for node in detected_nodes]
    #print("Collected letters:", letters)
    
    if use_anagram_mode:
        min_length = len(letters)
        neighbor_pairs = get_neighbor_pairs(detected_nodes)
        farthest_pairs = get_farthest_pairs(detected_nodes)
        #print("Candidate neighbor pairs:", neighbor_pairs)
        #print("Candidate farthest pairs:", farthest_pairs)
        candidates = find_candidates(letters, min_length, neighbor_pairs=neighbor_pairs, farthest_pairs=farthest_pairs)
        return candidates, letters, neighbor_pairs, farthest_pairs
    else:
        return None, letters, None, None

# -------------------- Interactive Filtering (Non-blocking) --------------------
def alt_tab_window():
    """Brings the terminal window back into focus."""
    time.sleep(0.1)  # Short delay to ensure smooth focus switch
    pyautogui.hotkey("alt", "tab")  # Simulate Alt+Tab to bring terminal to focus

def interactive_filtering(candidates):
    """
    Revised interactive filtering:
      - Type two letters then SPACE to confirm a connection (filter).
      - If you type a third letter (without a space) in the current block,
        the top three candidates are auto-copied.
      - You may add multiple confirmed pairs (separated by spaces) to refine the filter.
      - If the confirmed filter reduces candidates to < 3, they are auto-copied.
      - You can press a digit (1-9) to choose a candidate directly or ESC to cancel.
    """
    import msvcrt

    confirmed_pairs = []  # list of confirmed 2-letter connection pairs
    current_block = ""    # currently typed block (not yet confirmed)
    
    def compute_filtered_candidates():
        fc = candidates[:]
        for pair in confirmed_pairs:
            fc = [word for word in fc if pair in word]
        return fc

    def clear_console():
        os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        clear_console()
        print("Interactive Candidate Filtering Mode")
        #print("Instructions:")
        #print(" - Type two letters for a connection then press SPACE to confirm.")
        #print(" - If you type a third letter (without space), the top 3 candidates will be auto-copied.")
        #print(" - You may add multiple confirmed connections (separated by spaces) to refine the filter.")
        #print(" - Press digit (1-9) to select a candidate directly, or ESC to cancel.\n")
        print("Confirmed connections:", " ".join(confirmed_pairs))
        print("Current block: '{}'".format(current_block))
        print("-" * 40)
        
        filtered_candidates = compute_filtered_candidates()
        if filtered_candidates:
            for idx, word in enumerate(filtered_candidates, start=1):
                print(f"({idx}) {word}")
        else:
            print("No candidates match the confirmed connections.")
        print("-" * 40)
        
        # If filtering leaves fewer than 7 options, auto-copy them.
        if len(filtered_candidates) < 7:
            copy_text = "\n".join(filtered_candidates)
            pyperclip.copy(copy_text.lower())
            print("\3 or less candidates remain. Auto-copied:")
            print(copy_text)
            return filtered_candidates

        # Check for key press (non-blocking)
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.isdigit():
                sel_index = int(ch) - 1
                if 0 <= sel_index < len(filtered_candidates):
                    pyperclip.copy(filtered_candidates[sel_index].lower())
                    print(f"Copied candidate: {filtered_candidates[sel_index]}")
                    return filtered_candidates[sel_index]
            elif ch in ('\b', '\x08'):  # Backspace
                if current_block:
                    current_block = current_block[:-1]
                elif confirmed_pairs:
                    confirmed_pairs.pop()
            elif ch in ('\r', '\n'):
                # Ignore Enter key.
                pass
            elif ch == '\x1b':  # ESC key
                return None
            elif ch == ' ':
                # When SPACE is pressed:
                #   If current block is exactly 2 letters, confirm it.
                if len(current_block) == 2:
                    confirmed_pairs.append(current_block)
                current_block = ""
            elif ch.isalpha():
                current_block += ch.upper()
                if len(current_block) == 3:
                    # A 3-letter block (without a space) means “auto-copy top x”
                    topx = filtered_candidates[:6]
                    pyperclip.copy("\n".join(topx).lower())
                    print(f"Auto-copied top {len(topx)} candidates: {topx}")
                    return topx
            # Ignore other keys
        time.sleep(0.05)

# -------------------- Clipboard Monitoring Loop --------------------
def main():
    print("Monitoring clipboard for new images...")
    last_clip_hash = None

    while True:
        clipboard_content = ImageGrab.grabclipboard()
        if isinstance(clipboard_content, Image.Image):
            current_hash = get_image_hash(clipboard_content)
            if current_hash != last_clip_hash:
                last_clip_hash = current_hash
                print("New image detected; processing now...")
                cv2_img = pil_to_cv2(clipboard_content)
                candidates, letters, neighbor_pairs, farthest_pairs = process_letterloops_cv2_image(cv2_img, use_anagram_mode=USE_ANAGRAM_MODE)
                if not candidates:
                    print("No valid candidate words found.")
                else:
                    print("\nCandidates found:")
                    print(", ".join(candidates))
                    print("\nEntering interactive filtering mode...")
                    alt_tab_window()
                    chosen = interactive_filtering(candidates)
                    chosen_string = " ".join(chosen[:6]).lower() + (" " if chosen else "")
                    if chosen:
                        pyperclip.copy(chosen_string)
                        print(f"\nCopied candidate: {chosen_string}")
                        alt_tab_window()
                    else:
                        print("\nFiltering cancelled or no selection made.")
                    main()
        time.sleep(0.5)

if __name__ == '__main__':
    main()
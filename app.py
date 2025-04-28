from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from collections import Counter, deque
import pandas as pd
import qrcode
from collections import Counter
from io import BytesIO
import base64
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import re  # ✅ Added for validation

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model - This is where we need to create the model with the correct input shape
model = load_model(r'D:\Projects\DL\Grocery_app\Model\Fruit_Model.h5')

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["RetailStore"]

# Class labels
class_labels = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
    'radish', 'soy beans', 'spinach', 'sweetcorn', 'turnip', 'watermelon'
]

# Prices
item_prices = {
    'apple': 30, 'banana': 10, 'beetroot': 25, 'bell pepper': 35, 'cabbage': 20,
    'capsicum': 30, 'carrot': 20, 'cauliflower': 25, 'chilli pepper': 15, 'corn': 18,
    'cucumber': 12, 'eggplant': 22, 'garlic': 40, 'ginger': 35, 'grapes': 45,
    'jalepeno': 50, 'kiwi': 60, 'lemon': 10, 'lettuce': 30, 'mango': 50,
    'onion': 15, 'orange': 20, 'paprika': 40, 'pear': 25, 'peas': 18,
    'pineapple': 35, 'pomegranate': 45, 'potato': 12, 'radish': 10,
    'soy beans': 28, 'spinach': 20, 'sweetcorn': 22, 'turnip': 14, 'watermelon': 50
}

# Get the input shape from the model
input_shape = model.layers[0].input_shape
IMG_SIZE = input_shape[1] if input_shape and len(input_shape) >= 2 else 100  # Default to 100 if we can't determine

cap = cv2.VideoCapture(0)

EMAIL_ADDRESS = 'jacksparrowd492@gmail.com'
EMAIL_PASSWORD = 'xxwb kzrm fbrb ktbi'

detect_mode = False
current_detected_item = None
current_confidence = 0.0

# Global variables
last_detected_item = None
stable_start_time = None
saved_items = set()  # To track already saved items


# Buffer to smooth predictions
prediction_buffer = deque(maxlen=5)  # last 5 predictions


def generate_dynamic_qr(data, file_path):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img.save(file_path)
    print(f"[INFO] QR code generated and saved at {file_path}")


def send_transaction_email(recipient_email, transaction_id, payment_status, bill_qr_path):
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email

        # Decide subject and body based on payment status
        if payment_status.lower() == "success":
            subject = f"Payment Successful - Transaction ID: {transaction_id}"
            body = f"Thank you for your payment.\n\nTransaction ID: {transaction_id}\nPayment Status: SUCCESS"
        else:
            subject = f"Payment Failed - Transaction ID: {transaction_id}"
            body = f"Unfortunately, your payment has failed.\n\nTransaction ID: {transaction_id}\nPayment Status: FAILED"

        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        '''
        # Attach bill QR
        with open(bill_qr_path, 'rb') as f:
            bill_image = MIMEImage(f.read())
            bill_image.add_header('Content-Disposition', 'attachment', filename='bill_qr.png')
            msg.attach(bill_image)
        '''

        # Generate and attach dynamic payment status QR
        payment_qr_data = f"Transaction ID: {transaction_id}\nStatus: {payment_status.upper()}"
        payment_status_qr_path = f'payment_status_{transaction_id}.png'
        generate_dynamic_qr(payment_qr_data, payment_status_qr_path)

        with open(payment_status_qr_path, 'rb') as f:
            payment_status_image = MIMEImage(f.read())
            payment_status_image.add_header('Content-Disposition', 'attachment', filename=f'payment_status_{transaction_id}.png')
            msg.attach(payment_status_image)

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[INFO] Email sent to {recipient_email} for transaction {transaction_id} ({payment_status.upper()})")

        # Clean up temporary payment status QR file
        if os.path.exists(payment_status_qr_path):
            os.remove(payment_status_qr_path)

    except Exception as e:
        print(f"[ERROR] Failed to send transaction email: {e}")


def prepare_image(frame):
    # Using the IMG_SIZE determined from the model's input shape
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    img_array = np.expand_dims(normalized, axis=0)
    return img_array

def generate_frames(user_email=None):
    global current_detected_item, current_confidence, detect_mode
    global last_detected_item, stable_start_time, saved_items
    
    # Add new global variables for counting
    global stability_counter, last_count_time
    
    # Initialize counter variables if they don't exist
    if 'stability_counter' not in globals():
        stability_counter = 0
    if 'last_count_time' not in globals():
        last_count_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        label = "Waiting for detection..."
        counter_label = ""

        if detect_mode:
            img_array = prepare_image(frame)
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            if predicted_index < len(class_labels):
                predicted_class = class_labels[predicted_index]
                prediction_buffer.append(predicted_class)

                # Most common prediction
                if prediction_buffer:
                    most_common_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                    label = f"{most_common_prediction.capitalize()}"

                    current_detected_item = most_common_prediction
                    current_confidence = round(float(confidence) * 100, 2)

                    # Stability checking
                    if last_detected_item != current_detected_item:
                        last_detected_item = current_detected_item
                        stable_start_time = time.time()
                        stability_counter = 0  # Reset counter when item changes
                        last_count_time = None
                    else:
                        # If the same item is still being detected
                        if stable_start_time:
                            # Calculate how long the current item has been stable
                            current_stable_time = time.time() - stable_start_time
                            
                            # Check if we've reached another 3-second interval
                            if last_count_time is None or (time.time() - last_count_time) >= 3:
                                stability_counter += 1
                                last_count_time = time.time()
                                
                            # Show the counter in the display
                            counter_label = f" (Count: {stability_counter})"
                            
                            # After 3 seconds of stability (which means counter is at least 1)
                            if stability_counter >= 1 and user_email:
                                user_collection_name = f"DetectedItems_{user_email.replace(' ', '_')}"
                                user_collection = db[user_collection_name]

                                # If item was already saved, remove it
                                if current_detected_item in saved_items:
                                    user_collection.delete_one({"item": current_detected_item})
                                    saved_items.remove(current_detected_item)
                                    label = f"Removed {current_detected_item.capitalize()}"
                                else:
                                    # Save new detected item
                                    item_data = {
                                        "item": current_detected_item,
                                        "confidence": current_confidence,
                                        "count": stability_counter,  # Save the count in the database
                                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                    user_collection.insert_one(item_data)
                                    saved_items.add(current_detected_item)
                                    label = f"Saved {current_detected_item.capitalize()}"

                                # Reset after operation
                                stable_start_time = None
                                last_detected_item = None
                                stability_counter = 0
                                last_count_time = None

                else:
                    label = "Unknown"
                    current_detected_item = None
                    current_confidence = 0.0
                    stability_counter = 0
                    last_count_time = None

        # Display black background for text
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        # Add counter to the label if applicable
        display_label = label + counter_label
        cv2.putText(display_frame, display_label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()

        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html', username=session['user'])
    return redirect(url_for('login'))


@app.route('/video_feed')
def video_feed():
    user_email = session.get('email')
    return Response(generate_frames(user_email=user_email), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detect_mode
    detect_mode = not detect_mode
    return ('', 204)


@app.route('/generate_bill')
def generate_bill():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['email']
    user_collection_name = f"DetectedItems_{user_email.replace(' ', '_')}"
    user_collection = db[user_collection_name]
    items = list(user_collection.find({}))

    if not items:
        return "[INFO] No items were stored."

    item_names = [item['item'] for item in items]
    quantities = Counter(item_names)

    bill_data = []
    total_cost = 0
    purchase_time = time.strftime('%Y-%m-%d %H:%M:%S')

    for item, qty in quantities.items():
        price = item_prices.get(item, 0)
        cost = qty * price
        total_cost += cost
        bill_data.append({
            "Item": item.capitalize(),
            "Quantity": qty,
            "Unit Price": f"Rs.{price}",
            "Cost": f"Rs.{cost}"
        })

    summary = pd.DataFrame(bill_data)

    upi_id = "jacksparrowd492@oksbi"
    upi_name = "Retail Store"
    upi_link = f"upi://pay?pa={upi_id}&pn={upi_name}&am={total_cost}&cu=INR"

    # --- Generate QR Code dynamically ---
    qr = qrcode.make(upi_link)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    qr_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    qr_data_url = f"data:image/png;base64,{qr_base64}"
    # -------------------------------------

    return render_template("bill.html", table=summary.to_html(index=False),
                           total=total_cost, time=purchase_time,
                           upi_link=upi_link, qr_code=qr_data_url)


@app.route('/confirm_payment')
def confirm_payment():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['email']
    user_collection_name = f"DetectedItems_{user_email.replace(' ', '_')}"
    user_collection = db[user_collection_name]
    items = list(user_collection.find({}))

    if not items:
        return "[INFO] No items found."

    item_names = [item['item'] for item in items]
    quantities = Counter(item_names)

    bill_data = []
    total_cost = 0
    purchase_time = time.strftime('%Y-%m-%d %H:%M:%S')

    for item, qty in quantities.items():
        price = item_prices.get(item, 0)
        cost = qty * price
        total_cost += cost
        bill_data.append({
            "Item": item.capitalize(),
            "Quantity": qty,
            "Unit Price": f"Rs.{price}",
            "Cost": f"Rs.{cost}"
        })

    summary = pd.DataFrame(bill_data)

    # --- Simulate successful payment ---
    transaction_id = f"TXN{int(time.time())}"  # simple transaction id using timestamp
    payment_status = "Success"

    # --- Send confirmation email with transaction details ---
    send_transaction_email(
        recipient_email=user_email,
        transaction_id=transaction_id,
        payment_status=payment_status,
        bill_qr_path=None  # Not needed as we generate a new payment status QR
    )

    # Clear the user's detected items after successful payment
    user_collection.delete_many({})

    return render_template("payment_success.html", transaction_id=transaction_id,
                           total=total_cost, time=purchase_time, items=bill_data)



@app.route('/payment_success')
def payment_success():
    return render_template('payment_success.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Email validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "Invalid email format."

        # Phone number validation (10 digits)
        if not re.match(r"^\d{10}$", phone):
            return "Invalid phone number. Must be 10 digits."

        # Password validation (min 8 chars, 1 upper, 1 lower, 1 digit, 1 special char)
        if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$', password):
            return "Password must be at least 8 characters long and include uppercase, lowercase, number, and special character."

        if password != confirm_password:
            return "Passwords do not match."

        if db.users.find_one({'email': email}):
            return "User already exists!"

        db.users.insert_one({
            'name': name,
            'email': email,
            'phone': phone,
            'address': address,
            'password': password
        })
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = db.users.find_one({'email': email})
        if user and user['password'] == password:
            session['user'] = user['name']
            session['email'] = user['email']
            return redirect(url_for('index'))
        return "Invalid email or password."

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
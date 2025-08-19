# File: digit_recognizer.py
# Save this file in a folder called "digit_recognizer_project"

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import pickle

class DigitRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("800x600")
        
        # Drawing variables
        self.canvas_width = 400
        self.canvas_height = 400
        self.brush_size = 15
        
        # Create PIL image for drawing
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # ML Model
        self.model = None
        self.model_trained = False
        
        self.setup_ui()
        self.load_or_create_model()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Handwritten Digit Recognizer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, 
                               height=self.canvas_height, bg="white", 
                               relief=tk.RAISED, borderwidth=2)
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, padx=10, pady=10, sticky=tk.N)
        
        # Buttons
        ttk.Button(control_frame, text="Recognize Digit", 
                  command=self.recognize_digit).pack(pady=5, fill=tk.X)
        
        ttk.Button(control_frame, text="Clear Canvas", 
                  command=self.clear_canvas).pack(pady=5, fill=tk.X)
        
        ttk.Button(control_frame, text="Train Model", 
                  command=self.train_model).pack(pady=5, fill=tk.X)
        
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).pack(pady=5, fill=tk.X)
        
        # Brush size
        ttk.Label(control_frame, text="Brush Size:").pack(pady=(20,0))
        self.brush_scale = ttk.Scale(control_frame, from_=5, to=30, 
                                    value=self.brush_size, orient=tk.HORIZONTAL)
        self.brush_scale.pack(pady=5, fill=tk.X)
        self.brush_scale.configure(command=self.update_brush_size)
        
        # Prediction result
        ttk.Label(control_frame, text="Prediction:").pack(pady=(20,0))
        self.prediction_label = ttk.Label(control_frame, text="Draw a digit", 
                                         font=("Arial", 24, "bold"), 
                                         foreground="blue")
        self.prediction_label.pack(pady=10)
        
        # Confidence
        ttk.Label(control_frame, text="Confidence:").pack()
        self.confidence_label = ttk.Label(control_frame, text="0%")
        self.confidence_label.pack(pady=5)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to draw!")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)
        
    def start_draw(self, event):
        """Start drawing on canvas"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_on_canvas(self, event):
        """Draw on canvas when mouse is dragged"""
        # Draw on tkinter canvas
        self.canvas.create_oval(event.x - self.brush_size//2, 
                               event.y - self.brush_size//2,
                               event.x + self.brush_size//2, 
                               event.y + self.brush_size//2,
                               fill="black", outline="black")
        
        # Draw on PIL image
        self.draw.ellipse([event.x - self.brush_size//2, 
                          event.y - self.brush_size//2,
                          event.x + self.brush_size//2, 
                          event.y + self.brush_size//2],
                         fill="black", outline="black")
        
    def update_brush_size(self, value):
        """Update brush size from scale"""
        self.brush_size = int(float(value))
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit")
        self.confidence_label.config(text="0%")
        self.status_label.config(text="Canvas cleared!")
        
    def preprocess_image(self):
        """Preprocess the drawn image for prediction"""
        # Convert to grayscale
        gray_image = self.image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(gray_image)
        
        # Invert colors (black background, white digit)
        img_array = 255 - img_array
        
        # Resize to 28x28 (MNIST format)
        img_resized = cv2.resize(img_array, (28, 28))
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape for model input
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        return img_final
        
    def recognize_digit(self):
        """Recognize the drawn digit"""
        if not self.model_trained:
            messagebox.showwarning("Warning", "Model not trained yet! Please train the model first.")
            return
            
        try:
            # Preprocess image
            processed_img = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Update UI
            self.prediction_label.config(text=str(predicted_digit))
            self.confidence_label.config(text=f"{confidence:.1f}%")
            self.status_label.config(text=f"Predicted: {predicted_digit} (Confidence: {confidence:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            
    def create_model(self):
        """Create a CNN model for digit recognition"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model
        
    def train_model(self):
        """Train the model using MNIST dataset"""
        self.status_label.config(text="Training model... Please wait!")
        self.root.update()
        
        try:
            # Load MNIST dataset
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Preprocess data
            x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
            x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0
            
            # Create and train model
            self.model = self.create_model()
            
            # Train with progress updates
            history = self.model.fit(x_train, y_train,
                                   batch_size=128,
                                   epochs=5,
                                   validation_data=(x_test, y_test),
                                   verbose=1)
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            
            self.model_trained = True
            self.status_label.config(text=f"Model trained! Accuracy: {test_accuracy:.2%}")
            
            messagebox.showinfo("Success", f"Model trained successfully!\n"
                                         f"Test Accuracy: {test_accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_label.config(text="Training failed!")
            
    def save_model(self):
        """Save the trained model"""
        if not self.model_trained:
            messagebox.showwarning("Warning", "No trained model to save!")
            return
            
        try:
            # Create models directory if it doesn't exist
            if not os.path.exists("models"):
                os.makedirs("models")
                
            # Save model
            self.model.save("models/digit_recognizer_model.h5")
            
            # Save training status
            with open("models/model_status.pkl", "wb") as f:
                pickle.dump({"trained": True}, f)
                
            messagebox.showinfo("Success", "Model saved successfully!")
            self.status_label.config(text="Model saved!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
            
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists("models/digit_recognizer_model.h5"):
                self.model = keras.models.load_model("models/digit_recognizer_model.h5")
                
                # Check if model was trained
                if os.path.exists("models/model_status.pkl"):
                    with open("models/model_status.pkl", "rb") as f:
                        status = pickle.load(f)
                        self.model_trained = status.get("trained", False)
                        
                if self.model_trained:
                    self.status_label.config(text="Pre-trained model loaded!")
                else:
                    self.status_label.config(text="Model loaded but needs training!")
            else:
                self.status_label.config(text="No saved model found. Please train a new model!")
                
        except Exception as e:
            self.status_label.config(text="Failed to load model. Please train a new one!")
            
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Create popup window function
def create_popup_recognizer():
    """Create a popup window with the digit recognizer"""
    app = DigitRecognizer()
    app.run()

if __name__ == "__main__":
    # Create the main application
    print("Starting Handwritten Digit Recognizer...")
    print("This will open a popup window where you can draw digits!")
    
    # Check for required packages
    required_packages = ['tkinter', 'numpy', 'opencv-python', 'Pillow', 'tensorflow', 'matplotlib']
    print(f"Required packages: {', '.join(required_packages)}")
    print("If any package is missing, install it using: pip install package_name")
    print("-" * 50)
    
    # Start the application
    create_popup_recognizer()
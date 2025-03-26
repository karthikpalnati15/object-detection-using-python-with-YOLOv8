import cv2
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
from datetime import datetime

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Custom colors and styling
DARK_BG = "#2E3B4E"
LIGHT_BG = "#F5F5F5"
ACCENT_COLOR = "#4CAF50"
SECONDARY_COLOR = "#3498DB"
TEXT_COLOR = "#FFFFFF"
FONT = ("Helvetica", 10)
TITLE_FONT = ("Helvetica", 16, "bold")
BUTTON_FONT = ("Helvetica", 11)

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg=LIGHT_BG)
        self.root.minsize(800, 600)
        
        # Set app icon if available
        try:
            self.root.iconbitmap("app_icon.ico")  # Replace with your icon path
        except:
            pass
            
        # Load YOLOv8 model
        self.model = YOLO("yolov8s.pt").to(device)
        self.cap = None
        self.is_webcam_active = False
        self.confidence_threshold = 0.5
        self.saved_images_dir = "detected_images"
        
        # Create directory for saved images if it doesn't exist
        if not os.path.exists(self.saved_images_dir):
            os.makedirs(self.saved_images_dir)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.header_frame = tk.Frame(self.root, bg=DARK_BG, height=60)
        self.header_frame.pack(fill=tk.X)
        
        # App title
        title_label = tk.Label(self.header_frame, text="YOLOv8 Object Detection", 
                              font=TITLE_FONT, bg=DARK_BG, fg=TEXT_COLOR)
        title_label.pack(pady=15)
        
        # Create content frame with two columns
        self.content_frame = tk.Frame(self.root, bg=LIGHT_BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls
        self.control_panel = tk.Frame(self.content_frame, bg=LIGHT_BG, width=200)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for image display
        self.display_panel = tk.Frame(self.content_frame, bg="#FFFFFF", relief=tk.RIDGE, bd=1)
        self.display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display area with scrollbars
        self.canvas = tk.Canvas(self.display_panel, bg="#FFFFFF", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Label to display images or webcam feed
        self.image_label = tk.Label(self.canvas, bg="#FFFFFF")
        self.canvas.create_window((0, 0), window=self.image_label, anchor="nw")
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=DARK_BG, fg=TEXT_COLOR)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add controls to the control panel
        control_title = tk.Label(self.control_panel, text="Controls", font=("Helvetica", 12, "bold"), bg=LIGHT_BG)
        control_title.pack(pady=(0, 10), anchor="w")
        
        # Style for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=BUTTON_FONT, background=SECONDARY_COLOR)
        self.style.configure('Accent.TButton', font=BUTTON_FONT, background=ACCENT_COLOR)
        self.style.configure('TScale', background=LIGHT_BG)
        self.style.configure('TFrame', background=LIGHT_BG)
        
        # Confidence threshold slider
        conf_frame = ttk.Frame(self.control_panel)
        conf_frame.pack(fill=tk.X, pady=5)
        conf_label = tk.Label(conf_frame, text="Confidence Threshold:", bg=LIGHT_BG)
        conf_label.pack(anchor="w")
        
        self.conf_slider = ttk.Scale(conf_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                   value=self.confidence_threshold, command=self.update_confidence)
        self.conf_slider.pack(fill=tk.X, pady=5)
        
        self.conf_value_label = tk.Label(conf_frame, text=f"Value: {self.confidence_threshold:.2f}", bg=LIGHT_BG)
        self.conf_value_label.pack(anchor="w")
        
        # Buttons
        btn_frame = ttk.Frame(self.control_panel)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", 
                                   command=self.upload_image, style='TButton')
        self.upload_btn.pack(fill=tk.X, pady=5)
        
        self.webcam_btn = ttk.Button(btn_frame, text="Start Webcam", 
                                   command=self.toggle_webcam, style='Accent.TButton')
        self.webcam_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = ttk.Button(btn_frame, text="Save Current Image", 
                                 command=self.save_image, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=5)
        
        # Detection stats section
        stats_frame = ttk.LabelFrame(self.control_panel, text="Detection Stats")
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=25, font=FONT, bg="#F8F8F8", relief=tk.FLAT)
        self.stats_text.pack(fill=tk.X, pady=5)
        self.stats_text.insert(tk.END, "Upload an image or start\nwebcam to see detection stats")
        self.stats_text.config(state=tk.DISABLED)
        
        # Model info
        info_frame = ttk.LabelFrame(self.control_panel, text="System Info")
        info_frame.pack(fill=tk.X, pady=10)
        
        model_info = f"Model: YOLOv8s\nDevice: {device}\n"
        
        info_text = tk.Text(info_frame, height=4, width=25, font=FONT, bg="#F8F8F8", relief=tk.FLAT)
        info_text.pack(fill=tk.X, pady=5)
        info_text.insert(tk.END, model_info)
        info_text.config(state=tk.DISABLED)
        
        # Set initial state
        self.current_image = None

    def update_confidence(self, value):
        self.confidence_threshold = float(value)
        self.conf_value_label.config(text=f"Value: {self.confidence_threshold:.2f}")
        # Re-process current image if exists
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.process_frame(self.current_frame.copy())

    def upload_image(self):
        """Opens file dialog to select an image and runs YOLOv8 detection."""
        # Stop webcam if running
        self.stop_webcam()
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
            
        self.status_bar.config(text=f"Processing image: {os.path.basename(file_path)}")
        self.root.update()
        
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            self.status_bar.config(text="Error: Could not read image")
            return
            
        self.current_frame = img.copy()
        self.process_frame(img)
        self.save_btn.config(state=tk.NORMAL)
        self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")

    def process_frame(self, frame):
        """Process frame with YOLOv8 and display results."""
        # Run YOLOv8 detection
        results = self.model(frame)
        
        # Process results and draw bounding boxes
        processed_img = frame.copy()
        detections = {}
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]
                
                if conf >= self.confidence_threshold:
                    # Track detection count
                    if label in detections:
                        detections[label] += 1
                    else:
                        detections[label] = 1
                    
                    # Draw fancy box with fill
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add a small filled rectangle for the label background
                    text_size = cv2.getTextSize(f"{label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(processed_img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                    
                    # Add text label
                    cv2.putText(processed_img, f"{label} {conf:.2f}", (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Update stats display
        self.update_stats(detections)
        
        # Display the image
        self.display_image(processed_img)

    def display_image(self, img):
        """Display an OpenCV image in the Tkinter window."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Calculate size to maintain aspect ratio
        window_width = self.display_panel.winfo_width() - 20
        window_height = self.display_panel.winfo_height() - 20
        
        if window_width > 100 and window_height > 100:  # Ensure window has a valid size
            img_width, img_height = img_pil.size
            
            # Calculate scale factor to fit the window
            scale_width = window_width / img_width
            scale_height = window_height / img_height
            scale = min(scale_width, scale_height)
            
            # Resize the image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # Create PhotoImage and update label
        self.current_image = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image  # Keep a reference

    def update_stats(self, detections):
        """Update the detection stats display."""
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        if not detections:
            self.stats_text.insert(tk.END, "No objects detected")
        else:
            total = sum(detections.values())
            self.stats_text.insert(tk.END, f"Total objects: {total}\n\n")
            
            for label, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                self.stats_text.insert(tk.END, f"{label}: {count}\n")
        
        self.stats_text.config(state=tk.DISABLED)

    def toggle_webcam(self):
        """Toggle webcam on/off."""
        if self.is_webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        """Start webcam for real-time detection."""
        if self.cap is not None and self.cap.isOpened():
            return  # Webcam is already running
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_bar.config(text="Error: Could not open webcam")
            return
            
        self.is_webcam_active = True
        self.webcam_btn.config(text="Stop Webcam")
        self.status_bar.config(text="Webcam active - Press 'Stop Webcam' to end")
        self.save_btn.config(state=tk.NORMAL)
        
        self.update_webcam()

    def update_webcam(self):
        """Update webcam frame with detection."""
        if not self.is_webcam_active:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.process_frame(frame)
            
            # Continue updating
            self.root.after(10, self.update_webcam)
        else:
            self.stop_webcam()
            self.status_bar.config(text="Webcam disconnected")

    def stop_webcam(self):
        """Stop the webcam if it's running."""
        if self.is_webcam_active:
            self.is_webcam_active = False
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.webcam_btn.config(text="Start Webcam")
            self.status_bar.config(text="Webcam stopped")

    def save_image(self):
        """Save the current displayed image with detections."""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.saved_images_dir, f"detection_{timestamp}.jpg")
        
        # Run detection one more time to ensure consistent results
        results = self.model(self.current_frame)
        img = self.current_frame.copy()
        
        for result in results:
            for box in result.boxes:
                if box.conf[0].item() >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]
                    conf = box.conf[0].item()
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text_size = cv2.getTextSize(f"{label} {conf:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imwrite(filename, img)
        self.status_bar.config(text=f"Image saved: {filename}")

    def on_closing(self):
        """Handle window closing event."""
        self.stop_webcam()
        self.root.destroy()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')
model.conf = 0.25

# Trash class labels
TRASH_CLASSES = [
    "Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack",
    "Other plastic bottle", "Clear plastic bottle", "Glass bottle", "Plastic bottle cap",
    "Metal bottle cap", "Broken glass", "Food Can", "Aerosol", "Drink can", "Toilet tube",
    "Other carton", "Egg carton", "Drink carton", "Corrugated carton", "Meal carton",
    "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup",
    "Other plastic cup", "Food waste", "Glass jar", "Plastic lid", "Metal lid",
    "Other plastic", "Magazine paper", "Tissues", "Wrapping paper", "Normal paper",
    "Paper bag", "Plastified paper bag", "Plastic film", "Six pack rings",
    "Garbage bag", "Other plastic wrapper", "Single-use carrier bag",
    "Polypropylene bag", "Crisp packet", "Spread tub", "Tupperware",
    "Disposable food container", "Foam food container", "Other plastic container",
    "Plastic glooves", "Plastic utensils", "Pop tab", "Rope & strings",
    "Scrap metal", "Shoe", "Squeezable tube", "Plastic straw", "Paper straw",
    "Styrofoam piece", "Unlabeled litter", "Cigarette"
]

class TrashDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Trash Detection")
        self.video_capture = cv2.VideoCapture(0)
        self.running = True

        # Cumulative data
        self.cumulative_type_counts = {}
        self.cumulative_type_volumes = {}

        self.setup_gui()
        self.update_frame()

    def setup_gui(self):
        tab_control = ttk.Notebook(self.root)

        self.main_tab = ttk.Frame(tab_control)
        self.settings_tab = ttk.Frame(tab_control)

        tab_control.add(self.main_tab, text='Main Dashboard')
        tab_control.add(self.settings_tab, text='Settings')
        tab_control.pack(expand=1, fill='both')

        self.video_label = tk.Label(self.main_tab)
        self.video_label.pack()

        summary_frame = tk.Frame(self.main_tab)
        summary_frame.pack(fill='x')

        self.count_label = tk.Label(summary_frame, text="Live Trash Items: 0")
        self.count_label.pack(side='left', padx=10)

        self.volume_label = tk.Label(summary_frame, text="Live Estimated Volume: 0")
        self.volume_label.pack(side='left', padx=10)

        screenshot_button = tk.Button(self.main_tab, text="Take Screenshot", command=self.take_screenshot)
        screenshot_button.pack(pady=10)

        self.cumulative_tree = ttk.Treeview(self.main_tab, columns=("Type", "Count", "Volume"), show='headings', height=10)
        self.cumulative_tree.heading("Type", text="Trash Type")
        self.cumulative_tree.heading("Count", text="Number of Objects")
        self.cumulative_tree.heading("Volume", text="Total Estimated Volume")
        self.cumulative_tree.column("Type", width=200, anchor='w')
        self.cumulative_tree.column("Count", width=100, anchor='center')
        self.cumulative_tree.column("Volume", width=150, anchor='center')
        self.cumulative_tree.pack(fill='both', padx=10, pady=10)

        # Settings tab
        confidence_label = tk.Label(self.settings_tab, text="Confidence Threshold")
        confidence_label.pack(pady=5)

        self.confidence_slider = tk.Scale(self.settings_tab, from_=0, to=1, resolution=0.05,
                                          orient='horizontal', command=self.update_confidence)
        self.confidence_slider.set(model.conf)
        self.confidence_slider.pack()

        # Export CSV button
        export_button = tk.Button(self.settings_tab, text="Export All Screenshot Data to CSV", command=self.export_to_csv)
        export_button.pack(pady=10)

        exit_button = tk.Button(self.settings_tab, text="Exit", command=self.on_closing)
        exit_button.pack(pady=10)

    def update_confidence(self, value):
        model.conf = float(value)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.video_capture.read()
        if ret:
            results = model(frame)[0]
            detections = results.boxes
            annotated_frame = frame.copy()

            trash_count = 0
            total_volume = 0.0

            if detections is not None and detections.xyxy is not None:
                for i in range(len(detections.xyxy)):
                    conf = float(detections.conf[i])
                    if conf >= model.conf:
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        area = (x2 - x1) * (y2 - y1)
                        volume = area ** 0.5
                        total_volume += volume
                        trash_count += 1

                        color = (0, 255, 0)
                        cls_id = int(detections.cls[i])
                        label = TRASH_CLASSES[cls_id]

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            self.count_label.config(text=f"Live Trash Items: {trash_count}")
            self.volume_label.config(text=f"Live Estimated Volume: {total_volume:.2f}")

            image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def take_screenshot(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        results = model(frame)[0]
        detections = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        cls_ids = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        type_counts = {}
        type_volumes = {}

        for i, cls_id in enumerate(cls_ids):
            conf = confs[i]
            if conf < model.conf:
                continue
            label = TRASH_CLASSES[int(cls_id)]

            x1, y1, x2, y2 = detections[i]
            area = (x2 - x1) * (y2 - y1)
            volume = area ** 0.5

            type_counts[label] = type_counts.get(label, 0) + 1
            type_volumes[label] = type_volumes.get(label, 0) + volume

        for label, count in type_counts.items():
            self.cumulative_type_counts[label] = self.cumulative_type_counts.get(label, 0) + count
        for label, vol in type_volumes.items():
            self.cumulative_type_volumes[label] = self.cumulative_type_volumes.get(label, 0) + vol

        self.refresh_treeview()

    def refresh_treeview(self):
        self.cumulative_tree.delete(*self.cumulative_tree.get_children())
        for label in sorted(self.cumulative_type_counts.keys()):
            count = self.cumulative_type_counts[label]
            volume = self.cumulative_type_volumes.get(label, 0)
            self.cumulative_tree.insert("", "end", values=(label, count, f"{volume:.2f}"))

    def export_to_csv(self):
        if not self.cumulative_type_counts:
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv")],
                                                title="Save as")
        if not filepath:
            return

        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Trash Type", "Number of Objects", "Total Estimated Volume"])
            for label in sorted(self.cumulative_type_counts.keys()):
                count = self.cumulative_type_counts[label]
                volume = self.cumulative_type_volumes.get(label, 0)
                writer.writerow([label, count, f"{volume:.2f}"])

    def on_closing(self):
        self.running = False
        self.video_capture.release()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = TrashDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

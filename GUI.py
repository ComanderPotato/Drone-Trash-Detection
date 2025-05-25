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

def get_class_color(label):
    """Generate a consistent color for each class label."""
    np.random.seed(hash(label) % 2**32)
    return np.random.randint(50, 255, size=3, dtype=np.uint8)

class TrashDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 Drone Trash Detection System")
        self.video_capture = cv2.VideoCapture(0)
        self.running = True

        self.cumulative_type_counts = {}
        self.cumulative_type_volumes = {}

        self.setup_gui()
        self.update_frame()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'))
        style.configure("Treeview", font=('Segoe UI', 9), rowheight=25)
        style.configure("TButton", font=('Segoe UI', 10))
        style.configure("TLabel", font=('Segoe UI', 10))

        self.root.configure(bg="#f0f2f5")

        tab_control = ttk.Notebook(self.root)
        self.main_tab = ttk.Frame(tab_control, padding=10)
        self.settings_tab = ttk.Frame(tab_control, padding=10)

        tab_control.add(self.main_tab, text='📷 Main Dashboard')
        tab_control.add(self.settings_tab, text='⚙️ Settings')
        tab_control.pack(expand=1, fill='both')

        self.video_label = tk.Label(self.main_tab, bd=2, relief="solid", bg="black")
        self.video_label.pack(pady=10)

        summary_frame = tk.Frame(self.main_tab, bg="#f0f2f5")
        summary_frame.pack(fill='x', pady=5)

        self.count_label = tk.Label(summary_frame, text="Live Trash Items: 0", font=('Segoe UI', 10, 'bold'), bg="#f0f2f5")
        self.count_label.pack(side='left', padx=15)

        self.volume_label = tk.Label(summary_frame, text="Live Estimated Volume: 0", font=('Segoe UI', 10, 'bold'), bg="#f0f2f5")
        self.volume_label.pack(side='left', padx=15)

        screenshot_button = ttk.Button(self.main_tab, text="📸 Take Screenshot", command=self.take_screenshot)
        screenshot_button.pack(pady=10)

        self.cumulative_tree = ttk.Treeview(self.main_tab, columns=("Type", "Count", "Volume"), show='headings', height=10)
        self.cumulative_tree.heading("Type", text="Trash Type")
        self.cumulative_tree.heading("Count", text="Number of Objects")
        self.cumulative_tree.heading("Volume", text="Total Estimated Volume")
        self.cumulative_tree.column("Type", width=200, anchor='w')
        self.cumulative_tree.column("Count", width=120, anchor='center')
        self.cumulative_tree.column("Volume", width=150, anchor='center')
        self.cumulative_tree.pack(fill='both', padx=10, pady=10)

        tk.Label(self.settings_tab, text="Confidence Threshold", font=('Segoe UI', 10, 'bold')).pack(pady=(0, 5))

        self.confidence_slider = tk.Scale(self.settings_tab, from_=0, to=1, resolution=0.05,
                                          orient='horizontal', command=self.update_confidence,
                                          length=300, troughcolor="#cccccc", bg="#f0f2f5")
        self.confidence_slider.set(model.conf)
        self.confidence_slider.pack(pady=(0, 20))

        export_button = ttk.Button(self.settings_tab, text="📁 Export Screenshot Data to CSV", command=self.export_to_csv)
        export_button.pack(pady=10)

        exit_button = ttk.Button(self.settings_tab, text="❌ Exit", command=self.on_closing)
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
            masks = results.masks
            annotated_frame = frame.copy()

            trash_count = 0
            total_volume = 0.0

            if masks is not None and detections is not None:
                cls_ids = detections.cls.cpu().numpy()
                for i, mask in enumerate(masks.data.cpu().numpy()):
                    label = TRASH_CLASSES[int(cls_ids[i])]
                    binary_mask = (mask * 255).astype(np.uint8)
                    binary_mask = cv2.resize(binary_mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
                    color_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                    color = get_class_color(label)
                    color_mask[binary_mask > 127] = color
                    annotated_frame = cv2.addWeighted(annotated_frame, 1.0, color_mask, 1.0, 0)

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
        masks = results.masks

        if masks is not None and results.boxes is not None:
            for i, mask in enumerate(masks.data.cpu().numpy()):
                label = TRASH_CLASSES[int(cls_ids[i])]
                binary_mask = (mask * 255).astype(np.uint8)
                binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
                color_mask = np.zeros_like(frame, dtype=np.uint8)
                color = get_class_color(label)
                color_mask[binary_mask > 127] = color
                frame = cv2.addWeighted(frame, 1.0, color_mask, 0.8, 0)

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

import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    np.random.seed(hash(label) % 2**32)
    return np.random.randint(50, 255, size=3, dtype=np.uint8)

class TrashDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 Drone Trash Detection System")
        self.video_capture = cv2.VideoCapture(0)
        self.running = True
        self.current_figure = None  # For storing current visualization

        self.cumulative_type_counts = {}
        self.cumulative_type_volumes = {}
        self.screenshot_data = {}  # key: filename, value: {label: (count, volume)}

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

        tab_control.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main tab: divide into left and right frames
        self.left_frame = ttk.Frame(self.main_tab)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.right_frame = ttk.Frame(self.main_tab)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.main_tab.grid_rowconfigure(0, weight=1)
        self.main_tab.grid_columnconfigure(0, weight=1)
        self.main_tab.grid_columnconfigure(1, weight=1)

        # Left side widgets: video, buttons, table
        self.video_label = tk.Label(self.left_frame, bd=2, relief="solid", bg="black")
        self.video_label.pack(pady=10, fill='both', expand=True)

        # Summary labels (live count and volume)
        summary_frame = tk.Frame(self.left_frame, bg="#f0f2f5")
        summary_frame.pack(fill='x', pady=5)

        self.count_label = tk.Label(summary_frame, text="Live Trash Items: 0", font=('Segoe UI', 10, 'bold'), bg="#f0f2f5")
        self.count_label.pack(side='left', padx=15)

        self.volume_label = tk.Label(summary_frame, text="Live Estimated Volume: 0", font=('Segoe UI', 10, 'bold'), bg="#f0f2f5")
        self.volume_label.pack(side='left', padx=15)

        # Buttons horizontally - centered horizontally
        button_frame = tk.Frame(self.left_frame, bg="#f0f2f5")
        button_frame.pack(pady=10)

        inner_button_frame = tk.Frame(button_frame, bg="#f0f2f5")
        inner_button_frame.pack()

        screenshot_button = ttk.Button(inner_button_frame, text="📸 Take Screenshot", command=self.take_screenshot)
        screenshot_button.pack(side='left', padx=5)

        delete_button = ttk.Button(inner_button_frame, text="🗑️ Remove Screenshot", command=self.remove_screenshot)
        delete_button.pack(side='left', padx=5)

        visualize_button = ttk.Button(inner_button_frame, text="📊 Visualize Recorded Data", command=self.visualize_data)
        visualize_button.pack(side='left', padx=5)

        # Treeview table
        self.cumulative_tree = ttk.Treeview(self.left_frame, columns=("Type", "Count", "Volume"), show='headings', height=15)
        self.cumulative_tree.heading("Type", text="Trash Type")
        self.cumulative_tree.heading("Count", text="Number of Objects")
        self.cumulative_tree.heading("Volume", text="Total Estimated Volume")
        self.cumulative_tree.column("Type", width=200, anchor='w')
        self.cumulative_tree.column("Count", width=120, anchor='center')
        self.cumulative_tree.column("Volume", width=150, anchor='center')
        self.cumulative_tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Right side visualization frame
        self.visualization_frame = ttk.Frame(self.right_frame)
        self.visualization_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Default "No Current Visualization" label
        self.no_vis_label = tk.Label(self.visualization_frame, text="No Current Visualization", font=('Segoe UI', 14, 'italic'))
        self.no_vis_label.pack(expand=True)

        # Settings tab
        tk.Label(self.settings_tab, text="Confidence Threshold", font=('Segoe UI', 10, 'bold')).pack(pady=(0, 5))

        self.confidence_slider = tk.Scale(self.settings_tab, from_=0, to=1, resolution=0.05,
                                        orient='horizontal', command=self.update_confidence,
                                        length=300, troughcolor="#cccccc", bg="#f0f2f5")
        self.confidence_slider.set(model.conf)
        self.confidence_slider.pack(pady=(0, 20))

        load_model_button = ttk.Button(self.settings_tab, text="📦 Load Detection Model", command=self.load_model)
        load_model_button.pack(pady=5)

        export_button = ttk.Button(self.settings_tab, text="📁 Export Data to CSV", command=self.export_to_csv)
        export_button.pack(pady=5)

        export_vis_button = ttk.Button(self.settings_tab, text="📊 Export Current Visualization", 
                                     command=self.export_visualization)
        export_vis_button.pack(pady=5)

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
                    conf = float(detections.conf[i])
                    if conf < model.conf:
                        continue
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

        self.screenshot_data[filename] = {label: (type_counts[label], type_volumes[label]) for label in type_counts}

        for label, count in type_counts.items():
            self.cumulative_type_counts[label] = self.cumulative_type_counts.get(label, 0) + count
        for label, vol in type_volumes.items():
            self.cumulative_type_volumes[label] = self.cumulative_type_volumes.get(label, 0) + vol

        self.refresh_treeview()

    def remove_screenshot(self):
        top = tk.Toplevel(self.root)
        top.title("Remove Screenshots")
        top.geometry("400x350")

        listbox = tk.Listbox(top, selectmode=tk.MULTIPLE)
        listbox.pack(fill='both', expand=True, padx=10, pady=10)

        for filename in self.screenshot_data.keys():
            listbox.insert(tk.END, filename)

        def delete_selected():
            selections = listbox.curselection()
            if not selections:
                messagebox.showwarning("Warning", "Please select screenshot(s) to remove.")
                return

            deleted_files = []

            for idx in reversed(selections):
                selected_file = listbox.get(idx)
                if selected_file in self.screenshot_data:
                    data = self.screenshot_data.pop(selected_file)
                    for label, (count, vol) in data.items():
                        self.cumulative_type_counts[label] -= count
                        self.cumulative_type_volumes[label] -= vol
                        if self.cumulative_type_counts[label] <= 0:
                            self.cumulative_type_counts.pop(label, None)
                        if self.cumulative_type_volumes[label] <= 0:
                            self.cumulative_type_volumes.pop(label, None)
                    deleted_files.append(selected_file)
                    listbox.delete(idx)

            self.refresh_treeview()
            messagebox.showinfo("Info", f"Removed: {', '.join(deleted_files)}")

        remove_btn = ttk.Button(top, text="Remove Selected", command=delete_selected)
        remove_btn.pack(pady=10)

    def refresh_treeview(self):
        for item in self.cumulative_tree.get_children():
            self.cumulative_tree.delete(item)

        for label in sorted(self.cumulative_type_counts):
            count = self.cumulative_type_counts[label]
            volume = self.cumulative_type_volumes.get(label, 0)
            self.cumulative_tree.insert('', 'end', values=(label, count, f"{volume:.2f}"))

    def visualize_data(self):
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        if not self.cumulative_type_counts:
            self.no_vis_label = tk.Label(self.visualization_frame, text="No Current Visualization", 
                                       font=('Segoe UI', 14, 'italic'))
            self.no_vis_label.pack(expand=True)
            messagebox.showinfo("Info", "No data to visualize.")
            return

        labels = list(self.cumulative_type_counts.keys())
        counts = [self.cumulative_type_counts[label] for label in labels]
        volumes = [self.cumulative_type_volumes.get(label, 0) for label in labels]

        # Get the dimensions of the visualization frame
        width = self.visualization_frame.winfo_width() / 100  # Convert to inches
        height = self.visualization_frame.winfo_height() / 100  # Convert to inches
        
        # Ensure minimum dimensions
        width = max(width, 6)
        height = max(height, 6)
        
        # Create figure with adjusted size and DPI
        fig, axs = plt.subplots(2, 1, figsize=(width, height), dpi=100)
        fig.tight_layout(pad=3.0)

        # Bar chart for counts
        axs[0].bar(labels, counts, color='skyblue')
        axs[0].set_ylabel('Number of Objects')
        axs[0].set_title('Trash Items Count')
        
        # Bar chart for volumes
        axs[1].bar(labels, volumes, color='salmon')
        axs[1].set_ylabel('Estimated Volume')
        axs[1].set_title('Trash Items Estimated Volume')

        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Store the current figure for potential export
        self.current_figure = fig

    def export_visualization(self):
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            messagebox.showinfo("Info", "No visualization to export.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                              filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not filename:
            return

        try:
            self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Visualization exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save visualization: {e}")

    def export_to_csv(self):
        if not self.cumulative_type_counts:
            messagebox.showinfo("Info", "No data in the table to export.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                              filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Date", "Trash Type", "Count", "Estimated Volume"])
                current_date = datetime.now().strftime("%d.%m.%Y")
                
                for label in sorted(self.cumulative_type_counts):
                    count = self.cumulative_type_counts[label]
                    volume = self.cumulative_type_volumes.get(label, 0)
                    writer.writerow([current_date, label, count, f"{volume:.2f}"])
                    
            messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                global model
                model = YOLO(filepath)
                model.conf = self.confidence_slider.get()
                messagebox.showinfo("Success", f"Model loaded from:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")


    def on_closing(self):
        self.running = False
        self.video_capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1920x1080")
    app = TrashDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
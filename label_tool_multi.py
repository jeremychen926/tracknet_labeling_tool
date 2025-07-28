from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem, QLabel,
    QHBoxLayout, QVBoxLayout, QSlider, QPushButton, QMessageBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QIcon
import cv2, json, sys, os
from collections import OrderedDict
# import pandas as pd

# Predefined colors for annotation IDs
COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 165, 0), QColor(128, 0, 128), QColor(0, 255, 255),
    QColor(255, 192, 203), QColor(128, 128, 0)
]
# Number of previous frames to overlay
HISTORY = 5

class VideoView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(0, 0, parent.frame_w, parent.frame_h)

    def mousePressEvent(self, event):
        scene_pt = self.mapToScene(event.pos())
        x, y = int(scene_pt.x()), int(scene_pt.y())
        mw = self.window()
        if 0 <= x < mw.frame_w and 0 <= y < mw.frame_h:
            if event.button() == Qt.LeftButton:
                mw.add_point(x, y)
            else:
                mw.remove_point(x, y)

    def wheelEvent(self, event):
        mw = self.window()
        delta = event.angleDelta().y()
        if delta == 0:
            return
        new_val = mw.zoom_slider.value() + (1 if delta > 0 else -1)
        mw.zoom_slider.setValue(new_val)

class MainWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # df = pd.read_csv('./video/CameraReader_0_meta.csv')
        # self.timestamps = df['timestamp'].values

        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break   
            # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            self.frames.append(frame)
        self.cap.release()
        if not self.frames:
            raise ValueError("Cannot load video or video has no frames.")
    
        # ret, frame0 = self.cap.read()

        # self.total_frames = len(self.frames)
        self.total_frames = len(self.frames)
        self.frame_h, self.frame_w = self.frames[0].shape[:2]
        self.coordinates = [[] for _ in range(self.total_frames)]

        self.load_json()

        self.scale = 1.0
        self.playback_speed = 1.0

        self.video_view = VideoView(self)
        font = QFont()
        
        # --- Zoom controls ---
        font.setPointSize(14)
        self.zoom_slider = QSlider(Qt.Vertical)
        self.zoom_slider.setRange(0, 18)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        self.zoom_label = QLabel("Zoom: 1.0x")
        self.zoom_label.setFont(font)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_layout = QVBoxLayout()
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_widget = QWidget()
        zoom_widget.setLayout(zoom_layout)

        # --- ID list ---
        self.id_list = QListWidget()
        item0 = QListWidgetItem("ID 0")
        item0.setIcon(self.create_color_icon(0))
        self.id_list.addItem(item0)
        self.id_list.setCurrentRow(0)
        self.current_id = 0
        self.id_list.currentRowChanged.connect(self.change_label_id)
        self.id_list.setFont(font)

        # --- Show history or not ---
        self.show_history = True
        self.history_checkbox = QCheckBox("Show last 5 frames")
        self.history_checkbox.setFont(font)
        self.history_checkbox.setChecked(True)        # default on
        self.history_checkbox.stateChanged.connect(self.on_history_toggled)

        # --- Frame/ID display ---
        font.setPointSize(20)
        self.frame_label = QLabel(f"Frame: 0/{self.total_frames-1}")
        self.frame_label.setFont(font)
        self.current_id_label = QLabel(f"Current ID: {self.current_id}")
        self.current_id_label.setFont(font)

        # --- Navigation buttons ---
        font.setPointSize(16)
        self.add_id_btn = QPushButton("Add ID")
        self.add_id_btn.setFont(font)
        self.add_id_btn.clicked.connect(self.add_new_id)
        self.play_btn = QPushButton("Play")
        self.play_btn.setFont(font)
        self.play_btn.clicked.connect(self.play_pause)

        self.prev50_btn = QPushButton("<< 50")
        self.prev50_btn.clicked.connect(lambda: self.seek_frame(self.current_frame - 50))
        self.prev1_btn = QPushButton("< 1")
        self.prev1_btn.clicked.connect(lambda: self.seek_frame(self.current_frame - 1))
        self.next1_btn = QPushButton("1 >")
        self.next1_btn.clicked.connect(lambda: self.seek_frame(self.current_frame + 1))
        self.next50_btn = QPushButton("50 >>")
        self.next50_btn.clicked.connect(lambda: self.seek_frame(self.current_frame + 50))

        # --- Playback speed control ---
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_label.setFont(font)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.speed_changed)
        self.speed_slider.setMaximumWidth(150)

        # --- Frame slider ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames-1)
        self.slider.sliderMoved.connect(self.seek_frame)

        # --- Brightness controls ---
        self.brightness = 1.0  # alpha
        self.brightness_step = 0.1
        font.setPointSize(14)
        self.brighten_btn = QPushButton("Brighten")
        self.brighten_btn.setFont(font)
        self.darken_btn = QPushButton("Darken")
        self.darken_btn.setFont(font)
        self.brighten_btn.clicked.connect(self.on_brighten)
        self.darken_btn.clicked.connect(self.on_darken)

        # --- Layout assembly ---
        id_panel = QWidget()
        id_layout = QVBoxLayout(id_panel)
        id_layout.addWidget(self.frame_label)
        id_layout.addWidget(self.current_id_label)
        id_layout.addWidget(self.add_id_btn)
        id_layout.addWidget(self.id_list)
        id_layout.addWidget(self.history_checkbox)

        ctrl_panel = QWidget()
        cp_layout = QHBoxLayout(ctrl_panel)
        cp_layout.addWidget(self.prev50_btn)
        cp_layout.addWidget(self.prev1_btn)
        cp_layout.addWidget(self.slider)
        cp_layout.addWidget(self.next1_btn)
        cp_layout.addWidget(self.next50_btn)
        cp_layout.addWidget(self.play_btn)
        cp_layout.addWidget(self.speed_label)
        cp_layout.addWidget(self.speed_slider)
        cp_layout.addWidget(self.brighten_btn)
        cp_layout.addWidget(self.darken_btn)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        h_layout = QHBoxLayout()
        h_layout.addWidget(id_panel, 2)
        h_layout.addWidget(self.video_view, 12)
        h_layout.addWidget(zoom_widget, 1)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(ctrl_panel)

        self.setCentralWidget(central)

        # --- Timer setup ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.current_frame = 0
        self.update_frame()

        self.video_view.resetTransform()
        self.video_view.fitInView(self.video_view.sceneRect(), Qt.KeepAspectRatio)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_view.resetTransform()
        self.video_view.fitInView(self.video_view.sceneRect(), Qt.KeepAspectRatio)

    def create_color_icon(self, id_):
        # Create a colored circle icon for the given ID
        size = 16
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setBrush(COLORS[id_ % len(COLORS)])
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)
        painter.end()
        return QIcon(pix)
    
    def on_history_toggled(self, state):
        self.show_history = (state == Qt.Checked)
        self.update_frame()

    def zoom_changed(self, value):
        target_scale = 1.1 ** value
        factor = target_scale / self.scale
        self.video_view.scale(factor, factor)
        self.scale = target_scale
        self.zoom_label.setText(f"Zoom: {target_scale:.1f}x")

    def speed_changed(self, val):
        self.playback_speed = val / 10.0
        self.speed_label.setText(f"Speed: {self.playback_speed:.1f}x")
        if self.timer.isActive():
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            if self.current_frame >= self.total_frames - 1:
                self.current_frame = 0
                self.update_frame()
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_btn.setText("Pause")

    def next_frame(self):
        if self.current_frame + 1 < self.total_frames:
            self.current_frame += 1
            self.update_frame()
        else:
            self.timer.stop()
            self.play_btn.setText("Play")

    def get_unused_label_id(self):
        used = {int(self.id_list.item(i).text().split()[1]) for i in range(self.id_list.count())}
        i = 0
        while i in used:
            i += 1
        return i

    def change_label_id(self, idx):
        if idx >= 0:
            self.current_id = int(self.id_list.item(idx).text().split()[1])
            self.current_id_label.setText(f"Current ID: {self.current_id}")

    def add_new_id(self):
        nid = self.get_unused_label_id()
        row = self.insert_id_sorted(nid)
        self.id_list.setCurrentRow(row)

    def insert_id_sorted(self, nid):
        # Insert a new ID item in sorted order with its color icon
        for i in range(self.id_list.count()):
            existing = int(self.id_list.item(i).text().split()[1])
            if nid < existing:
                item = QListWidgetItem(f"ID {nid}")
                item.setIcon(self.create_color_icon(nid))
                self.id_list.insertItem(i, item)
                return i
        item = QListWidgetItem(f"ID {nid}")
        item.setIcon(self.create_color_icon(nid))
        self.id_list.addItem(item)
        return self.id_list.count()-1

    def seek_frame(self, idx):
        idx = max(0, min(self.total_frames-1, idx))
        if idx == 0:
            used = {p[0] for p in self.coordinates[0]}
            labels = sorted(int(self.id_list.item(i).text().split()[1]) for i in range(self.id_list.count()))
            for lbl in labels:
                if lbl not in used:
                    did = lbl
                    break
            else:
                did = self.get_unused_label_id()
            for i in range(self.id_list.count()):
                if int(self.id_list.item(i).text().split()[1]) == did:
                    self.id_list.setCurrentRow(i)
                    self.current_id = did
                    break
        elif idx > 0:
            last_non_empty_idx = -1
            for i in range(idx - 1, -1, -1):
                if self.coordinates[i]:
                    last_non_empty_idx = i
                    break

            used = {p[0] for p in self.coordinates[idx]}
            labels = sorted(int(self.id_list.item(i).text().split()[1]) for i in range(self.id_list.count()))

            if last_non_empty_idx >= 0:
                prev_ids = [p[0] for p in self.coordinates[last_non_empty_idx]]
                m = min(prev_ids)
                if m not in used:
                    did = m
                else:
                    did = next((l for l in labels if l > m and l not in used), self.get_unused_label_id())
            else:
                did = next((l for l in labels if l not in used), self.get_unused_label_id())

            for i in range(self.id_list.count()):
                if int(self.id_list.item(i).text().split()[1]) == did:
                    self.id_list.setCurrentRow(i)
                    self.current_id = did
                    break

        self.current_frame = idx
        self.update_frame()

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Z, Qt.Key_Left):
            self.seek_frame(self.current_frame - 1)
        elif k in (Qt.Key_X, Qt.Key_Right):
            self.seek_frame(self.current_frame + 1)
        elif k == Qt.Key_D:
            self.seek_frame(self.current_frame - 50)
        elif k == Qt.Key_F:
            self.seek_frame(self.current_frame + 50)
        elif k == Qt.Key_W:
            self.id_list.setCurrentRow(max(0, self.id_list.currentRow()-1))
        elif k == Qt.Key_S:
            self.id_list.setCurrentRow(min(self.id_list.count()-1, self.id_list.currentRow()+1))
        elif k == Qt.Key_Space:
            self.play_pause()
        else:
            super().keyPressEvent(event)

    def add_point(self, x, y):
        pts = [(i, px, py) for (i, px, py) in self.coordinates[self.current_frame] if i != self.current_id]
        pts.append((self.current_id, x, y))
        self.coordinates[self.current_frame] = pts
        self.update_frame()
        used = {p[0] for p in pts}
        labels = sorted(int(self.id_list.item(i).text().split()[1]) for i in range(self.id_list.count()))
        nid = next((l for l in labels if l > self.current_id and l not in used), self.get_unused_label_id())
        row = (self.insert_id_sorted(nid) if nid not in labels
               else next(i for i in range(self.id_list.count()) if int(self.id_list.item(i).text().split()[1]) == nid))
        self.id_list.setCurrentRow(row)

    def remove_point(self, x, y):
        pts = self.coordinates[self.current_frame]
        if not pts:
            return
        idx, _ = min(enumerate(pts), key=lambda t: (x-pts[t[0]][1])**2 + (y-pts[t[0]][2])**2)
        rid, _, _ = pts[idx]
        pts.pop(idx)
        self.coordinates[self.current_frame] = pts
        self.update_frame()
        for i in range(self.id_list.count()):
            if int(self.id_list.item(i).text().split()[1]) == rid:
                self.id_list.setCurrentRow(i)
                break

    def on_brighten(self):
        self.brightness = min(self.brightness + self.brightness_step, 3.0)
        self.update_frame()

    def on_darken(self):
        self.brightness = max(self.brightness - self.brightness_step, 0.1)
        self.update_frame()

    def update_frame(self):
        # Start with current RGB frame copy
        src = self.frames[self.current_frame].copy()

        display = cv2.convertScaleAbs(src, alpha=self.brightness, beta=0)
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        # self.cap.set(cv2.CAP_PROP_POS_MSEC, (self.timestamps[self.current_frame] - self.timestamps[0]) * 1000)
    
        marker_radius = max(1, int(min(self.frame_h, self.frame_w) * 0.002))
        font_scale = max(0.3, marker_radius / 4.0)
        thickness = max(1, marker_radius)
        print(marker_radius, font_scale, thickness)
        # marker_radius = 1
        # font_scale = 0.3
        # thickness = 1
        # Overlay previous frames with decreasing opacity
        if self.show_history:
            for h in range(1, HISTORY+1):
                idx = self.current_frame - h
                if idx < 0:
                    break
                alpha = (HISTORY+1 - h) / (HISTORY+1)
                overlay = display.copy()
                for id_, px, py in self.coordinates[idx]:
                    color = COLORS[id_ % len(COLORS)]
                    bgr = (color.blue(), color.green(), color.red())
                    cv2.circle(overlay, (px, py), marker_radius, bgr, -1)
                # Blend overlay onto display
                cv2.addWeighted(overlay, alpha, display, 1-alpha, 0, display)
        # Draw current frame annotations fully opaque
        for id_, px, py in self.coordinates[self.current_frame]:
            color = COLORS[id_ % len(COLORS)]
            bgr = (color.blue(), color.green(), color.red())
            cv2.circle(display, (px, py), marker_radius, bgr, -1)
            text_pos = (px + marker_radius + 1, py - marker_radius - 1)
            cv2.putText(display, str(id_), text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, thickness)
        # Update UI elements
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.total_frames-1}")
        self.current_id_label.setText(f"Current ID: {self.current_id}")
        h, w, ch = display.shape
        img = QImage(display.data, w, h, ch*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(img)
        self.video_view.pixmap_item.setPixmap(pix)
        self.video_view.scene.setSceneRect(0, 0, self.frame_w, self.frame_h)
        self.slider.setValue(self.current_frame)

    def closeEvent(self, event):
        ans = QMessageBox.question(
            self, "Confirm Exit", "Save and quit?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if ans == QMessageBox.Yes:
            self.save_json()
            event.accept()
        elif ans == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    def load_json(self):
        path = "json/output_ball.json"
        if not os.path.exists(path):
            print(f"JSON file {path} does not exist, skipping load.")
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load failed",
                                f"無法讀取 {path}:\n{e}")
            return

        # data 是 list of { "Frame": fid, "Objects": [ {id, X, Y}, ... ] }
        for entry in data:
            fid = entry.get("Frame")
            if fid is None or fid < 0 or fid >= self.total_frames:
                continue
            pts = []
            for obj in entry.get("Objects", []):
                i = obj.get("id")
                x = obj.get("X")
                y = obj.get("Y")
                if i is None or x is None or y is None:
                    continue
                pts.append((i, x, y))
            self.coordinates[fid] = pts

    def save_json(self):
        data = []
        for fid, pts in enumerate(self.coordinates):
            if not pts:
                continue
            data.append({
                "Frame": fid,
                "Objects": [{"id": i, "X": x, "Y": y} for i, x, y in pts]
            })
        with open("output_ball.json", "w") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(sys.argv[1])
    w.showMaximized()
    sys.exit(app.exec_())

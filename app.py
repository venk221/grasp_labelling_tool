import os
import sys
import glob
import json
import time
import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from typing import List
from canvas import Canvas
from label_list import LabelListWidget
from file_list import FileListWidget
from tool_bar import ToolBar
import torch
import random
from grasp import GraspRect
from grasp_inference import predict_grasps_for_image
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning) 
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
IMAGE_EXTENTIONS = ["jpg", "png"]

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import utils
import action

class MainWindow(QMainWindow):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setWindowTitle("My Label Tool")
        self.canvas = Canvas(self)
        self.dirty = False
        self.last_resize_time = 0
        self.label_list = LabelListWidget()
        self.shape_dock = QDockWidget(self.tr(u"Grasp list"), self)
        self.shape_dock.setObjectName(u"Grasp list")
        self.shape_dock.setWidget(self.label_list)
        self.canvas.shapesAdded.connect(self.label_list.addShapes)
        self.canvas.gradientPointSelected.connect(self.run_gradient_inference)
        self.canvas.shapesRemoved.connect(self.label_list.removeShapes)
        self.canvas.shapesSelectionChanged.connect(self.label_list.changeShapesSelection)
        self.canvas.shapesAreaChanged.connect(self.label_list.updateShapesArea)
        self.label_list.shapesRemoved.connect(self.canvas.removeShapes)
        self.label_list.shapesSelectionChanged.connect(self.canvas.changeShapesSelection)
        self.label_list.shapeVisibleChanged.connect(self.canvas.changeShapesVisible)
        self.label_list.shapesOrderChanged.connect(self.canvas.changeShapesOrder)
        self.label_list.reconnectCanvasDataRequest.connect(self._reconnectCanvasAndLabel)
        self.file_list = FileListWidget()
        self.file_list.filesSelectionChanged.connect(self._changeFilesSelection)
        self.file_list.fileLabeledChanged.connect(self._changeFileLabeled)
        self.file_dock = QDockWidget(self.tr(u"File list"), self)
        self.file_dock.setObjectName(u"File list")
        self.file_dock.setWidget(self.file_list)
        self.canvas.shapesAdded.connect(self.setDirty)
        self.canvas.shapesRemoved.connect(self.setDirty)
        self.canvas.shapesAreaChanged.connect(self.setDirty)
        self.label_list.shapesRemoved.connect(self.setDirty)
        self.label_list.shapesOrderChanged.connect(self.setDirty)
        self.file_list.fileLabeledChanged.connect(self.setDirty)
        features = QDockWidget.DockWidgetFeatures()
        self.shape_dock.setFeatures(features | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.shape_dock.setVisible(True)
        self.file_dock.setFeatures(features | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        self.file_dock.setVisible(True)
        self.setCentralWidget(self.canvas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.shape_dock)
        self.image_folder = None
        self.image_files = None
        self.output_folder = None
        self.output_name = None
        self.results = {
            "image_folder": "unknown",
            "image_files": {}
        }
        openProject = action.new_action(
            self, self.tr("Open Project"), lambda: self.importProject(self.openProjectDialog()),
            None, "open-project.png"
        )
        saveProject = action.new_action(
            self, self.tr("Save Project"), self.saveProject, "Ctrl+S", "save.png"
        )
        openImages = action.new_action(
            self, self.tr("Open Images"), lambda: self.importImages(self.openImagesDialog()),
            None, "open-images.png"
        )
        openDir = action.new_action(
            self, self.tr("Open Dir"), lambda: self.importDirImages(self.openDirDialog()),
            None, "open-dir.png"
        )
        openPrevImg = action.new_action(
            self, self.tr("Prev Image"), self.openPrevImg, "A", "prev.png"
        )
        openNextImg = action.new_action(
            self, self.tr("Next Image"), self.openNextImg, "D", "next.png"
        )
        createMode = action.new_action(
            self, self.tr("Create Mode"), lambda: self.canvas.setMode(self.canvas.CREATE),
            "Ctrl+N", "create.png"
        )
        editMode = action.new_action(
            self, self.tr("Edit Mode"), lambda: self.canvas.setMode(self.canvas.EDIT),
            "Ctrl+E", "edit.png"
        )
        gradientMode = action.new_action(
            self, self.tr("Gradient Mode"), lambda: self.canvas.setMode(self.canvas.GRADIENT),
            "Ctrl+M", "gradient.png"
        )
        rotateClockwise = action.new_action(
            self, self.tr("Rotate Clockwise"), self.rotateGraspClockwise, "R", "rotate-cw.png"
        )
        rotateAnticlockwise = action.new_action(
            self, self.tr("Rotate Anticlockwise"), self.rotateGraspAnticlockwise, "Shift+R", "rotate-ccw.png"
        )
        fitWindow = action.new_action(
            self, self.tr("Fit Window"), lambda: self.canvas.adjustPainter("fit_window"),
            None, "fit-window.png"
        )
        fitOrigin = action.new_action(
            self, self.tr("Origin Size"), lambda: self.canvas.adjustPainter("origin_size"),
            None, "origin-size.png"
        )
        changeOutputDir = action.new_action(
            self, self.tr("Change Output Dir"), self.changeOutputDir, None, "open-dir.png"
        )
        inference_action = action.new_action(
            self, "&Grasp Inference", self.run_model_inference, "Ctrl+G", "inference.png"
        )
        self.actions = utils.Struct(
            openPrevImg=openPrevImg, openNextImg=openNextImg
        )
        self.tool_bar = self.addToolBar_(
            "Tools",
            [
                openProject, openDir, openPrevImg, openNextImg, saveProject,
                createMode, editMode, gradientMode, rotateClockwise, rotateAnticlockwise,
                fitWindow, fitOrigin, inference_action
            ]
        )
        self.menus = utils.Struct(
            file=self.addMenu(self.tr("File")),
            edit=self.addMenu(self.tr("Edit")),
            view=self.addMenu(self.tr("View")),
            tools=self.addMenu(self.tr("Tools"))
        )
        action.add_actions(self.menus.tools, [inference_action])
        action.add_actions(
            self.menus.file,
            [openProject, openImages, openDir, None, saveProject, changeOutputDir]
        )
        action.add_actions(
            self.menus.edit,
            [createMode, editMode, gradientMode, rotateClockwise, rotateAnticlockwise]
        )
        action.add_actions(
            self.menus.view,
            [fitWindow, fitOrigin]
        )

    def run_model_inference(self):
        if not self.image_files:
            print("[ERROR] No image files loaded.")
            return
        selected_indexes = self.file_list.selectedIndexes()
        if not selected_indexes:
            print("[ERROR] No image selected for inference.")
            return
        current_idx = selected_indexes[0].row()
        current_file = self.image_files[current_idx]
        rgb_path = os.path.join(self.image_folder or "", current_file)
        depth_path = rgb_path.replace('_rgb.png', '_depth.tiff')
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"[ERROR] Cannot load RGB image: {rgb_path}")
            return
        rgb = rgb.astype('float32') / 255.0
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = depth[..., None]
            stacked = np.concatenate([rgb, depth], axis=-1)
        else:
            stacked = rgb
            print(f"[INFO] No depth image found at: {depth_path}. Running RGB-only inference.")
        predicted_grasps = predict_grasps_for_image(stacked, model_path=None, model_type='cornell')
        gui_grasps = []
        for g in predicted_grasps:
            print(f"[INFO] Predicted grasp: {g.points}")
            points = np.copy(g.points)
            swapped_points = np.zeros_like(points)
            swapped_points[:, 0] = points[:, 1]
            swapped_points[:, 1] = points[:, 0]
            try:
                gui_grasp = GraspRect(points=swapped_points)
                gui_grasps.append(gui_grasp)
            except Exception as e:
                print(f"[ERROR] Failed to convert grasp: {e}")
        print(f"[INFO] Adding {[str(g) for g in gui_grasps]} grasps to canvas...")
        self.canvas.addShapes(gui_grasps)
        self.file_list[current_idx].setCheckState(Qt.Checked)
        self.canvas.setMode(self.canvas.EDIT)
        self.setDirty()
        print(f"[INFO] Inference complete. Added {len(gui_grasps)} grasps.")

    def run_gradient_inference(self, click_pos: QPointF):
        print(f"[INFO] run_gradient_inference triggered with click_pos=({click_pos.x()}, {click_pos.y()})")
        if not self.image_files:
            print("[ERROR] No image files loaded.")
            return
        selected_indexes = self.file_list.selectedIndexes()
        if not selected_indexes:
            print("[ERROR] No image selected for gradient inference.")
            return
        current_idx = selected_indexes[0].row()
        current_file = self.image_files[current_idx]
        rgb_path = os.path.join(self.image_folder or "", current_file)
        depth_path = rgb_path.replace('_rgb.png', '_depth.tiff')
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"[ERROR] Failed to load images: RGB={rgb_path}, Depth={depth_path}")
            return
        print(f"[INFO] Depth image shape: {depth.shape}, min: {depth.min()}, max: {depth.max()}")
        x, y = int(click_pos.x()), int(click_pos.y())
        if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
            print(f"[ERROR] Click position ({x}, {y}) out of bounds for image size ({depth.shape[1]}, {depth.shape[0]})")
            return
        center_depth = float(depth[y, x])
        print(f"[INFO] Clicked depth value: {center_depth}")
        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[(depth > center_depth - 0.1) & (depth < center_depth + 0.1)] = 255  # Relaxed threshold
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel for robustness
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[ERROR] No contours found. Try adjusting depth threshold or click position.")
            return
        nearest_point = self.find_nearest_edge_point(contours, (x, y))
        selected_contour = None
        for cnt in contours:
            if nearest_point in cnt.squeeze().tolist():
                selected_contour = cnt
                break
        if selected_contour is None:
            selected_contour = contours[0]
        angle = self.compute_tangent_at_point(selected_contour, nearest_point)
        angle_rad = np.radians(angle)
        width, height = 10, 40
        center = np.array([x, y], dtype=np.float64)
        grasp = GraspRect.from_model_output(
            center=center,
            gripper_open=width,
            gripper_size=height,
            angle=angle_rad
        )
        print(f"[INFO] Adding grasp: center=({x}, {y}), angle={angle:.2f}°")
        self.canvas.addShapes([grasp])
        self.file_list[current_idx].setCheckState(Qt.Checked)
        self.canvas.setMode(self.canvas.EDIT)
        self.setDirty()
        self.canvas.update()  # Force canvas update
        print("[INFO] Gradient inference complete. Grasp added.")

    def find_nearest_edge_point(self, contours, point):
        min_dist = float('inf')
        closest = None
        for contour in contours:
            for pt in contour:
                dist = np.linalg.norm(np.array(pt[0]) - np.array(point))
                if dist < min_dist:
                    min_dist = dist
                    closest = tuple(pt[0])
        return closest

    def compute_tangent_at_point(self, contour, target_point):
        contour = np.squeeze(contour)
        idx = np.argmin(np.linalg.norm(contour - target_point, axis=1))
        prev_idx = (idx - 1) % len(contour)
        next_idx = (idx + 1) % len(contour)
        delta = contour[next_idx] - contour[prev_idx]
        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return angle

    def rotateGraspClockwise(self):
        if self.canvas.mode == self.canvas.EDIT:
            self.canvas.rotateSelectedGrasps(clockwise=True)
            self.setDirty()

    def rotateGraspAnticlockwise(self):
        if self.canvas.mode == self.canvas.EDIT:
            self.canvas.rotateSelectedGrasps(clockwise=False)
            self.setDirty()

    def _reconnectCanvasAndLabel(self):
        self.label_list.reconnectCanvasData(self.canvas.shapes)

    def _changeFilesSelection(self, selected, deselected):
        assert len(selected) <= 1, "Single selection mode"
        assert len(deselected) <= 1, "Single selection mode"
        if len(deselected):
            print("[INFO] [from_app] Saving current work...")
            current_file = self.image_files[deselected[0]]
            self.results["image_files"][current_file]["shapes"] = self.canvas.exportShapes()
        if len(selected):
            current_file = self.image_files[selected[0]]
            print("[INFO] [from_app] Loading data for image {}...".format(current_file))
            if current_file not in self.results["image_files"]:
                self.canvas.clear()
            else:
                self.canvas.loadShapes(self.results["image_files"][current_file]["shapes"])
            if self.image_folder is not None:
                self.canvas.loadImage(os.path.join(self.image_folder, current_file))
            else:
                self.canvas.loadImage(current_file)
        self.setClean()

    def _changeFileLabeled(self, index: int, labeled: bool):
        file = self.image_files[index]
        self.results["image_files"][file]["labeled"] = labeled

    def setDirty(self):
        self.dirty = True

    def setClean(self):
        self.dirty = False

    def isDirty(self):
        return self.dirty

    def addMenu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            action.add_actions(menu, actions)
        return menu

    def addToolBar_(self, title, actions=None):
        tool_bar = ToolBar(title)
        tool_bar.setObjectName("{}ToolBar".format(title))
        tool_bar.setOrientation(Qt.Vertical)
        tool_bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            action.add_actions(tool_bar, actions)
        self.addToolBar(Qt.LeftToolBarArea, tool_bar)
        return tool_bar

    def openProjectDialog(self):
        path = QFileDialog.getOpenFileName(
            self, self.tr("Open Project"), "./", self.tr("Project File (*.json)")
        )[0]
        return path

    def openImagesDialog(self):
        paths = QFileDialog.getOpenFileNames(
            self, self.tr("Open Images"), "./",
            self.tr("Image Files ({})".format(" ".join(["*." + ext for ext in IMAGE_EXTENTIONS])))
        )[0]
        return paths

    def openDirDialog(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Open Directory"), "./", QFileDialog.ShowDirsOnly
        )
        if len(path) == 0:
            path = None
        print("[INFO] [from app] Choose dir = {}".format(path))
        return path

    def importProject(self, path: str):
        if not path:
            return
        self.file_list.clear()
        self.canvas.clear()
        with open(path, "r", encoding="utf-8") as j:
            self.results = json.load(j)
        self.image_folder = self.results["image_folder"].lower() \
            if self.results["image_folder"].lower() != "absolute_path" else None
        self.image_files = list(self.results["image_files"].keys())
        self.image_files.sort()
        if self.image_folder is None:
            found, not_found = [], []
            for file in self.image_files:
                if os.path.exists(file):
                    found.append(file)
                else:
                    not_found.append(file)
            self.image_files = found
            if len(not_found):
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setText("{} file(s) cannot be found.".format(len(not_found)))
                box.setInformativeText("Do you want the project to keep them?")
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                box.setDefaultButton(QMessageBox.No)
                box.setDetailedText("The following file(s) cannot be found:\n\n" +
                                    "\n".join(["  - " + f for f in not_found]))
                ret = box.exec()
                if ret == QMessageBox.No:
                    for f in not_found:
                        self.results["image_files"].pop(f)
        else:
            if not os.path.exists(self.image_folder):
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Critical)
                box.setText("Directory not found. Import project failed.")
                box.setInformativeText("Select directory: {}".format(self.image_folder))
                box.setStandardButtons(QMessageBox.Ok)
                box.setDefaultButton(QMessageBox.Ok)
                box.exec()
                return
            else:
                found, not_found = [], []
                for file in self.image_files:
                    if os.path.exists(os.path.join(self.image_folder, file)):
                        found.append(file)
                    else:
                        not_found.append(file)
                self.image_files = found
                if len(not_found):
                    box = QMessageBox(self)
                    box.setIcon(QMessageBox.Warning)
                    box.setText("{} file(s) in directory \"{}\" cannot be found."
                                .format(len(not_found), self.image_folder))
                    box.setInformativeText("Do you want the project to keep them?")
                    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    box.setDefaultButton(QMessageBox.No)
                    box.setDetailedText(
                        "The following file(s) cannot be found:\n\n" +
                        "\n".join(["  - " + f for f in not_found])
                    )
                    ret = box.exec()
                    if ret == QMessageBox.No:
                        for f in not_found:
                            self.results["image_files"].pop(f)
        self.file_list.addFiles(self.image_files)
        for i, file in enumerate(self.image_files):
            self.file_list[i].setCheckState(Qt.Checked if self.results["image_files"][file]["labeled"]
                                            else Qt.Unchecked)
        self.file_list.selectNext()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setText("Continue working on this opened project?")
        box.setInformativeText("If not, you may have to choose another output path"
                               " when you save the project.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        ret = box.exec()
        if ret == QMessageBox.Yes:
            self.output_folder, self.output_name = os.path.split(path)
            if self.output_folder == "":
                self.output_folder = "."
            self.setClean()
        else:
            self.setDirty()

    def importImages(self, paths: List[str]):
        if len(paths) == 0:
            return
        self.file_list.clear()
        self.canvas.clear()
        self.image_folder = None
        self.image_files = paths
        self.image_files.sort()
        self.file_list.addFiles(self.image_files)
        self.results = {
            "image_folder": "absolute_path",
            "image_files": {
                f: {"labeled": False, "shapes": []} for f in self.image_files
            }
        }
        self.file_list.selectNext()
        self.setDirty()

    def importDirImages(self, path: str):
        if not path:
            return
        self.file_list.clear()
        self.canvas.clear()
        self.image_folder = path
        self.image_files = []
        for ext in IMAGE_EXTENTIONS:
            self.image_files.extend(glob.glob(os.path.join(self.image_folder, "*.{}".format(ext))))
        self.image_files = [os.path.split(f)[-1] for f in self.image_files]
        self.image_files.sort()
        self.file_list.addFiles(self.image_files)
        self.results = {
            "image_folder": path,
            "image_files": {
                f: {"labeled": False, "shapes": []} for f in self.image_files
            }
        }
        self.file_list.selectNext()
        self.setDirty()

    def openNextImg(self):
        print("[INFO] Open next image triggered.")
        current_select, next_select = self.file_list.selectNext()
        if current_select is not None:
            self.file_list[current_select].setCheckState(Qt.Checked)

    def openPrevImg(self):
        print("[INFO] Open previous image triggered.")
        current_select, prev_select = self.file_list.selectPrev()
        if current_select is not None:
            self.file_list[current_select].setCheckState(Qt.Checked)

    def saveProject(self):
        print("[INFO] [from app] Saving current work...")
        selected = [i.row() for i in self.file_list.selectedIndexes()]
        assert len(selected) <= 1, "Single selection mode."
        if len(selected):
            current_file = self.image_files[selected[0]]
            self.results["image_files"][current_file]["shapes"] = self.canvas.exportShapes()
            self.file_list[selected[0]].setCheckState(Qt.Checked)
        if self.output_folder is None:
            self.output_folder = self.openDirDialog()
            if self.output_folder is None:
                return None
            time_stamp = time.strftime("%m%d%H%M%S", time.localtime())
            self.output_name = "proj_" + time_stamp + ".json"
        path = os.path.join(self.output_folder, self.output_name)
        print("[INFO] [from app] Saving project to {}...".format(path))
        with open(path, "w", encoding="utf-8") as j:
            json.dump(self.results, j, ensure_ascii=False, indent=4)
        self.setClean()
        return path

    def changeOutputDir(self):
        self.output_folder = self.openDirDialog()
        return self.output_folder

    def resizeEvent(self, e: QResizeEvent):
        super(MainWindow, self).resizeEvent(e)
        current_time = time.time()
        if current_time - self.last_resize_time > 0.1:
            self.canvas.adjustPainter("fit_window")
            self.last_resize_time = current_time

    def closeEvent(self, e: QCloseEvent):
        if not self.dirty:
            e.accept()
        else:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setText("Save before closing? Or press cancel to back to canvas.")
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Yes)
            ret = box.exec()
            if ret == QMessageBox.Yes:
                out_path = self.saveProject()
                if out_path is None:
                    box = QMessageBox(self)
                    box.setIcon(QMessageBox.Information)
                    box.setText("Saving project canceled. Will go back to canvas.")
                    box.setStandardButtons(QMessageBox.Ok)
                    box.setDefaultButton(QMessageBox.Ok)
                    box.exec()
                    e.ignore()
                else:
                    e.accept()
            elif ret == QMessageBox.No:
                e.accept()
            else:
                e.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
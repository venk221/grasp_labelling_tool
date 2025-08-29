# LabelGrasp: A Grasp Detection Labeling Tool

**LabelGrasp** is a PyQt5-based application designed for labeling grasp poses in images. It integrates with grasp detection models to visualize and refine predictions, streamlining the process of creating labeled datasets for robot grasping tasks.

## Key Features

* **Image Loading:** Supports loading individual images or entire directories of images.
* **Grasp Labeling:** Provides tools to create and edit grasp annotations (e.g., rectangles) directly on the image.
* **Model Inference Integration:** Allows users to load and run grasp detection models (e.g., GG-CNN, GRCNN, GPNN) to generate initial grasp predictions.
* **Visualization of Predictions:** Displays model-predicted grasps overlaid on the image, enabling easy comparison and refinement.
* **Label Management:** A dedicated list view to manage created grasp labels, including visibility control and ordering.
* **Project Management:** Save and load projects to preserve annotations and continue work across sessions.
* **File List Management:** Navigate through images within a project using a file list.

##  Functionality

The application combines a graphical interface for manual labeling with the ability to incorporate automated grasp predictions:

1.  **Image Loading:** The user can load images or a directory of images into the application.
2.  **Model Inference:**
    * The application loads pre-trained grasp detection models.
    * When triggered, the loaded model processes the current image to predict potential grasp locations.
    * These predictions are converted into visual representations (rectangles with orientation) and displayed on the image.
3.  **Grasp Labeling/Editing:**
    * Users can create new grasp labels manually using tools within the application.
    * Existing labels (whether manually created or model-generated) can be edited to adjust position, size, and orientation.
4.  **Label Management:** All created labels are listed in a separate panel, allowing for selection, visibility toggling, and ordering.
5.  **Project Save/Load:** Users can save their work as a project file, preserving image paths and label data. This allows for resuming labeling sessions.

##  Code Structure

The project is organized as follows:

* `app.py`:  The main application logic, handling UI elements, file management, and interaction between components.
* `canvas.py`:   Implements the image display and drawing area where users create and edit grasp labels. It also handles displaying the model's grasp predictions.
* `label_list.py`:  Defines the widget for displaying and managing the list of grasp labels.
* `file_list.py`:   Defines the widget for displaying the list of image files in the project.
* `tool_bar.py`:   Creates the toolbar with buttons and actions for various application functions.
* `inference.py`:   (Separate file) Contains the code for loading grasp detection models, preprocessing images, running inference, and post-processing the model's output.
* `utils.py`:   Provides utility functions.
* `action.py`:   Defines helper functions to create actions for the toolbar and menus.

##  Dependencies

* PyQt5
* OpenCV (cv2)
* PyTorch
* NumPy
* SciPy
* Other dependencies specific to the grasp detection models (e.g., TensorFlow, if used)

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/venk221/grasp_annotations.git>
    cd <grasp_annotations>
    ```
2.  **Install dependencies:**
    ```bash
    pip install PyQt5 opencv-python torch numpy scipy
    # Install other model-specific dependencies as needed (e.g., tensorflow)
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```

##  Usage

1.  **Open Images/Directory:** Use the "Open Images" or "Open Dir" buttons to load the images you want to label.
2.  **Open Project (Optional):** If you have a previously saved project, use "Open Project" to load it.
3.  **Run Grasp Inference:** Click the "Grasp Inference" button (or use the Ctrl+G shortcut) to run the loaded grasp detection model on the current image. The predicted grasps will be displayed as rectangles.
4.  **Create/Edit Labels:**
    * Use the "Create Mode" and "Edit Mode" buttons to switch between adding new grasp labels and modifying existing ones.
    * Click and drag on the image to create rectangular grasp annotations.
    * Adjust the position, size, and orientation of labels as needed.
5.  **Manage Labels:** Use the grasp list panel to:
    * Select labels.
    * Toggle the visibility of labels.
    * Change the order of labels.
6.  **Navigate Images:** Use the "Prev Image" and "Next Image" buttons (or the A and D keys) to move between images.
7.  **Save Project:** Use the "Save Project" button to save your annotations. You can also change the output directory for saving.

##  Troubleshooting

* **Model Loading Errors:**
    * Ensure that the model paths in `inference.py` are correct.
    * Verify that all necessary model-specific dependencies are installed.
    * Check that the model files are not corrupted.
* **Display Issues:**
    * Make sure your graphics drivers are up to date.
    * If you encounter coordinate misalignment, carefully check the coordinate systems used by OpenCV, PyTorch, and PyQt5.
* **Performance:**
    * Grasp detection can be computationally intensive.  Consider using a GPU if possible.

##  Acknowledgments

*Cazacu, H. (2022). *LabelGrasp* [GitHub repository]. https://github.com/hhcaz/LabelGrasp
# grasp_labelling_tool

import sys
import numpy as np
import pydicom
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog,
                           QComboBox, QGroupBox, QProgressBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import tempfile
from vtk.util import numpy_support
vtk.vtkObject.GlobalWarningDisplayOff()

class VolumeRenderer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_modality = None
        self.temp_dir = None
        self.volume_data_type = "Unknown"
        self.hounsfield_mode = False
        self.presets = self.create_presets()
        self.preset_ranges = {
            'CT-Bone': [-100, 1500], 'CT-Soft': [-160, 240], 'CT-Lung': [-1050, 100],
            'CT-Angio': [50, 600], 'MR-T1': [0, 1500], 'DSA-Vessel': [500, 4000],
            'XA-Vessel': [100, 3000], 'Radiant-Vessel': [100, 500], 'Radiant-Surface': [150, 600]
        }
        self.initUI()

    def create_presets(self):
        presets = {}
        # CT Bone Preset
        presets['CT-Bone'] = {
            'name': 'CT-Bone', 'color_points': [(-100, 0.86, 0.86, 0.86), (300, 0.89, 0.85, 0.78), 
                                                (800, 0.94, 0.91, 0.82), (1500, 1.0, 0.99, 0.95)],
            'opacity_points': [(-100, 0.0), (150, 0.0), (350, 0.8), (1500, 1.0)],
            'ambient': 0.2, 'diffuse': 0.9, 'specular': 0.3, 'specular_power': 15
        }
        # Add other presets similarly (omitted for brevity, same as original)
        return presets

    def initUI(self):
        self.setWindowTitle('Advanced DICOM Volume Renderer')
        self.setGeometry(100, 100, 1400, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget, stretch=4)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        layout.addWidget(control_panel, stretch=1)

        self.addControls(control_layout)
        self.initVTK()

    def initVTK(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        self.volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.ShadeOn()
        self.volume_property.SetAmbient(0.2)
        self.volume_property.SetDiffuse(0.9)
        self.volume_property.SetSpecular(0.5)
        self.volume_property.SetSpecularPower(25)
        
        self.color_transfer = vtk.vtkColorTransferFunction()
        self.opacity_transfer = vtk.vtkPiecewiseFunction()
        self.volume_property.SetColor(self.color_transfer)
        self.volume_property.SetScalarOpacity(self.opacity_transfer)
        
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        self.interactor.Initialize()
        self.setupCamera()

    def setupCamera(self):
        camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(camera_style)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, -1, 0)
        camera.SetViewUp(0, 0, 1)
        light = vtk.vtkLight()
        light.SetColor(1, 1, 1)
        light.SetIntensity(1.0)
        light.SetPosition(200, 200, 200)
        self.renderer.AddLight(light)

    def addControls(self, layout):
        load_group = QGroupBox("Load Controls")
        load_layout = QVBoxLayout()
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(['CT', 'CTA', 'DSA', 'XA', 'MR'])
        self.modality_combo.currentTextChanged.connect(self.onModalityChanged)
        load_btn = QPushButton('Load DICOM Directory')
        load_btn.clicked.connect(self.loadDICOMDirectory)
        load_layout.addWidget(QLabel('Select Modality:'))
        load_layout.addWidget(self.modality_combo)
        load_layout.addWidget(load_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        preset_group = QGroupBox("Rendering Presets")
        preset_layout = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.applyPreset)
        self.hounsfield_checkbox = QCheckBox("Use Hounsfield Units (CT)")
        self.hounsfield_checkbox.toggled.connect(self.toggleHounsfieldMode)
        preset_layout.addWidget(QLabel('Rendering Preset:'))
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.hounsfield_checkbox)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        preset_group.setHidden(True)

        render_group = QGroupBox("Rendering Controls")
        render_layout = QVBoxLayout()
        
        # Existing sliders
        self.opacity_slider = QSlider(Qt.Horizontal, minimum=0, maximum=300, value=100)
        self.opacity_slider.valueChanged.connect(self.updateTransferFunction)
        render_layout.addWidget(QLabel('Opacity Threshold:'))
        render_layout.addWidget(self.opacity_slider)
        
        self.lower_threshold_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=20)
        self.lower_threshold_slider.valueChanged.connect(self.updateTransferFunction)
        render_layout.addWidget(QLabel('Lower Threshold:'))
        render_layout.addWidget(self.lower_threshold_slider)
        
        # New sliders for Smoothing and Detail from save3DModel
        # Smoothing iterations slider
        smoothing_label = QLabel("Smoothing Iterations:")
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(5)
        self.smoothing_slider.setMaximum(30)
        self.smoothing_slider.setValue(25)  # Default value
        self.smoothing_value_label = QLabel(f"Current: {self.smoothing_slider.value()}")
        
        def update_smoothing_label():
            self.smoothing_value_label.setText(f"Current: {self.smoothing_slider.value()}")
        self.smoothing_slider.valueChanged.connect(update_smoothing_label)
        
        render_layout.addWidget(smoothing_label)
        render_layout.addWidget(self.smoothing_slider)
        render_layout.addWidget(self.smoothing_value_label)
        
        # Detail level (triangle reduction) slider
        detail_label = QLabel("Detail Level (lower = more triangles):")
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setMinimum(10)
        self.detail_slider.setMaximum(90)
        self.detail_slider.setValue(20)  # Default 50% reduction
        self.detail_value_label = QLabel(f"Current: {self.detail_slider.value()}% reduction")
        
        def update_detail_label():
            self.detail_value_label.setText(f"Current: {self.detail_slider.value()}% reduction")
        self.detail_slider.valueChanged.connect(update_detail_label)
        
        render_layout.addWidget(detail_label)
        render_layout.addWidget(self.detail_slider)
        render_layout.addWidget(self.detail_value_label)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)

        save_group = QGroupBox("Export Controls")
        save_layout = QVBoxLayout()
        save_3d_btn = QPushButton('Save 3D Model (STL)')
        save_3d_btn.clicked.connect(self.save3DModel)
        screenshot_btn = QPushButton('Take Screenshot (PNG)')
        screenshot_btn.clicked.connect(self.takeScreenshot)
        save_layout.addWidget(save_3d_btn)
        save_layout.addWidget(screenshot_btn)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

    def loadDICOMDirectory(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder_path:
            return
        
        # Scan the folder for DICOM files
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f))]
        dicom_files = []
        
        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, 'SOPClassUID'):  # Basic check for DICOM
                    dicom_files.append(f)
            except Exception as e:
                print(f"Skipping non-DICOM file {f}: {e}")
        
        if not dicom_files:
            QMessageBox.warning(self, "Warning", "No valid DICOM files found in the directory")
            return
        
        # Check if any file is a multi-frame (video) DICOM
        is_video = False
        video_file = None
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                    is_video = True
                    video_file = f
                    break
            except Exception as e:
                print(f"Error checking frames in {f}: {e}")
        
        if is_video and video_file:
            # Process as 3D volumetric video
            self.loading_thread = VideoLoadingThread(video_file)
        else:
            # Process as DICOM series
            self.loading_thread = LoadingThread(folder_path)
        
        self.loading_thread.progress_signal.connect(self.progress_bar.setValue)
        self.loading_thread.result_signal.connect(self.processDICOMData)
        self.loading_thread.error_signal.connect(self.showErrorMessage)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.loading_thread.start()

    def processDICOMData(self, result):
        dicom_data, pixel_data, spacing = result
        self.progress_bar.setVisible(False)
        if pixel_data is None or len(pixel_data) == 0:
            QMessageBox.warning(self, "Warning", "No valid DICOM data found")
            return
        
        # Check for irregular slice intervals before processing
        if not isinstance(dicom_data, list) or len(dicom_data) == 0:
            QMessageBox.warning(self, "Warning", "Invalid DICOM data format")
            return
        
        # Verify slice regularity
        if len(dicom_data) > 1:  # Only check for series (not videos)
            slice_locations = []
            slice_thicknesses = []
            for ds in dicom_data:
                if hasattr(ds, 'SliceLocation'):
                    slice_locations.append(float(ds.SliceLocation))
                if hasattr(ds, 'SliceThickness'):
                    slice_thicknesses.append(float(ds.SliceThickness))
            
            if slice_locations and len(slice_locations) > 1:
                # Check if slice locations are sequential and regular
                slice_locations.sort()
                intervals = [slice_locations[i+1] - slice_locations[i] for i in range(len(slice_locations)-1)]
                if intervals:
                    mean_interval = np.mean(intervals)
                    max_deviation = max(abs(interval - mean_interval) for interval in intervals)
                    if max_deviation > 0.1 * mean_interval:  # Threshold for irregularity (10% deviation)
                        response = QMessageBox.warning(
                            self, 
                            "Warning", 
                            "The slice interval is not regular. Distortion in presentation and measurements may be present.\n"
                            "Do you want to continue loading this data?",
                            QMessageBox.Yes | QMessageBox.No, 
                            QMessageBox.No
                        )
                        if response == QMessageBox.No:
                            return
        
        self.volume_data_type = self.detectModality(dicom_data)
        index = self.modality_combo.findText(self.volume_data_type)
        if index >= 0:
            self.modality_combo.setCurrentIndex(index)
        
        if self.volume_data_type in ['CT', 'CTA']:
            self.hounsfield_mode = True
            self.hounsfield_checkbox.setChecked(True)
        else:
            self.hounsfield_mode = False
            self.hounsfield_checkbox.setChecked(False)

        depth, height, width = pixel_data.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(0, 0, 0)
        
        if pixel_data.min() >= 0 and pixel_data.max() <= 255:
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        else:
            vtk_image.AllocateScalars(vtk.VTK_SHORT, 1)
        vtk_array = numpy_support.numpy_to_vtk(pixel_data.ravel())
        vtk_image.GetPointData().GetScalars().DeepCopy(vtk_array)
        
        self.volume_mapper.SetInputData(vtk_image)
        self.volume.SetMapper(self.volume_mapper)
        self.renderer.ResetCamera()
        self.applyPreset(self.preset_combo.currentText())
        self.vtk_widget.GetRenderWindow().Render()

    def detectModality(self, dicom_data):
        if hasattr(dicom_data[0], 'Modality'):
            modality = dicom_data[0].Modality
            if modality == 'CT' and hasattr(dicom_data[0], 'SeriesDescription'):
                series_desc = dicom_data[0].SeriesDescription.lower()
                if 'angio' in series_desc or 'cta' in series_desc or 'contrast' in series_desc:
                    return 'CTA'
            return modality.upper()
        return 'CT'

    def onModalityChanged(self, modality):
        self.current_modality = modality
        if modality == 'CT':
            self.preset_combo.setCurrentText('CT-Bone')
        elif modality == 'CTA':
            self.preset_combo.setCurrentText('CT-Angio')
        elif modality == 'DSA':
            self.preset_combo.setCurrentText('DSA-Vessel')
        elif modality == 'XA':
            self.preset_combo.setCurrentText('XA-Vessel')
        elif modality == 'MR':
            self.preset_combo.setCurrentText('MR-T1')
        if self.volume_mapper.GetInput() is not None:
            self.applyPreset(self.preset_combo.currentText())

    def applyPreset(self, preset_name):
        if not preset_name in self.presets or not self.volume_mapper.GetInput():
            return
        preset = self.presets[preset_name]
        scalar_range = self.volume_mapper.GetInput().GetScalarRange()
        
        self.color_transfer.RemoveAllPoints()
        self.opacity_transfer.RemoveAllPoints()
        
        if self.hounsfield_mode and self.current_modality in ['CT', 'CTA']:
            for point in preset['color_points']:
                self.color_transfer.AddRGBPoint(point[0], point[1], point[2], point[3])
            for point in preset['opacity_points']:
                self.opacity_transfer.AddPoint(point[0], point[1])
        else:
            preset_min, preset_max = self.preset_ranges[preset_name]
            range_min, range_max = scalar_range
            for point in preset['color_points']:
                value = range_min + (point[0] - preset_min) * (range_max - range_min) / (preset_max - preset_min)
                self.color_transfer.AddRGBPoint(value, point[1], point[2], point[3])
            for point in preset['opacity_points']:
                value = range_min + (point[0] - preset_min) * (range_max - range_min) / (preset_max - preset_min)
                self.opacity_transfer.AddPoint(value, point[1])
        
        self.volume_property.SetAmbient(preset['ambient'])
        self.volume_property.SetDiffuse(preset['diffuse'])
        self.volume_property.SetSpecular(preset['specular'])
        self.volume_property.SetSpecularPower(preset['specular_power'])
        self.vtk_widget.GetRenderWindow().Render()

    def updateTransferFunction(self):
        if not self.volume_mapper.GetInput():
            return
        scalar_range = self.volume_mapper.GetInput().GetScalarRange()
        threshold = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * (self.lower_threshold_slider.value() / 100.0)
        opacity = self.opacity_slider.value() / 300.0
        self.opacity_transfer.RemoveAllPoints()
        self.opacity_transfer.AddPoint(scalar_range[0], 0.0)
        self.opacity_transfer.AddPoint(threshold, 0.0)
        self.opacity_transfer.AddPoint(threshold + 1, opacity)
        self.opacity_transfer.AddPoint(scalar_range[1], opacity)
        self.vtk_widget.GetRenderWindow().Render()

    def save3DModel(self):
        if not hasattr(self, 'volume_mapper') or self.volume_mapper.GetInput() is None:
            QMessageBox.warning(self, "Warning", "No volume data loaded")
            return
        
        try:
            # Get export settings from UI sliders directly
            threshold_percent = self.lower_threshold_slider.value() / 100.0  # Use existing lower threshold
            # print("threshold value: ", self.lower_threshold_slider.value())
            # print("threshold %: ", threshold_percent)
            smoothing_iterations = self.smoothing_slider.value()
            vessel_preservation = 0.7  # Default value (you can add a slider later if needed)
            detail_reduction = self.detail_slider.value() / 100.0
            
            # Create progress dialog
            progress = QProgressBar(self)
            progress.setMinimum(0)
            progress.setMaximum(100)
            progress.setValue(0)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Generating 3D Model")
            progress.show()
            
            def update_progress(value):
                progress.setValue(value)
                QApplication.processEvents()
            
            # Step 1: Create the initial isosurface
            update_progress(10)
            scalar_range = self.volume_mapper.GetInput().GetScalarRange()
            threshold = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * threshold_percent
            
            gaussian = vtk.vtkImageGaussianSmooth()
            gaussian.SetInputData(self.volume_mapper.GetInput())
            gaussian.SetStandardDeviation(1.0)
            gaussian.SetRadiusFactor(1.5)
            gaussian.Update()
            
            update_progress(20)
            
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(gaussian.GetOutput())
            contour.SetValue(0, threshold)
            contour.ComputeNormalsOn()
            contour.ComputeGradientsOn()
            contour.Update()
            
            update_progress(40)
            
            # Step 2: Apply topology-preserving smoothing
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetNumberOfIterations(smoothing_iterations)
            smoother.SetFeatureEdgeSmoothing(0)
            smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(0.1)
            smoother.SetNonManifoldSmoothing(1)
            smoother.SetNormalizeCoordinates(1)
            smoother.BoundarySmoothingOff()
            smoother.Update()
            
            update_progress(60)
            
            # Step 3: Enhance small vessels using connectivity filter
            connectivity = vtk.vtkPolyDataConnectivityFilter()
            connectivity.SetInputConnection(smoother.GetOutputPort())
            connectivity.SetExtractionModeToLargestRegion()
            connectivity.Update()
            
            update_progress(70)
            
            # Step 4: Intelligently reduce triangles while preserving vessel structure
            decimator = vtk.vtkDecimatePro()
            decimator.SetInputConnection(connectivity.GetOutputPort())
            decimator.SetTargetReduction(detail_reduction)
            decimator.PreserveTopologyOn()
            decimator.SetFeatureAngle(75.0 * (1.0 + vessel_preservation * 0.5))
            decimator.SplittingOff()
            decimator.BoundaryVertexDeletionOff()
            decimator.Update()
            
            update_progress(80)
            
            # Step 5: Clean and repair the mesh
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(decimator.GetOutputPort())
            cleaner.ConvertLinesToPointsOff()
            cleaner.ConvertPolysToLinesOff()
            cleaner.ConvertStripsToPolysOff()
            cleaner.Update()
            
            # Step 6: Correct orientation for anterior view
            transform = vtk.vtkTransform()
            # Assuming DICOM data is typically in axial orientation (Z up, Y anterior, X right)
            # Rotate to ensure anterior view (Y-axis points forward, X right, Z up)
            transform.RotateX(180)  # Flip to correct anterior-posterior orientation if inverted
            transform.RotateY(180)
            transform.RotateZ(180)
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(cleaner.GetOutputPort())
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            
            # Compute normals after transformation to ensure correct lighting
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(transform_filter.GetOutputPort())
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()
            
            update_progress(90)
            
            # Get save file path
            default_name = "output.stl"
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Save 3D Model",
                default_name,
                "STL Files (*.stl);;OBJ Files (*.obj);;All Files (*)"
            )
            
            if filepath:
                if not (filepath.lower().endswith('.stl') or filepath.lower().endswith('.obj')):
                    filepath += '.stl'
                
                if filepath.lower().endswith('.obj'):
                    writer = vtk.vtkOBJWriter()
                else:
                    writer = vtk.vtkSTLWriter()
                
                writer.SetFileName(filepath)
                writer.SetInputConnection(normals.GetOutputPort())
                writer.Write()
                
                update_progress(100)
                progress.close()
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    "3D model saved successfully!\n\n"
                    f"- Threshold: {self.lower_threshold_slider.value()}%\n"
                    f"- Smoothing: {smoothing_iterations} iterations\n"
                    f"- Triangle reduction: {self.detail_slider.value()}% (default vessel preservation)"
                )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save 3D model: {str(e)}")
            print(f"Error saving 3D model: {str(e)}")
            if progress:
                progress.close()

    # def save3DModel(self):
    #     if not self.volume_mapper.GetInput():
    #         QMessageBox.warning(self, "Warning", "No volume data loaded")
    #         return
        
    #     try:
    #         # Create the initial isosurface
    #         contour = vtk.vtkContourFilter()
    #         contour.SetInputData(self.volume_mapper.GetInput())
            
    #         # Get scalar range of the volume data
    #         scalar_range = self.volume_mapper.GetInput().GetScalarRange()
            
    #         # Get current transfer function points to determine proper threshold
    #         opacity_points = []
    #         for i in range(self.opacity_transfer.GetSize()):
    #             node = [0.0, 0.0, 0.0, 0.0]  # (value, opacity, midpoint, sharpness)
    #             self.opacity_transfer.GetNodeValue(i, node)
    #             opacity_points.append((node[0], node[1]))  # (scalar value, opacity)

    #         # Sort by value (first element of each tuple)
    #         opacity_points.sort(key=lambda x: x[0])
            
    #         # Find the first point where opacity becomes significant (transition point)
    #         threshold = None
    #         for i in range(1, len(opacity_points)):
    #             if opacity_points[i-1][1] < 0.01 and opacity_points[i][1] > 0.01:
    #                 # This is the transition point - use this as our threshold
    #                 threshold = opacity_points[i][0]
    #                 break
            
    #         # Fallback if we couldn't determine a threshold from the opacity function
    #         if threshold is None:
    #             # Use slider values directly from UI
    #             lower_threshold_pct = self.lower_threshold_slider.value() / 100.0  # 0 to 1 scale
    #             opacity_threshold_pct = self.opacity_slider.value() / 300.0       # 0 to 1 scale
                
    #             # Determine threshold based on slider values and current rendering settings
    #             if self.hounsfield_mode and self.current_modality in ['CT', 'CTA']:
    #                 # Use Hounsfield units from the current preset range
    #                 current_preset = self.preset_combo.currentText()
    #                 if current_preset in self.preset_ranges:
    #                     preset_min, preset_max = self.preset_ranges[current_preset]
    #                     # Calculate threshold in Hounsfield units
    #                     threshold = preset_min + lower_threshold_pct * (preset_max - preset_min)
    #                 else:
    #                     # Default to CT-Bone range if preset not found
    #                     threshold = 130 + lower_threshold_pct * 1370  # From -100 to 1500 (bone range)
    #             else:
    #                 # Use scalar range for non-CT/CTA modalities
    #                 threshold = scalar_range[0] + lower_threshold_pct * (scalar_range[1] - scalar_range[0])
            
    #         print(f"Using threshold value: {threshold}")
    #         contour.SetValue(0, threshold)
    #         contour.Update()
            
    #         # Ensure we extract only connected regions and filter out small noise
    #         connectivity = vtk.vtkPolyDataConnectivityFilter()
    #         connectivity.SetInputConnection(contour.GetOutputPort())
    #         connectivity.SetExtractionModeToLargestRegion()
    #         connectivity.Update()

    #         # Clean up the polydata to remove small artifacts
    #         clean_polydata = vtk.vtkCleanPolyData()
    #         clean_polydata.SetInputConnection(connectivity.GetOutputPort())
    #         clean_polydata.SetTolerance(0.001)
    #         clean_polydata.Update()

    #         # Smooth the surface to match the rendered quality
    #         smoother = vtk.vtkSmoothPolyDataFilter()
    #         smoother.SetInputConnection(clean_polydata.GetOutputPort())
    #         smoother.SetNumberOfIterations(30)
    #         smoother.SetRelaxationFactor(0.15)
    #         smoother.FeatureEdgeSmoothingOn()
    #         smoother.Update()

    #         # Reduce triangles while preserving topology
    #         decimator = vtk.vtkDecimatePro()
    #         decimator.SetInputConnection(smoother.GetOutputPort())
    #         decimator.SetTargetReduction(0.3)
    #         decimator.PreserveTopologyOn()
    #         decimator.Update()

    #         # Save the 3D model
    #         filepath, _ = QFileDialog.getSaveFileName(self, "Save 3D Model", "output.stl", "STL Files (*.stl);;All Files (*)")
    #         if filepath:
    #             if not filepath.lower().endswith('.stl'):
    #                 filepath += '.stl'
    #             stl_writer = vtk.vtkSTLWriter()
    #             stl_writer.SetFileName(filepath)
    #             stl_writer.SetInputConnection(decimator.GetOutputPort())
    #             stl_writer.Write()
    #             QMessageBox.information(self, "Success", f"3D model saved successfully to {filepath}!")
                
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to save 3D model: {str(e)}")
    #         print(f"Error saving 3D model: {str(e)}")

    def takeScreenshot(self):
        if not self.volume_mapper.GetInput():
            QMessageBox.warning(self, "Warning", "No volume data loaded")
            return
        try:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "screenshot.png", "PNG Files (*.png);;All Files (*)")
            if filepath:
                if not filepath.lower().endswith('.png'):
                    filepath += '.png'
                window_to_image = vtk.vtkWindowToImageFilter()
                window_to_image.SetInput(self.vtk_widget.GetRenderWindow())
                window_to_image.SetInputBufferTypeToRGBA()
                window_to_image.ReadFrontBufferOff()
                window_to_image.Update()
                png_writer = vtk.vtkPNGWriter()
                png_writer.SetFileName(filepath)
                png_writer.SetInputConnection(window_to_image.GetOutputPort())
                png_writer.Write()
                QMessageBox.information(self, "Success", "Screenshot saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save screenshot: {str(e)}")

    def toggleHounsfieldMode(self, enabled):
        self.hounsfield_mode = enabled
        if self.volume_mapper.GetInput():
            self.applyPreset(self.preset_combo.currentText())

    def showErrorMessage(self, message):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", message)

class LoadingThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(tuple)
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
            dicom_files = [f for f in files if pydicom.dcmread(f, force=True).SOPClassUID]
            if not dicom_files:
                self.error_signal.emit("No valid DICOM files found")
                return
            
            dicom_data = [pydicom.dcmread(f, force=True) for f in dicom_files]
            dicom_data.sort(key=lambda x: x.get('InstanceNumber', 0))
            
            # Check for irregular slice intervals
            slice_locations = []
            slice_thicknesses = []
            for ds in dicom_data:
                if hasattr(ds, 'SliceLocation'):
                    slice_locations.append(float(ds.SliceLocation))
                if hasattr(ds, 'SliceThickness'):
                    slice_thicknesses.append(float(ds.SliceThickness))
            
            if slice_locations and len(slice_locations) > 1:
                slice_locations.sort()
                intervals = [slice_locations[i+1] - slice_locations[i] for i in range(len(slice_locations)-1)]
                if intervals:
                    mean_interval = np.mean(intervals)
                    max_deviation = max(abs(interval - mean_interval) for interval in intervals)
                    if max_deviation > 0.1 * mean_interval:  # Threshold for irregularity (10% deviation)
                        self.error_signal.emit(
                            "The slice interval is not regular. Distortion in presentation and measurements may be present.\n"
                            "Skipping this dataset to prevent application crash."
                        )
                        return
            
            spacing = [float(dicom_data[0].get('PixelSpacing', [1.0, 1.0])[0]), 
                      float(dicom_data[0].get('PixelSpacing', [1.0, 1.0])[1]), 
                      float(dicom_data[0].get('SliceThickness', 1.0))]
            
            pixel_arrays = []
            for i, ds in enumerate(dicom_data):
                pixel_array = ds.pixel_array
                if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                pixel_arrays.append(pixel_array)
                self.progress_signal.emit(int(100 * (i+1) / len(dicom_data)))
            
            pixel_data = np.stack(pixel_arrays)
            self.result_signal.emit((dicom_data, pixel_data, spacing))
        except Exception as e:
            self.error_signal.emit(f"Error loading DICOM data: {str(e)}")

class VideoLoadingThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(tuple)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            ds = pydicom.dcmread(self.file_path, force=True)
            if 'NumberOfFrames' not in ds or ds.NumberOfFrames <= 1:
                self.error_signal.emit("Not a multi-frame DICOM video file")
                return
            
            pixel_array = ds.pixel_array  # shape: (frames, height, width)
            dicom_data = [ds] * ds.NumberOfFrames  # Duplicate metadata for compatibility
            
            spacing = [float(ds.get('PixelSpacing', [1.0, 1.0])[0]), 
                      float(ds.get('PixelSpacing', [1.0, 1.0])[1]), 
                      float(ds.get('SliceThickness', 1.0))]
            
            if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            for i in range(ds.NumberOfFrames):
                self.progress_signal.emit(int(100 * (i+1) / ds.NumberOfFrames))
            
            self.result_signal.emit((dicom_data, pixel_array, spacing))
        except Exception as e:
            self.error_signal.emit(f"Error loading DICOM video: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolumeRenderer()
    window.show()
    sys.exit(app.exec_())
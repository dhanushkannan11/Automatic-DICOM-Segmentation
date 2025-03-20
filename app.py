import sys
import numpy as np
import pydicom
import os
import time
import threading
from vtk.util import numpy_support
import vtk
import gc  # For garbage collection

# Suppress VTK output window
vtk.vtkObject.GlobalWarningDisplayOff()

# Define a custom exception for timeout
class TimeoutException(Exception):
    pass

# Define a custom exception for invalid STL size
class InvalidSTLSizeException(Exception):
    pass

def timeout_function(func, args, timeout):
    """
    Run a function with a timeout using threading.
    If the function exceeds the timeout, raise TimeoutException.
    """
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException("Processing time exceeded 3 minutes!")
    if exception[0] is not None:
        raise exception[0]
    return result[0]

def process_dicom_series(folder_path):
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
        print("Error: No valid DICOM files found in the directory")
        sys.exit(1)
    
    # Sort DICOM files by InstanceNumber
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
                print(
                    "Warning: The slice interval is not regular. Distortion in presentation and measurements may be present."
                )
                # Optionally, you can exit here or continue with a warning
                # For now, we’ll continue but you can uncomment the line below to exit
                # sys.exit(1)
    
    # Get spacing from the first DICOM file
    spacing = [float(dicom_data[0].get('PixelSpacing', [1.0, 1.0])[0]), 
               float(dicom_data[0].get('PixelSpacing', [1.0, 1.0])[1]), 
               float(dicom_data[0].get('SliceThickness', 1.0))]
    
    # Load and stack pixel arrays efficiently
    pixel_arrays = []
    for i, ds in enumerate(dicom_data):
        pixel_array = ds.pixel_array
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        pixel_arrays.append(pixel_array)
        print(f"Processing slice {i+1}/{len(dicom_data)}... {int(100 * (i+1) / len(dicom_data))}%")
    
    # Use memory-efficient stacking
    pixel_data = np.stack(pixel_arrays, out=np.zeros((len(pixel_arrays), pixel_arrays[0].shape[0], pixel_arrays[0].shape[1]), dtype=pixel_arrays[0].dtype))
    return dicom_data, pixel_data, spacing

def process_dicom_video(file_path):
    try:
        ds = pydicom.dcmread(file_path, force=True)
        if 'NumberOfFrames' not in ds or ds.NumberOfFrames <= 1:
            print("Error: Not a multi-frame DICOM video file")
            sys.exit(1)
        
        pixel_array = ds.pixel_array  # shape: (frames, height, width)
        dicom_data = [ds] * ds.NumberOfFrames  # Duplicate metadata for compatibility
        
        spacing = [float(ds.get('PixelSpacing', [1.0, 1.0])[0]), 
                  float(ds.get('PixelSpacing', [1.0, 1.0])[1]), 
                  float(ds.get('SliceThickness', 1.0))]
        
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        for i in range(ds.NumberOfFrames):
            print(f"Processing frame {i+1}/{ds.NumberOfFrames}... {int(100 * (i+1) / ds.NumberOfFrames)}%")
        
        return dicom_data, pixel_array, spacing
    except Exception as e:
        print(f"Error loading DICOM video: {str(e)}")
        sys.exit(1)

def otsu_thresholding(hist, bins):
    """
    Apply Otsu's thresholding method to find an optimal threshold.
    Returns the threshold that maximizes inter-class variance.
    """
    hist = hist.astype(float)
    hist = hist / hist.sum()  # Normalize histogram
    total = hist.sum()
    
    sumT = np.arange(len(hist)).dot(hist)  # Sum of intensities
    sumB = 0.0
    wB = 0.0
    wF = 0.0
    
    max_var = -1
    threshold = 0
    
    for t in range(len(hist)):
        wB += hist[t]  # Weight of background
        if wB == 0:
            continue
        wF = total - wB  # Weight of foreground
        if wF == 0:
            break
        
        sumB += t * hist[t]  # Sum of background intensities
        mB = sumB / wB  # Mean of background
        mF = (sumT - sumB) / wF  # Mean of foreground
        
        # Inter-class variance
        var = wB * wF * (mB - mF) ** 2
        
        if var > max_var:
            max_var = var
            threshold = t
    
    return bins[threshold]

def generate_3d_model(dicom_data, pixel_data, spacing, output_path):
    print("Starting 3D model generation...")
    
    def core_generate():
        # Create VTK image data
        depth, height, width = pixel_data.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])  # X, Y, Z
        vtk_image.SetOrigin(0, 0, 0)
        
        # Handle scalar allocation based on data range
        if pixel_data.min() >= 0 and pixel_data.max() <= 255:
            vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        else:
            vtk_image.AllocateScalars(vtk.VTK_SHORT, 1)
        
        vtk_array = numpy_support.numpy_to_vtk(pixel_data.ravel())
        vtk_image.GetPointData().GetScalars().DeepCopy(vtk_array)
        
        # Create volume mapper with specified ray casting parameters
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)
        
        # Check VTK version and set interpolation mode appropriately
        try:
            # Try newer VTK versions (e.g., VTK 9.x) with SetInterpolate
            volume_mapper.InterpolateOn()  # Enable linear interpolation
        except AttributeError:
            try:
                # Try older VTK versions (e.g., VTK 8.x) with SetInterpolate
                volume_mapper.SetInterpolate(1)  # Enable linear interpolation
            except AttributeError:
                print("Warning: Linear interpolation not supported in this VTK version. Using default interpolation.")
                # If neither method works, proceed with default interpolation (likely linear in most VTK versions)
        
        # Set other ray casting parameters
        volume_mapper.SetBlendModeToComposite()  # Composite blending mode
        volume_mapper.SetAutoAdjustSampleDistances(True)  # Auto adjust sample distances
        volume_mapper.SetSampleDistance(1.0)  # Sample distance of 1.0
        
        # Create volume property with specified parameters
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()  # Linear interpolation
        volume_property.ShadeOff()  # Shading disabled
        volume_property.SetAmbient(0.0)  # Ambient lighting
        volume_property.SetDiffuse(1.0)  # Diffuse lighting
        volume_property.SetSpecular(0.0)  # No specular reflection
        volume_property.SetSpecularPower(1.0)  # Specular power
        
        # Set up color and opacity transfers (using CT-Bone preset as default, tailored for vessels)
        color_transfer = vtk.vtkColorTransferFunction()
        opacity_transfer = vtk.vtkPiecewiseFunction()
        
        # Vessel-specific color and opacity points (optimized for CTA/DSA vessel visualization)
        color_points = [(-100, 0.0, 0.0, 0.0),  # Background (black)
                        (50, 0.0, 0.0, 0.5),    # Low intensity (semi-transparent)
                        (200, 1.0, 0.0, 0.0),   # Vessel start (red)
                        (600, 1.0, 0.5, 0.0),   # Vessel highlight (orange)
                        (4000, 1.0, 1.0, 1.0)]  # High intensity (white)
        opacity_points = [(-100, 0.0),          # Background (transparent)
                          (50, 0.0),            # Low intensity (transparent)
                          (200, 0.1),           # Vessel start (slightly visible)
                          (600, 0.8),           # Vessel highlight (mostly opaque)
                          (4000, 1.0)]          # High intensity (fully opaque)
        
        scalar_range = vtk_image.GetScalarRange()
        threshold = scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.2

        for point in color_points:
            value = point[0]
            color_transfer.AddRGBPoint(value, point[1], point[2], point[3])
        for point in opacity_points:
            value = point[0]
            opacity_transfer.AddPoint(value, point[1])
        
        volume_property.SetColor(color_transfer)
        volume_property.SetScalarOpacity(opacity_transfer)
        
        # Create volume (not rendered, just for data processing)
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        # Automatically determine optimal threshold for vessels using Otsu’s method
        print("Computing optimal threshold for vessel generation...")
        flattened_data = pixel_data.ravel()
        hist, bins = np.histogram(flattened_data, bins=256, range=(flattened_data.min(), flattened_data.max()))
        optimal_threshold = otsu_thresholding(hist, bins[:-1])  # Use bin edges (excluding last bin)
        print(f"Optimal threshold for vessels: {optimal_threshold}")
        
        # Create and apply the 3D model generation pipeline
        print("Applying Gaussian smoothing...")
        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetInputData(vtk_image)
        gaussian.SetStandardDeviation(1.0)
        gaussian.SetRadiusFactor(1.5)
        gaussian.Update()
        
        print("Generating isosurface with Marching Cubes using optimal threshold...")
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(gaussian.GetOutput())
        contour.SetValue(0, threshold)  # Use Otsu’s threshold for vessel structures
        contour.ComputeNormalsOn()
        contour.ComputeGradientsOn()
        contour.Update()
        
        print("Extracting largest region with connectivity filter...")
        # Use SetExtractionModeToLargestRegion as requested
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(contour.GetOutputPort())
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()
        
        print("Applying topology-preserving smoothing...")
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(connectivity.GetOutputPort())
        smoother.SetNumberOfIterations(17)  # Default value from your UI
        smoother.SetFeatureEdgeSmoothing(0)
        smoother.SetFeatureAngle(120.0)
        smoother.SetPassBand(0.1)
        smoother.SetNonManifoldSmoothing(1)
        smoother.SetNormalizeCoordinates(1)
        smoother.BoundarySmoothingOff()
        smoother.Update()
        
        print("Reducing triangles with decimation...")
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(0.4)  # Default 20% reduction from your UI
        decimator.PreserveTopologyOn()
        decimator.SetFeatureAngle(75.0 * 1.35)  # Default vessel preservation factor
        decimator.SplittingOff()
        decimator.BoundaryVertexDeletionOff()
        decimator.Update()
        
        print("Cleaning and repairing the mesh...")
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(decimator.GetOutputPort())
        cleaner.ConvertLinesToPointsOff()
        cleaner.ConvertPolysToLinesOff()
        cleaner.ConvertStripsToPolysOff()
        cleaner.Update()
        
        print("Correcting orientation for anterior view...")
        transform = vtk.vtkTransform()
        transform.RotateX(180)  # Flip to correct anterior-posterior orientation if inverted
        transform.RotateY(180)
        transform.RotateZ(180)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(cleaner.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        print("Computing normals for lighting...")
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(transform_filter.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()
        
        # Set up camera (not rendered, but for consistency with your settings)
        print("Setting up camera parameters...")
        camera = vtk.vtkCamera()
        camera.SetPosition(0, 0, 1)  # Position
        camera.SetFocalPoint(0, 0, 0)  # Focal point
        camera.SetViewUp(0, 1, 0)  # View up
        camera.ParallelProjectionOff()  # Perspective projection
        
        # Set up lighting (not directly used for STL, but for consistency)
        print("Setting up lighting parameters...")
        light = vtk.vtkLight()
        light.SetIntensity(1.0)  # Intensity
        light.SetPosition(0, 0, 1)  # Position
        light.SetColor(1.0, 1.0, 1.0)  # White light
        
        # Save as STL
        print("Saving 3D model as STL...")
        if output_path.lower().endswith('.stl'):
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(output_path)
            stl_writer.SetInputConnection(normals.GetOutputPort())
            stl_writer.Write()
        else:
            print("Error: Output file must have .stl extension")
            sys.exit(1)
        
        # Check the size of the generated STL file
        stl_size = os.path.getsize(output_path)
        print(f"Generated STL file size: {stl_size / 1024 / 1024:.2f} MB")
        
        # Define size limits (in bytes)
        MIN_SIZE = 2 * 1024 * 1024  # 2 MB
        MAX_SIZE = 160 * 1024 * 1024  # 160 MB
        
        if stl_size < MIN_SIZE:
            os.remove(output_path)
            raise InvalidSTLSizeException(f"STL file size ({stl_size / 1024 / 1024:.2f} MB) is less than the minimum allowed size (2 MB). File deleted.")
        elif stl_size > MAX_SIZE:
            os.remove(output_path)
            raise InvalidSTLSizeException(f"STL file size ({stl_size / 1024 / 1024:.2f} MB) exceeds the maximum allowed size (160 MB). File deleted.")
        
        print(f"3D model saved successfully to {output_path}")

    try:
        # Run the core generation with a 3-minute timeout
        timeout_function(core_generate, (), 180)  # 180 seconds = 3 minutes
    except TimeoutException as e:
        print(f"Error: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted incomplete STL file: {output_path}")
        sys.exit(1)
    except InvalidSTLSizeException as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during 3D model generation: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted incomplete STL file: {output_path}")
        sys.exit(1)
    finally:
        # Force garbage collection to free up memory
        gc.collect()

def main():
    # Check command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <dicom_directory> <output_stl_filename>")
        sys.exit(1)
    
    dicom_dir = sys.argv[1]
    output_stl = sys.argv[2]
    
    # Ensure DICOM directory exists
    if not os.path.isdir(dicom_dir):
        print(f"Error: Directory '{dicom_dir}' does not exist")
        sys.exit(1)
    
    # Ensure output path is valid and has .stl extension
    output_path = os.path.join("C:\SurgeonsLab\Pre-op planning\python-v2\DICOM-Viewer-with-Volume-Rendering-main\output\output", output_stl)
    if not output_path.lower().endswith('.stl'):
        output_path += '.stl'
    
    # Scan for DICOM files and detect if it's a video or series
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) 
             if os.path.isfile(os.path.join(dicom_dir, f))]
    dicom_files = []
    
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'SOPClassUID'):  # Basic check for DICOM
                dicom_files.append(f)
        except Exception as e:
            print(f"Skipping non-DICOM file {f}: {e}")
    
    if not dicom_files:
        print("Error: No valid DICOM files found in the directory")
        sys.exit(1)
    
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
    
    # Process the data
    if is_video and video_file:
        print("Processing DICOM video...")
        dicom_data, pixel_data, spacing = process_dicom_video(video_file)
    else:
        print("Processing DICOM series...")
        dicom_data, pixel_data, spacing = process_dicom_series(dicom_dir)
    
    # Generate and save the 3D model
    generate_3d_model(dicom_data, pixel_data, spacing, output_path)

if __name__ == "__main__":
    main()

# preprocess.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

class PreProcess:
    def __init__(self, image_path):
        """
        Initialize the PreProcess object by loading the image in grayscale.
        
        Parameters:
            image_path (str): Path to the input image.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise IOError(f"Could not load image at {image_path}")
        
        # Initialize intermediate images as None
        self.cropped_image = None
        self.bilateral_filtered = None
        self.contrastive_image = None
        self.sharpened = None
        self.edge_map = None
        self.morphed_image = None

    def crop_image(self, half_width=200, left_offset=0.2, right_offset=0.8):
        """
        Crop the image horizontally around its center.
        The vertical dimension remains unchanged.
        
        Parameters:
            half_width (int): A parameter controlling the crop width.
                              The crop is taken from (center - 0.2*half_width) 
                              to (center + 0.8*half_width).
        
        Returns:
            np.ndarray: The cropped image.
        """
        h, w = self.image.shape[:2]
        center_x = w // 2
        left = max(center_x - int(0.2 * half_width), 0)
        right = min(center_x + int(0.8 * half_width), w)
        self.cropped_image = self.image[:, left:right]
        return self.cropped_image

    def apply_bilateral_filter(self, d=3, sigmaColor=75, sigmaSpace=75):
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        
        Parameters:
            d (int): Diameter of each pixel neighborhood.
            sigmaColor (float): Filter sigma in the color space.
            sigmaSpace (float): Filter sigma in the coordinate space.
        
        Returns:
            np.ndarray: The bilateral filtered image.
        """
        if self.cropped_image is None:
            raise ValueError("Image not cropped yet. Call crop_image() first.")
        self.bilateral_filtered = cv2.bilateralFilter(self.cropped_image, d, sigmaColor, sigmaSpace)
        return self.bilateral_filtered

    def apply_clahe(self, clipLimit=5, tileGridSize=(3, 3)):
        """
        Enhance the image contrast using CLAHE.
        
        Parameters:
            clipLimit (float): Threshold for contrast limiting.
            tileGridSize (tuple): Size of grid for histogram equalization.
        
        Returns:
            np.ndarray: The CLAHE enhanced image.
        """
        if self.bilateral_filtered is None:
            raise ValueError("Bilateral filtering not applied yet. Call apply_bilateral_filter() first.")
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.contrastive_image = clahe.apply(self.bilateral_filtered)
        return self.contrastive_image

    def sharpen_image(self, weight=0.5):
        """
        Sharpen the image by blending it with its Laplacian edge map.
        
        Parameters:
            weight (float): Weight for the Laplacian edge map in the blending.
        
        Returns:
            np.ndarray: The sharpened image.
        """
        if self.contrastive_image is None:
            raise ValueError("Contrast-enhanced image not available. Call apply_clahe() first.")
        laplacian = cv2.Laplacian(self.contrastive_image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        self.sharpened = cv2.addWeighted(self.contrastive_image, 1.0, laplacian, weight, 0)
        return self.sharpened

    def apply_sobel_edge(self, scales=[1, 3, 5, 7]):
        """
        Apply multi-scale Sobel edge detection to enhance edge features.
        
        Parameters:
            scales (list): List of kernel sizes for the Sobel operator.
        
        Returns:
            np.ndarray: The edge map after applying Sobel filtering.
        """
        if self.sharpened is None:
            raise ValueError("Sharpened image not available. Call sharpen_image() first.")
        
        edge_map = np.zeros_like(self.sharpened, dtype=np.float32)
        for ksize in scales:
            # Compute gradients in x and y directions
            grad_x = cv2.Sobel(self.sharpened, cv2.CV_32F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(self.sharpened, cv2.CV_32F, 0, 1, ksize=ksize)
            # Compute the gradient magnitude
            magnitude = cv2.magnitude(grad_x, grad_y)
            # Update the edge map with the maximum magnitude
            edge_map = np.maximum(edge_map, magnitude)
        
        # Normalize to the range 0-255 and convert to 8-bit unsigned
        edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
        self.edge_map = edge_map.astype(np.uint8)
        return self.edge_map

    def apply_morph_skeletonize(self,kernel=(3,3)):
        image = self.edge_map.astype(np.uint8)
        # Ensure binary image
        _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Initialize skeleton
        skeleton = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel)

        while True:
            # Erosion and opening
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()

            if cv2.countNonZero(binary) == 0:
                break
        self.skeleton = skeleton
        return skeleton

    def show_image(self, image, title="Image"):
        """
        Display an image using matplotlib.
        
        Parameters:
            image (np.ndarray): Image to display.
            title (str): Title for the display window.
        """
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def overlay_fault_skeleton(self,original_image, skeleton):
        # Step 1: Convert original grayscale image to BGR
        image_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Step 2: Create a green mask for the thinned edges
        green_mask = np.zeros_like(image_bgr)
        green_mask[skeleton == 255] = [0, 255, 0]  # Green in BGR

        # Step 3: Overlay the green edges on the original image
        # Blend with some transparency (alpha blending)
        overlay = cv2.addWeighted(image_bgr, 0.8, green_mask, 0.8, 0)
        return overlay
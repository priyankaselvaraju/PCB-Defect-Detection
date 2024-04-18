import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from PIL import Image

def browse_image():
    global input_image_path
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    input_image_label.config(text=input_image_path)

def preprocess_image(image):
    # Convert the image to grayscale
    return image.convert('L')

def visualize_annotations():
    # Load the input image
    input_image = Image.open(input_image_path)

    # Create subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 6))

    # Display the original image
    ax1.imshow(input_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Preprocess the image (e.g., convert to grayscale)
    processed_image = preprocess_image(input_image)
    ax2.imshow(processed_image, cmap='gray')
    ax2.set_title('Processed Image')
    ax2.axis('off')

    # Display the annotated image
    ax3.imshow(input_image)
    ax3.set_title('Annotated Image')

    # Find the annotation file corresponding to the input image
    annotations_folder = r'I:\2023-2024\Python Projects\Pcb_defects_identification\PCB_DATASET\Annotations'

    # Find all XML files in the annotations folder and its subfolders
    xml_files = []
    for root, dirs, files in os.walk(annotations_folder):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))

    # Match the input image filename with the XML file names to find the corresponding annotation file
    image_filename = os.path.basename(input_image_path)
    matching_xml_files = [xml_file for xml_file in xml_files if os.path.splitext(os.path.basename(xml_file))[0] == os.path.splitext(image_filename)[0]]

    if not matching_xml_files:
        print(f"Annotation file not found for {image_filename}")
        return

    annotation_file_path = matching_xml_files[0]

    # Display the annotated image
    ax3.imshow(input_image)
    ax3.set_title('Output Image')

    # Load the annotations
    tree = ET.parse(annotation_file_path)
    root = tree.getroot()

    # Plot bounding boxes and defect types
    for annotation in root.iter('object'):
        xmin = int(annotation.find('bndbox/xmin').text)
        ymin = int(annotation.find('bndbox/ymin').text)
        xmax = int(annotation.find('bndbox/xmax').text)
        ymax = int(annotation.find('bndbox/ymax').text)
        defect_type = annotation.find('name').text

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)
        ax3.text(xmin, ymin, defect_type, fontsize=8, color='r', verticalalignment='top')

    ax3.axis('off')

    # Hide the empty subplots
    ax4.axis('off')
    ax5.axis('off')

    plt.tight_layout()
    plt.show()

# Create the main window
root = tk.Tk()
root.title("PCB Defect Annotation Visualizer")

# Input Image
input_image_label = tk.Label(root, text="Select input image:")
input_image_label.pack()
browse_image_button = tk.Button(root, text="Browse", command=browse_image)
browse_image_button.pack()

# Visualize Annotations
visualize_button = tk.Button(root, text="Visualize Annotations", command=visualize_annotations)
visualize_button.pack()

root.mainloop()

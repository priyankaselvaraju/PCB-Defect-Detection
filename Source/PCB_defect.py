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

def visualize_annotations():
    # Load the input image
    input_image = Image.open(input_image_path)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax1.imshow(input_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Find the annotation file corresponding to the input image
    annotations_folder = r'C:/Users/Priyanka/Desktop/Pcb_defects_identification/PCB_DATASET/Annotations'

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
        print(f"No defects found for {image_filename}")
        ax2.imshow(input_image)
        ax2.set_title('Output Image (No Defects)')
        ax2.axis('off')
    else:
        annotation_file_path = matching_xml_files[0]

        # Display the annotated image
        ax2.imshow(input_image)
        ax2.set_title('Output Image')

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
            ax2.add_patch(rect)
            ax2.text(xmin, ymin, defect_type, fontsize=8, color='r', verticalalignment='top')

        ax2.axis('off')

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

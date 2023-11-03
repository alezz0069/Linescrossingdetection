
from pdf2image import convert_from_path

def convert_pdf_to_png(pdf_path, output_path):
    # Convert the PDF to a list of PIL images
    images = convert_from_path(pdf_path)
    
    # Save the first page of the PDF as a PNG
    images[0].save(output_path, 'PNG') 

# If this script is run directly, it will convert the specified PDF to PNG
if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = '/content/10199-104-PID-180-F-20006_1.pdf'
    output_path = '/content/ex1cross.png'
    
    convert_pdf_to_png(pdf_path, output_path)

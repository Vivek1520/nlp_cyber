from PIL import Image, ImageDraw, ImageFont

# Create a blank image
width, height = 800, 400
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Define the text
text = """- Collect data
- Data Processing
- Model Development
- Training
- Testing
- LR, SVM, RF, XGBoost, Voting
- Model Evaluation
- Comparative analysis
- Feature Selection
- Chi Square
- RFE
- Extract Important Features"""

# Load a font
try:
    font = ImageFont.truetype("arial.ttf", size=20)
except IOError:
    font = ImageFont.load_default()

# Draw the text on the image
draw.text((10, 10), text, fill="black", font=font)

# Save the image
image.save("output_image.png")

# Optional: Show the image
image.show()

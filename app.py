from flask import Flask, request, render_template, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/install')
def install():
    return render_template('install.html')

@app.route('/process', methods=['POST'])
def process_image():
    from image2text import image2text
    image_file = request.files['image']
    block_size = int(request.form['block_size'])
    quantize = request.form.get('quantize') == 'on'
    image2text(image_file, size_str=block_size, quantize_in=quantize)
    # Save the processed image to a file
    print(f"Image file: {image_file.filename}")
    print(f"Block size: {block_size}")
    print(f"Quantize: {quantize}")

    return render_template("emoji_art.html")

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
from scripts.diffusion_upscale import upscale

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lr_img = request.form['img']
        resolution = request.form['resolution']
        up_img, video, = upscale(lr_img, resolution)

        # Handle form submission or other POST request logic here
        pass
    else:
        # Handle GET request logic here
        pass
    return render_template('index.html',
                           lr_img=lr_img if 'lr_rimg' in locals() else None,
                           up_img=up_img if 'up_img' in locals() else None
    )


if __name__ == '__main__':
    app.run()
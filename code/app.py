from flask import Flask, render_template, request, redirect, url_for, session
import os
import subprocess
import shutil

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

UPLOAD_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dummy login
USERNAME = 'Vinayaka'
PASSWORD = 'Vinay@1432'

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username == USERNAME and password == PASSWORD:
        session['user'] = username
        return redirect('/home')
    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'user' not in session:
        return redirect('/')

    text = request.form['text']

    captions_path = os.path.join('..', 'data', 'birds', 'example_captions.txt')
    with open(captions_path, 'w') as f:
        f.write(text)

    subprocess.run(['python', 'main.py', '--cfg', 'cfg/eval_bird.yml'])

    output_folder = '../models/bird_AttnGAN2/generated'
    image_files = sorted(
        [f for f in os.listdir(output_folder) if f.endswith('g2.png')],
        key=lambda x: os.path.getmtime(os.path.join(output_folder, x)),
        reverse=True
    )

    if not image_files:
        return render_template('index.html', error="No image was generated.")

    latest_image = image_files[0]
    src_image_path = os.path.join(output_folder, latest_image)
    final_output_path = os.path.join(UPLOAD_FOLDER, 'result.png')
    shutil.copyfile(src_image_path, final_output_path)

    return render_template('index.html', generated_image='generated/result.png')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

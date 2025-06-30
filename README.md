
This project presents a deep learning-based web application that generates high-resolution images from
natural language text descriptions using an Attention Generative Adversarial Network (AttnGAN).
Built on top of PyTorch and powered by pretrained models, the system takes descriptive sentences
from users and produces contextually relevant images with fine-grained details. The application
leverages the CUB-200-2011 bird dataset to ensure meaningful generation aligned with the given text.
A Flask-based backend manages inference requests, while a modern frontend integrates features like
animated eye loaders and smooth user interaction. Unlike traditional GAN models, AttnGAN
incorporates an attention mechanism to better correlate specific words with image regions, resulting in
significantly improved image realism and text-image coherence. The system is optimized for fast
generation and high-resolution output, demonstrating practical deployment of AI-powered creative
tools.
The architecture employs an advanced attention mechanism that dynamically associates individual
words in the input text with corresponding regions in the generated image, allowing for enhanced detail
and improved semantic consistency compared to traditional GAN approaches. This attention-driven
design facilitates the generation of images that not only reflect the overall content of the text but also
capture subtle attributes, such as color patterns and shapes, thereby significantly elevating image
realism.
A Flask-based backend efficiently manages inference requests, ensuring robust and scalable
deployment of the model in a web environment. The frontend is crafted with modern web technologies,
featuring interactive elements like animated eye loaders and smooth transitions to improve user
engagement and provide real-time feedback during image generation.
In addition to demonstrating the technical feasibility of combining natural language processing with
high-fidelity image synthesis, the system emphasizes practical considerations such as fast inference
times and generation of high-resolution outputs suitable for real-world applications. This project
showcases the potential of AI-powered creative tools to bridge language and vision, offering new
avenues for content creation, design prototyping, and educational visualization.
[GENERATIVE ADVERSARIALNETWORK (GAN) FORIMAGE GENERATION]
CSE(AI-ML), MLRITM 1
**CHAPTER 1
INTRODUCTION**
In the ever-evolving field of artificial intelligence and deep learning, generative models have gained
immense popularity for their ability to create realistic images from textual descriptions. Among
these, Attention GAN (Generative Adversarial Network) has emerged as a powerful architecture
that enhances image synthesis by incorporating attention mechanisms, ensuring that the generated
images capture fine-grained details. This project leverages Attention GAN and DAMSAM (Deep
Attentional Multimodal Similarity Model) encoders to generate images based on textual
descriptions from the CUB-200-2011 dataset, which consists of fine-grained bird species images
and their corresponding textual annotations.
Beyond the image generation aspect, this project integrates an interactive frontend where user
engagement plays a significant role. One of its unique features is the dynamic eye movement
animation that responds to cursor movement, creating an engaging visual experience. This frontend
not only provides a user-friendly interface for exploring generated images but also incorporates
responsiveness that adds a layer of interactivity, making the AI-generated images feel more lifelike.
To ensure seamless interaction between the frontend and the deep learning model, Flask has been
employed as the backend framework. Flask facilitates communication between the client-side
application and the trained Attention GAN model, allowing real-time image generation and display.
The Flask API serves as a bridge, ensuring efficient handling of requests and responses while
maintaining scalability.
This project aims to combine the strength of AI-powered image synthesis and interactive web
design to deliver a compelling user experience. Attention GAN, known for its capability to refine
image details based on textual prompts, helps in generating high-quality bird images that align with
human descriptions. Meanwhile, DAMSAM encoders ensure that the textual descriptions are
effectively encoded into meaningful embeddings, bridging the gap between vision and language
processing.

A major highlight of this project is its interactive front-end, which enhances user engagement
through cursor-responsive animations. The dynamic eye movement feature is an innovative addition
that makes the interaction feel more organic. The connection between the AI-generated visuals and
real-time user input through cursor tracking demonstrates a fusion of artificial intelligence and user
experience (UX) design, offering a fresh perspective on how deep learning models can be integrated
into real-world applications.
The motivation behind this project stems from the need to create a seamless and engaging AIpowered interface that transforms image generation from a passive output to an interactive
experience. Traditional generative models often focus on the quality of generated images, but this
project pushes the boundaries by introducing user-driven engagement elements, making AIgenerated art feel more responsive and immersive.
Furthermore, the use of Flask ensures that the backend remains lightweight yet efficient, enabling
real-time inference without compromising on performance. This framework allows easy
deployment, making the project accessible to a wider audience while maintaining the integrity of
complex deep learning computations.
Ultimately, this project showcases the synergy between generative AI, multimodal learning, and
interactive web development, serving as a stepping stone towards more intuitive AI-driven
applications. The fusion of Attention GAN, DAMSAM encoders, Flask, and a responsive frontend
brings together multiple disciplines, proving that AI-generated content is no longer confined to
static outputs but can instead become an engaging experience.
**DIRECTORY STRUCTURE
miniproject/
│
├── code/ # Contains main.py (inference script)
│ └── main.py
├── models/ # Stores pretrained models and generated results
│ └── birds_AttnGAN2/
│ └── ...
├── data/ # Containsthe processed dataset and captions
│ └── birds/
├── templates/ # HTML templates for Flask
│ └── index.html
├── static/ # Static files like generated images
[GENERATIVE ADVERSARIALNETWORK (GAN) FORIMAGE GENERATION]
CSE(AI-ML), MLRITM 24
│ └── generated/
├── app.py # Flask app entry point
└── cfg/ # YAML configuration files
└── eval_bird.yml**

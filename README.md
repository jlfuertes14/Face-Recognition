<a name="readme-top"></a>

<div align="center">
  <h1>Face Recognition System (OpenCV DNN)</h1>
  <p>
    Real-time face detection and recognition using OpenCV DNN and OpenFace embeddings.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python badge" />
    <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV badge" />
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT license badge" />
  </p>
  <p>
    <a href="#about-the-project">About</a>
    .
    <a href="#getting-started">Getting Started</a>
    .
    <a href="#usage">Usage</a>
    .
    <a href="#roadmap">Roadmap</a>
    .
    <a href="#contributing">Contributing</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#screenshots">Screenshots</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#author">Author</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project is a real-time face detection and recognition app built with OpenCV's DNN
module. It uses a Caffe SSD model for face detection and OpenFace embeddings for
recognition. The app runs from your system camera and supports registering, deleting,
and listing known faces.

Features:
- Real-time face detection and recognition from a webcam
- On-screen HUD and keyboard shortcuts
- Local persistent storage of face embeddings in `dataset/encodings.pkl`
- Automatic model download if files are missing

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- SCREENSHOTS -->
## Screenshots

![App UI](assets/screenshot.png)

Save the attached screenshot as `assets/screenshot.png` to display it here.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- BUILT WITH -->
## Built With

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Follow these steps to run the app locally.

### Prerequisites

- Python 3.x
- A webcam or built-in camera

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

When the window opens, use these keyboard shortcuts:

- `R` Register a new face
- `D` Delete a registered face
- `L` List all registered faces
- `T` Toggle between detection and recognition
- `Q` or `ESC` Quit

Tip: register the same person a few times from different angles for better accuracy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- PROJECT STRUCTURE -->
## Project Structure

```
.
├── main.py          # App entry point
├── detector.py      # Face detection (OpenCV DNN)
├── recognizer.py    # Face embeddings (OpenFace)
├── database.py      # Persistent storage for embeddings
├── ui.py            # UI drawing helpers
├── dataset/         # Saved encodings.pkl
└── models/          # DNN model files
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [ ] Add a simple setup script for first-time users
- [ ] Add optional CSV export for registered identities
- [ ] Improve registration flow with on-screen prompts

See the [open issues](../../issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome and appreciated.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- AUTHOR -->
## Author

John Lester Fuertes

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [OpenFace Model](https://github.com/cmusatyalab/openface)
- [Best README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

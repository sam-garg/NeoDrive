# neoDrive 🚗💨

**neoDrive** is an innovative, AI-powered gesture control system designed to revolutionize the way we interact with driving simulations. By leveraging advanced computer vision techniques, it allows users to drive a virtual car using natural hand and foot movements, eliminating the need for physical hardware like steering wheels or pedals.

🏆 **1st Place Winner** at the Hackathon organized by **VIPS College**.

## 👥 Team
**Members:** 2
*Developed with passion and precision during the hackathon.*

---

## 🌟 Features

*   **Dual Camera Support:** Utilizes two separate camera feeds simultaneously—one for hand tracking and another for foot tracking.
*   **Virtual Steering:** intuitive steering control calculated by tracking the relative position of your hands, mimicking a real steering wheel.
*   **Gesture-Based Gear Shifting:** Shift gears (1-4) instantly using specific finger configurations.
*   **Foot Pedal Control:** Real-time detection of foot movements to control Acceleration (Right Foot) and Braking (Left Foot).
*   **Universal Compatibility:** Maps gestures to standard keyboard keys (`W`, `A`, `S`, `D`, `1-4`), making it compatible with most PC racing games and simulators.

## 🛠️ Tech Stack

*   **Python**: The core logic and processing.
*   **OpenCV (`cv2`)**: For robust real-time video capture and image processing.
*   **MediaPipe**: Google's cutting-edge framework for precise Hand and Pose landmark detection.
*   **Custom Input Module**: For simulating low-level keyboard events to interface with games.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/neoDrive.git
    cd neoDrive
    ```

2.  **Install Dependencies**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Setup Cameras**: Connect two webcams to your PC.
    *   **Camera 1 (Index 2 in code):** Positioned to view your hands for steering/gears.
    *   **Camera 2 (Index 0 in code):** Positioned to view your feet for pedals.
    *   *(Note: You may need to adjust the camera indices in `main.py` lines 13 & 14 depending on your system).*

2.  **Run the Application**
    ```bash
    python main.py
    ```

3.  **Controls**:
    *   **Steering:** Move your hands left or right as if holding a wheel.
    *   **Accelerator:** Move your right foot down to press 'W'.
    *   **Brake:** Move your left foot down to press 'S'.
    *   **Gears:**
        *   STOP / Parking: Thumb & Index extended.
        *   Gear 2: Thumb, Index, Middle extended.
        *   Gear 3: Thumb, Index, Middle, Ring extended.
        *   Gear 4: Thumb, Index, Ring, Pinky extended (Spiderman/Rock gesture).

## 📄 License
This project is **NOT** open source. It is shared for demonstration and portfolio purposes only.

If you use any part of this project in your own work, strictly for educational or non-commercial purposes, you **MUST** give explicit credit to the original team members. Commercial use is prohibited without permission.

# Go Solving App (圍棋解題)

A cross-platform mobile app (Android/iOS) that recognizes Go board positions from photos and uses KataGo AI to suggest the best next move.

## Features

- **Board Recognition**: Take a photo or pick from gallery, and the app recognizes the board position using OpenCV
- **AI Analysis**: KataGo ONNX model analyzes the position and suggests top moves
- **Multiple Board Sizes**: Supports 9x9, 13x13, and 19x19 boards (auto-detected)
- **Win Rate Display**: Shows black/white win rate percentages
- **Ownership Map**: Visualizes territory ownership as a heatmap
- **Player Toggle**: Switch between analyzing for black or white's next move

## Tech Stack

| Component | Technology | Package |
|-----------|-----------|---------|
| Framework | Flutter | -- |
| Camera/Image | Flutter camera & image picker | `camera`, `image_picker` |
| Image Processing | OpenCV via Dart FFI | `opencv_dart` |
| AI Inference | ONNX Runtime | `flutter_onnxruntime` |
| AI Model | KataGo b6c64 ONNX | from katagotraining.org |

## Project Structure

```
lib/
├── main.dart                    # App entry point
├── app.dart                     # MaterialApp config & routing
├── models/
│   ├── board_state.dart         # Board state data model
│   └── analysis_result.dart     # KataGo analysis result model
├── services/
│   ├── board_recognition.dart   # OpenCV board recognition
│   ├── katago_engine.dart       # KataGo ONNX inference engine
│   └── image_service.dart       # Camera/gallery image service
├── screens/
│   ├── home_screen.dart         # Home (photo/gallery selection)
│   ├── camera_screen.dart       # Camera preview
│   └── analysis_screen.dart     # Board display & analysis results
└── widgets/
    ├── board_painter.dart       # Custom board rendering (CustomPainter)
    ├── stone_widget.dart        # Stone rendering
    └── move_suggestion.dart     # Best move suggestion display
```

## Setup

### Prerequisites

- Flutter SDK >= 3.10
- Android Studio / Xcode for mobile development

### Model Setup

1. Download the KataGo b6c64 ONNX model from [HuggingFace](https://huggingface.co/kaya-go/kaya) or convert from [katagotraining.org](https://katagotraining.org/extra_networks/)
2. Place the model file at `assets/models/katago_b6c64.onnx`

### Build & Run

```bash
# Install dependencies
flutter pub get

# Run on connected device/emulator
flutter run

# Run tests
flutter test

# Static analysis
flutter analyze
```

## Board Recognition Pipeline

1. **Preprocessing**: Grayscale, Gaussian blur, CLAHE contrast enhancement
2. **Board Detection**: Canny edge detection + contour detection for board boundary
3. **Perspective Correction**: 4-point perspective transform to rectify the board
4. **Grid Detection**: Hough line transform to detect grid lines, clustering to find N lines
5. **Board Size Detection**: Auto-detect 9x9, 13x13, or 19x19 from line count
6. **Stone Detection**: HSV color sampling at each intersection to classify black/white/empty

## KataGo Feature Encoding

- **Binary spatial features**: 22 channels (own stones, opponent stones, empty, perspective, move history, ladder features, etc.)
- **Global features**: 19 values (komi, player indicator, ko state, rules, etc.)
- **Output**: Policy (move probabilities), value (win rate), ownership (territory map)

## Testing

```bash
# Run all tests (43 tests)
flutter test

# Run specific test suite
flutter test test/models/board_state_test.dart
flutter test test/services/katago_engine_test.dart
flutter test test/services/board_recognition_test.dart
flutter test test/widgets/board_painter_test.dart
```

import 'package:flutter/material.dart';
import '../services/image_service.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final ImageService _imageService = ImageService();
  bool _isCapturing = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('拍照辨識'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.camera_alt,
              size: 80,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 24),
            const Text(
              '將棋盤放在畫面中央',
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 32),
            if (_isCapturing)
              const CircularProgressIndicator()
            else
              FilledButton.icon(
                onPressed: _capturePhoto,
                icon: const Icon(Icons.camera),
                label: const Text('拍照'),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _capturePhoto() async {
    setState(() => _isCapturing = true);
    try {
      final imagePath = await _imageService.captureFromCamera();
      if (imagePath != null && mounted) {
        Navigator.pushReplacementNamed(
          context,
          '/analysis',
          arguments: imagePath,
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isCapturing = false);
      }
    }
  }
}

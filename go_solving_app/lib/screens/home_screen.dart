import 'package:flutter/material.dart';
import '../services/image_service.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('圍棋解題'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.grid_on,
                size: 120,
                color: Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(height: 24),
              Text(
                '圍棋解題',
                style: Theme.of(context).textTheme.headlineLarge,
              ),
              const SizedBox(height: 8),
              Text(
                '拍照或選取棋盤圖片，AI 分析最佳下一手',
                style: Theme.of(context).textTheme.bodyLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: FilledButton.icon(
                  onPressed: () {
                    Navigator.pushNamed(context, '/camera');
                  },
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('拍照辨識'),
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: OutlinedButton.icon(
                  onPressed: () async {
                    final imageService = ImageService();
                    final imagePath = await imageService.pickFromGallery();
                    if (imagePath != null && context.mounted) {
                      Navigator.pushNamed(
                        context,
                        '/analysis',
                        arguments: imagePath,
                      );
                    }
                  },
                  icon: const Icon(Icons.photo_library),
                  label: const Text('從相簿選取'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

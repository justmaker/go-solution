import 'package:image_picker/image_picker.dart';

/// 影像擷取服務（相機拍照 + 相簿選取）
class ImageService {
  final ImagePicker _picker = ImagePicker();

  /// 從相機拍照
  Future<String?> captureFromCamera() async {
    final XFile? photo = await _picker.pickImage(
      source: ImageSource.camera,
      maxWidth: 1920,
      maxHeight: 1920,
      imageQuality: 90,
    );
    return photo?.path;
  }

  /// 從相簿選取圖片
  Future<String?> pickFromGallery() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1920,
      maxHeight: 1920,
      imageQuality: 90,
    );
    return image?.path;
  }
}

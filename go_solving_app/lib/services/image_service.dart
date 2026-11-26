import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';

/// 影像擷取服務（相機拍照 + 相簿選取）
class ImageService {
  final ImagePicker _picker = ImagePicker();

  /// 從相機拍照
  Future<String?> captureFromCamera() async {
    try {
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 90,
      );
      return photo?.path;
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[ImageService] 拍照失敗: $e');
      }
      rethrow;
    }
  }

  /// 從相簿選取圖片
  ///
  /// 使用 `ImageSource.gallery`，並明確捕捉例外（例如使用者未授權相簿權限、
  /// 或平台未設定對應的權限描述）以避免選取流程靜默失敗。
  Future<String?> pickFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 90,
      );
      return image?.path;
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[ImageService] 相簿選取失敗: $e');
      }
      rethrow;
    }
  }
}

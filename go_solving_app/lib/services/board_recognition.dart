import 'dart:math';
import '../models/board_state.dart';

/// OpenCV 棋盤辨識服務
///
/// 注意：完整的 OpenCV 辨識功能需要在實體裝置上使用 opencv_dart 套件。
/// 在模擬器上使用 stub 實作，回傳範例棋盤用於測試。
class BoardRecognition {
  /// 從影像檔案辨識棋盤狀態
  ///
  /// 在沒有 opencv_dart 的環境下，回傳一個範例棋盤用於 UI 測試。
  Future<BoardState> recognizeFromImage(String imagePath) async {
    // Stub: 回傳一個有幾顆棋子的範例 9x9 棋盤
    return _generateSampleBoard();
  }

  /// 產生範例棋盤用於測試
  BoardState _generateSampleBoard() {
    const boardSize = 9;
    final grid = List.generate(
      boardSize,
      (_) => List.filled(boardSize, StoneColor.empty),
    );

    // 放一些示範棋子（小型星位佈局）
    // 黑棋
    grid[2][2] = StoneColor.black;
    grid[2][6] = StoneColor.black;
    grid[4][4] = StoneColor.black;
    grid[6][2] = StoneColor.black;
    // 白棋
    grid[2][4] = StoneColor.white;
    grid[4][2] = StoneColor.white;
    grid[4][6] = StoneColor.white;
    grid[6][6] = StoneColor.white;

    return BoardState(boardSize: boardSize, grid: grid);
  }

  /// 聚類分析（保留供測試使用）
  List<double> clusterValues(List<double> values, double threshold) {
    if (values.isEmpty) return [];
    values.sort();

    final clusters = <double>[];
    var clusterSum = values[0];
    var clusterCount = 1;

    for (int i = 1; i < values.length; i++) {
      if (values[i] - values[i - 1] < threshold) {
        clusterSum += values[i];
        clusterCount++;
      } else {
        clusters.add(clusterSum / clusterCount);
        clusterSum = values[i];
        clusterCount = 1;
      }
    }
    clusters.add(clusterSum / clusterCount);

    return clusters;
  }

  /// 均勻分佈位置（保留供測試使用）
  List<double> generateUniformPositions(int count, double totalSize) {
    final margin = totalSize * 0.05;
    final step = (totalSize - 2 * margin) / (count - 1);
    return List.generate(count, (i) => margin + i * step);
  }
}

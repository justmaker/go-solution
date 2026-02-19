import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../models/board_state.dart';

/// 交叉點取樣資料
class _IntersectionSample {
  final int row;
  final int col;
  final double avgV;
  final double avgS;
  final double stdV;

  _IntersectionSample({
    required this.row,
    required this.col,
    required this.avgV,
    required this.avgS,
    required this.stdV,
  });
}

/// 辨識管線除錯資訊
class RecognitionDebugInfo {
  int contourCount = 0;
  int horizontalLineCount = 0;
  int verticalLineCount = 0;
  int detectedBoardSize = 0;
  List<double> clusterCenters = [];
  double thresholdBlackBoard = 0;
  double thresholdBoardWhite = 0;
  double adaptiveSatThreshold = 0;
  int blackCount = 0;
  int whiteCount = 0;
  int emptyCount = 0;
  double vMin = 0;
  double vMax = 0;
  double vMean = 0;

  @override
  String toString() {
    return '''
=== 棋盤辨識除錯 ===
輪廓數: $contourCount
水平線: $horizontalLineCount, 垂直線: $verticalLineCount
棋盤大小: ${detectedBoardSize}x$detectedBoardSize
V 值範圍: ${vMin.toStringAsFixed(1)} ~ ${vMax.toStringAsFixed(1)}, 平均: ${vMean.toStringAsFixed(1)}
聚類中心: ${clusterCenters.map((c) => c.toStringAsFixed(1)).join(', ')}
閾值: 黑/盤=${thresholdBlackBoard.toStringAsFixed(1)}, 盤/白=${thresholdBoardWhite.toStringAsFixed(1)}
飽和度閾值: ${adaptiveSatThreshold.toStringAsFixed(1)}
結果: 黑=$blackCount, 白=$whiteCount, 空=$emptyCount
====================''';
  }
}

/// OpenCV 棋盤辨識服務
class BoardRecognition {
  /// 最近一次辨識的除錯資訊
  RecognitionDebugInfo? lastDebugInfo;
  /// 從影像檔案辨識棋盤狀態
  Future<BoardState> recognizeFromImage(String imagePath) async {
    // 1. 讀取影像
    final img = cv.imread(imagePath);
    if (img.isEmpty) {
      throw Exception('無法讀取影像: $imagePath');
    }

    try {
      // 2. 影像前處理
      final processed = _preprocess(img);

      // 3. 偵測棋盤邊界並進行透視校正
      final warped = _findAndWarpBoard(processed, img);
      processed.dispose();

      // 4. 偵測格線並推斷棋盤大小
      final (boardSize, intersections) = _detectGridLines(warped);

      // 5. 在每個交叉點偵測棋子
      final grid = _detectStones(warped, boardSize, intersections);
      warped.dispose();

      return BoardState(boardSize: boardSize, grid: grid);
    } finally {
      img.dispose();
    }
  }

  /// 產生範例棋盤用於測試（無需 OpenCV）
  /// 使用 19x19 棋盤以充分利用 KataGo 模型
  BoardState generateSampleBoard() {
    const boardSize = 19;
    final grid = List.generate(
      boardSize,
      (_) => List.filled(boardSize, StoneColor.empty),
    );
    // 星位附近的開局範例
    grid[3][3] = StoneColor.black;   // D16
    grid[3][15] = StoneColor.black;  // Q16
    grid[15][3] = StoneColor.black;  // D4
    grid[2][5] = StoneColor.white;   // F17
    grid[3][9] = StoneColor.white;   // K16
    grid[15][15] = StoneColor.white; // Q4
    return BoardState(
      boardSize: boardSize,
      grid: grid,
      nextPlayer: StoneColor.white,
      komi: 7.5,
    );
  }

  /// 影像前處理：灰階、模糊、CLAHE 對比增強
  cv.Mat _preprocess(cv.Mat img) {
    final gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (5, 5), 1.0);
    final clahe = cv.CLAHE(2.0, (8, 8));
    final enhanced = clahe.apply(blurred);
    gray.dispose();
    blurred.dispose();
    return enhanced;
  }

  /// 偵測棋盤邊界並進行四點透視校正
  cv.Mat _findAndWarpBoard(cv.Mat processed, cv.Mat original) {
    final edges = cv.canny(processed, 50, 150);

    final (contours, hierarchy) = cv.findContours(
      edges,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    edges.dispose();

    if (contours.isEmpty) {
      return original.clone();
    }

    cv.VecPoint? bestContour;
    double maxArea = 0;

    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area < original.rows * original.cols * 0.1) continue;

      final peri = cv.arcLength(contour, true);
      final approx = cv.approxPolyDP(contour, 0.02 * peri, true);

      if (approx.length == 4 && area > maxArea) {
        maxArea = area;
        bestContour = approx;
      }
    }

    if (bestContour == null || bestContour.length != 4) {
      return original.clone();
    }

    final corners = _orderPoints(bestContour);

    final width = max(
      _distance(corners[0], corners[1]),
      _distance(corners[2], corners[3]),
    ).toInt();
    final height = max(
      _distance(corners[0], corners[3]),
      _distance(corners[1], corners[2]),
    ).toInt();

    final size = max(width, height);

    final srcPoints = cv.VecPoint2f.fromList(
      corners.map((p) => cv.Point2f(p.x.toDouble(), p.y.toDouble())).toList(),
    );
    final dstPoints = cv.VecPoint2f.fromList([
      cv.Point2f(0, 0),
      cv.Point2f(size.toDouble() - 1, 0),
      cv.Point2f(size.toDouble() - 1, size.toDouble() - 1),
      cv.Point2f(0, size.toDouble() - 1),
    ]);

    final matrix = cv.getPerspectiveTransform2f(srcPoints, dstPoints);
    final warped = cv.warpPerspective(original, matrix, (size, size));

    matrix.dispose();
    return warped;
  }

  List<cv.Point> _orderPoints(cv.VecPoint points) {
    final pts = points.toList();
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];
    final remaining = [pts[1], pts[2]];
    remaining.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    final tr = remaining[0];
    final bl = remaining[1];
    return [tl, tr, br, bl];
  }

  double _distance(cv.Point a, cv.Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2).toDouble());
  }

  /// 使用霍夫線變換偵測格線，自動推斷棋盤大小
  (int, List<List<cv.Point2f>>) _detectGridLines(cv.Mat warped) {
    final gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (3, 3), 0);
    final edges = cv.canny(blurred, 50, 150);

    final linesMat = cv.HoughLinesP(
      edges,
      1,
      pi / 180,
      80,
      minLineLength: warped.cols * 0.3,
      maxLineGap: warped.cols / 20,
    );

    gray.dispose();
    blurred.dispose();
    edges.dispose();

    final horizontalYs = <double>[];
    final verticalXs = <double>[];

    for (int i = 0; i < linesMat.rows; i++) {
      final line = linesMat.at<cv.Vec4i>(i, 0);
      final x1 = line.val1.toDouble();
      final y1 = line.val2.toDouble();
      final x2 = line.val3.toDouble();
      final y2 = line.val4.toDouble();

      final angle = atan2((y2 - y1).abs(), (x2 - x1).abs());
      if (angle < pi / 6) {
        horizontalYs.add((y1 + y2) / 2);
      } else if (angle > pi / 3) {
        verticalXs.add((x1 + x2) / 2);
      }
    }
    linesMat.dispose();

    final hClusters = _clusterValues(horizontalYs, warped.rows * 0.02);
    final vClusters = _clusterValues(verticalXs, warped.cols * 0.02);

    final detectedSize = ((hClusters.length + vClusters.length) / 2).round();
    final int boardSize;
    if (detectedSize <= 11) {
      boardSize = 9;
    } else if (detectedSize <= 16) {
      boardSize = 13;
    } else {
      boardSize = 19;
    }

    hClusters.sort();
    vClusters.sort();

    final hLines = hClusters.length >= boardSize
        ? hClusters.sublist(0, boardSize)
        : _generateUniformPositions(boardSize, warped.rows.toDouble());
    final vLines = vClusters.length >= boardSize
        ? vClusters.sublist(0, boardSize)
        : _generateUniformPositions(boardSize, warped.cols.toDouble());

    final intersections = <List<cv.Point2f>>[];
    for (int r = 0; r < boardSize; r++) {
      final row = <cv.Point2f>[];
      for (int c = 0; c < boardSize; c++) {
        row.add(cv.Point2f(vLines[c], hLines[r]));
      }
      intersections.add(row);
    }

    return (boardSize, intersections);
  }

  List<double> _clusterValues(List<double> values, double threshold) {
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

  List<double> _generateUniformPositions(int count, double totalSize) {
    final margin = totalSize * 0.05;
    final step = (totalSize - 2 * margin) / (count - 1);
    return List.generate(count, (i) => margin + i * step);
  }

  /// 1D k-means 聚類（k 群）
  /// 回傳排序後的聚類中心
  List<double> _kMeans1D(List<double> values, int k, {int maxIter = 50}) {
    if (values.isEmpty) return [];
    final sorted = List<double>.from(values)..sort();

    // 初始中心：使用百分位數
    List<double> centers;
    if (k == 3) {
      final p10 = sorted[(sorted.length * 0.1).floor()];
      final p50 = sorted[sorted.length ~/ 2];
      final p90 = sorted[(sorted.length * 0.9).floor().clamp(0, sorted.length - 1)];
      centers = [p10, p50, p90];
    } else {
      centers = List.generate(k, (i) => sorted[(sorted.length * (i + 1) / (k + 1)).floor()]);
    }

    for (int iter = 0; iter < maxIter; iter++) {
      // 分配每個值到最近的中心
      final clusters = List.generate(k, (_) => <double>[]);
      for (final v in values) {
        int bestIdx = 0;
        double bestDist = (v - centers[0]).abs();
        for (int i = 1; i < k; i++) {
          final dist = (v - centers[i]).abs();
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
          }
        }
        clusters[bestIdx].add(v);
      }

      // 重算中心
      bool converged = true;
      for (int i = 0; i < k; i++) {
        if (clusters[i].isEmpty) continue;
        final newCenter = clusters[i].reduce((a, b) => a + b) / clusters[i].length;
        if ((newCenter - centers[i]).abs() > 0.5) {
          converged = false;
        }
        centers[i] = newCenter;
      }

      if (converged) break;
    }

    centers.sort();
    return centers;
  }

  /// 計算自適應閾值，處理邊界情況（<3 有效群體）
  ({double thresholdBB, double thresholdBW, double satThreshold})
      _computeAdaptiveThresholds(
    List<double> centers,
    List<_IntersectionSample> samples,
  ) {
    if (centers.length < 2) {
      // 只有一個群體，用固定 fallback
      return (thresholdBB: 80, thresholdBW: 180, satThreshold: 40);
    }

    // 合併距離過近的群體（< 15% V 值全距）
    final vValues = samples.map((s) => s.avgV).toList();
    final vRange = (vValues.reduce(max) - vValues.reduce(min));
    final mergeThreshold = vRange * 0.15;

    final merged = <double>[centers[0]];
    final memberCounts = <int>[1];
    for (int i = 1; i < centers.length; i++) {
      if ((centers[i] - merged.last).abs() < mergeThreshold) {
        // 合併：取加權平均
        merged.last = (merged.last * memberCounts.last + centers[i]) / (memberCounts.last + 1);
        memberCounts.last++;
      } else {
        merged.add(centers[i]);
        memberCounts.add(1);
      }
    }

    double thresholdBB;
    double thresholdBW;

    if (merged.length >= 3) {
      // 三群：黑、盤、白
      thresholdBB = (merged[0] + merged[1]) / 2;
      thresholdBW = (merged[1] + merged[2]) / 2;
    } else if (merged.length == 2) {
      // 兩群：可能無白子或無黑子
      final gap = merged[1] - merged[0];
      if (merged[0] < 100) {
        // 可能是黑+盤（無白子）
        thresholdBB = (merged[0] + merged[1]) / 2;
        thresholdBW = merged[1] + gap * 0.5;
      } else {
        // 可能是盤+白（無黑子）
        thresholdBB = merged[0] - gap * 0.5;
        thresholdBW = (merged[0] + merged[1]) / 2;
      }
    } else {
      thresholdBB = 80;
      thresholdBW = 180;
    }

    // 自適應飽和度門檻：取亮群體（V > thresholdBW）的飽和度中位數 × 1.5
    final brightSamples = samples.where((s) => s.avgV > thresholdBW).toList();
    double satThreshold;
    if (brightSamples.length >= 2) {
      final satValues = brightSamples.map((s) => s.avgS).toList()..sort();
      final medianSat = satValues[satValues.length ~/ 2];
      satThreshold = medianSat * 1.5;
      // 至少給 30，避免過度嚴格
      satThreshold = max(satThreshold, 30);
    } else {
      // fallback：使用所有交叉點飽和度的全局統計
      final allSat = samples.map((s) => s.avgS).toList()..sort();
      satThreshold = allSat[allSat.length ~/ 2] * 0.8;
      satThreshold = max(satThreshold, 30);
    }

    return (
      thresholdBB: thresholdBB,
      thresholdBW: thresholdBW,
      satThreshold: satThreshold,
    );
  }

  List<List<StoneColor>> _detectStones(
    cv.Mat warped,
    int boardSize,
    List<List<cv.Point2f>> intersections,
  ) {
    final debug = RecognitionDebugInfo();
    debug.detectedBoardSize = boardSize;

    final hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV);
    final grid = List.generate(
      boardSize,
      (_) => List.filled(boardSize, StoneColor.empty),
    );

    final sampleRadius = (warped.cols / boardSize * 0.2).round();

    // === 第一階段：收集所有交叉點的特徵值 ===
    final samples = <_IntersectionSample>[];

    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        final pt = intersections[r][c];
        final x = pt.x.round().clamp(0, warped.cols - 1);
        final y = pt.y.round().clamp(0, warped.rows - 1);

        var totalV = 0.0;
        var totalS = 0.0;
        var totalV2 = 0.0; // for stdV
        var sampleCount = 0;

        for (int dy = -sampleRadius; dy <= sampleRadius; dy++) {
          for (int dx = -sampleRadius; dx <= sampleRadius; dx++) {
            final sx = (x + dx).clamp(0, warped.cols - 1);
            final sy = (y + dy).clamp(0, warped.rows - 1);
            final pixel = hsv.atPixel(sy, sx);
            totalS += pixel[1];
            final v = pixel[2].toDouble();
            totalV += v;
            totalV2 += v * v;
            sampleCount++;
          }
        }

        final avgV = totalV / sampleCount;
        final avgS = totalS / sampleCount;
        final variance = (totalV2 / sampleCount) - (avgV * avgV);
        final stdV = sqrt(max(0, variance));

        samples.add(_IntersectionSample(
          row: r,
          col: c,
          avgV: avgV,
          avgS: avgS,
          stdV: stdV,
        ));
      }
    }

    hsv.dispose();

    // 收集 V 值統計
    final vValues = samples.map((s) => s.avgV).toList();
    debug.vMin = vValues.reduce(min);
    debug.vMax = vValues.reduce(max);
    debug.vMean = vValues.reduce((a, b) => a + b) / vValues.length;

    // === 第二階段：1D k-means 聚類 ===
    final centers = _kMeans1D(vValues, 3);
    debug.clusterCenters = List.from(centers);

    // === 計算自適應閾值 ===
    final thresholds = _computeAdaptiveThresholds(centers, samples);
    debug.thresholdBlackBoard = thresholds.thresholdBB;
    debug.thresholdBoardWhite = thresholds.thresholdBW;
    debug.adaptiveSatThreshold = thresholds.satThreshold;

    // === 第三階段：用自適應閾值分類每個交叉點 ===
    for (final sample in samples) {
      if (sample.avgV < thresholds.thresholdBB) {
        grid[sample.row][sample.col] = StoneColor.black;
        debug.blackCount++;
      } else if (sample.avgV > thresholds.thresholdBW &&
          sample.avgS < thresholds.satThreshold) {
        grid[sample.row][sample.col] = StoneColor.white;
        debug.whiteCount++;
      } else {
        debug.emptyCount++;
      }
    }

    lastDebugInfo = debug;
    debugPrint(debug.toString());

    return grid;
  }
}

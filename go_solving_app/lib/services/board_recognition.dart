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
  String boardDetectionMethod = '';
  int detectedBoardSize = 0;
  int hLineCount = 0;
  int vLineCount = 0;
  double hSpacing = 0;
  double vSpacing = 0;
  List<double> clusterCenters = [];
  double thresholdBlackBoard = 0;
  double thresholdBoardWhite = 0;
  double satLimitBlack = 0;
  double satLimitWhite = 0;
  int blackCount = 0;
  int whiteCount = 0;
  int emptyCount = 0;
  double vMin = 0;
  double vMax = 0;

  @override
  String toString() {
    return '''
=== 棋盤辨識除錯 ===
板面偵測: $boardDetectionMethod
格線: H=$hLineCount, V=$vLineCount → ${detectedBoardSize}x$detectedBoardSize
間距: H=${hSpacing.toStringAsFixed(1)}, V=${vSpacing.toStringAsFixed(1)}
V 範圍: ${vMin.toStringAsFixed(1)} ~ ${vMax.toStringAsFixed(1)}
聚類中心: ${clusterCenters.map((c) => c.toStringAsFixed(1)).join(', ')}
閾值: B<${thresholdBlackBoard.toStringAsFixed(1)} S<${satLimitBlack.toStringAsFixed(1)}, W>${thresholdBoardWhite.toStringAsFixed(1)} S<${satLimitWhite.toStringAsFixed(1)}
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
    final img = cv.imread(imagePath);
    if (img.isEmpty) {
      throw Exception('無法讀取影像: $imagePath');
    }

    try {
      // 1. 偵測棋盤邊界並進行透視校正
      final warped = _findAndWarpBoard(img);

      // 2. 偵測格線並推斷棋盤大小（均勻間距）
      final (boardSize, intersections) = _detectGridLines(warped);

      // 3. 在每個交叉點偵測棋子
      final grid = _detectStones(warped, boardSize, intersections);
      warped.dispose();

      return BoardState(boardSize: boardSize, grid: grid);
    } finally {
      img.dispose();
    }
  }

  /// 產生範例棋盤用於測試（無需 OpenCV）
  BoardState generateSampleBoard() {
    const boardSize = 19;
    final grid = List.generate(
      boardSize,
      (_) => List.filled(boardSize, StoneColor.empty),
    );
    grid[3][3] = StoneColor.black;
    grid[3][15] = StoneColor.black;
    grid[15][3] = StoneColor.black;
    grid[2][5] = StoneColor.white;
    grid[3][9] = StoneColor.white;
    grid[15][15] = StoneColor.white;
    return BoardState(
      boardSize: boardSize,
      grid: grid,
      nextPlayer: StoneColor.white,
      komi: 7.5,
    );
  }

  // ============================================================
  // 步驟 1：棋盤偵測與透視校正
  // ============================================================

  /// 嘗試用顏色偵測棋盤區域，失敗則用邊緣偵測，再失敗返回原圖
  cv.Mat _findAndWarpBoard(cv.Mat original) {
    // 方法 A：顏色偵測（棋盤木色 H≈12-35, S>50）
    final warped = _findBoardByColor(original);
    if (warped != null) return warped;

    // 方法 B：邊緣偵測（Canny + 輪廓）
    final gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (5, 5), 1.0);
    final clahe = cv.CLAHE(2.0, (8, 8));
    final enhanced = clahe.apply(blurred);
    final edgeWarped = _findBoardByEdge(enhanced, original);
    gray.dispose();
    blurred.dispose();
    enhanced.dispose();
    if (edgeWarped != null) return edgeWarped;

    return original.clone();
  }

  /// 顏色偵測：用 HSV 過濾暖色木板，找最大連通區域
  cv.Mat? _findBoardByColor(cv.Mat original) {
    final hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV);

    // 棋盤木色範圍：暖黃/橘（H 12-35）
    final lower = cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [12, 50, 100]);
    final upper = cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [35, 255, 255]);
    final mask = cv.inRange(hsv, lower, upper);
    hsv.dispose();
    lower.dispose();
    upper.dispose();

    // 形態學清理
    final kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15));
    final closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel);
    final cleaned = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel);
    mask.dispose();
    closed.dispose();
    kernel.dispose();

    final (contours, _) = cv.findContours(
      cleaned,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    cleaned.dispose();

    if (contours.isEmpty) return null;

    // 找最大輪廓
    cv.VecPoint? bestContour;
    double maxArea = 0;
    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area > maxArea) {
        maxArea = area;
        bestContour = contour;
      }
    }

    // 面積至少佔影像 20%
    if (bestContour == null ||
        maxArea < original.rows * original.cols * 0.2) {
      return null;
    }

    // 取最小面積矩形
    final rect = cv.minAreaRect(bestContour);
    final boxPoints = cv.boxPoints(rect);
    final corners = _orderPoints2f(boxPoints);

    final w = max(
      _distance2f(corners[0], corners[1]),
      _distance2f(corners[2], corners[3]),
    ).toInt();
    final h = max(
      _distance2f(corners[0], corners[3]),
      _distance2f(corners[1], corners[2]),
    ).toInt();
    final size = max(w, h);
    if (size < 100) return null;

    final srcPoints = cv.VecPoint2f.fromList(corners);
    final dstPoints = cv.VecPoint2f.fromList([
      cv.Point2f(0, 0),
      cv.Point2f(size.toDouble() - 1, 0),
      cv.Point2f(size.toDouble() - 1, size.toDouble() - 1),
      cv.Point2f(0, size.toDouble() - 1),
    ]);

    final matrix = cv.getPerspectiveTransform2f(srcPoints, dstPoints);
    final warped = cv.warpPerspective(original, matrix, (size, size));
    matrix.dispose();

    debugPrint('[BoardRecognition] 顏色偵測成功: ${size}x$size');
    return warped;
  }

  List<cv.Point2f> _orderPoints2f(cv.VecPoint2f points) {
    final pts = points.toList();
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];
    final remaining = [pts[1], pts[2]];
    remaining.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    return [tl, remaining[0], br, remaining[1]];
  }

  double _distance2f(cv.Point2f a, cv.Point2f b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  /// 邊緣偵測：Canny + 找四邊形輪廓
  cv.Mat? _findBoardByEdge(cv.Mat processed, cv.Mat original) {
    final edges = cv.canny(processed, 50, 150);
    final (contours, _) = cv.findContours(
      edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE,
    );
    edges.dispose();

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

    if (bestContour == null || bestContour.length != 4) return null;

    final corners = _orderPoints(bestContour);
    final w = max(
      _distance(corners[0], corners[1]),
      _distance(corners[2], corners[3]),
    ).toInt();
    final h = max(
      _distance(corners[0], corners[3]),
      _distance(corners[1], corners[2]),
    ).toInt();
    final size = max(w, h);

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

    debugPrint('[BoardRecognition] 邊緣偵測成功: ${size}x$size');
    return warped;
  }

  List<cv.Point> _orderPoints(cv.VecPoint points) {
    final pts = points.toList();
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];
    final remaining = [pts[1], pts[2]];
    remaining.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    return [tl, remaining[0], br, remaining[1]];
  }

  double _distance(cv.Point a, cv.Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2).toDouble());
  }

  // ============================================================
  // 步驟 2：格線偵測（暴力搜尋最佳間距）
  // ============================================================

  (int, List<List<cv.Point2f>>) _detectGridLines(cv.Mat warped) {
    final size = warped.rows; // 正方形影像

    // === Hough 線偵測 ===
    final gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (3, 3), 0);
    final edges = cv.canny(blurred, 50, 150);

    final linesMat = cv.HoughLinesP(
      edges, 1, pi / 180,
      50,
      minLineLength: warped.cols * 0.2,
      maxLineGap: warped.cols / 15,
    );

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

    final hClusters = _clusterValues(horizontalYs, size * 0.02);
    final vClusters = _clusterValues(verticalXs, size * 0.02);

    // === 投影法 dip 偵測 ===
    final hProj = _computeProjection(gray, true);
    final vProj = _computeProjection(gray, false);
    gray.dispose();

    var k = max(3, size ~/ 200);
    if (k % 2 == 0) k++;
    final hSmooth = _smoothProfile(hProj, k);
    final vSmooth = _smoothProfile(vProj, k);

    final minDist = size ~/ 30;
    final hDips = _findDips(hSmooth, minDist);
    final vDips = _findDips(vSmooth, minDist);

    // === 合併 Hough + 投影 ===
    final hCombined = _combinePositions(hDips, hClusters, size * 0.025);
    final vCombined = _combinePositions(vDips, vClusters, size * 0.025);

    debugPrint('[BoardRecognition] Hough: H=${hClusters.length}, V=${vClusters.length}');
    debugPrint('[BoardRecognition] Dips: H=${hDips.length}, V=${vDips.length}');
    debugPrint('[BoardRecognition] Combined: H=${hCombined.length}, V=${vCombined.length}');

    // === 暴力搜尋 H/V 各自的最佳間距和相位 ===
    final (hSpacing, hPhase, hInl) =
        _findBestSpacing(hCombined, size.toDouble());
    final (vSpacing, vPhase, vInl) =
        _findBestSpacing(vCombined, size.toDouble());

    // 從最佳間距生成格線
    final hLines =
        _generateGridFromSpacing(hPhase, hSpacing, size.toDouble());
    final vLines =
        _generateGridFromSpacing(vPhase, vSpacing, size.toDouble());

    // 決定棋盤大小：用較多的一邊向上取整到標準尺寸
    final maxLines = hLines.length > vLines.length ? hLines.length : vLines.length;
    final int boardSize;
    if (maxLines <= 10) {
      boardSize = 9;
    } else if (maxLines <= 14) {
      boardSize = 13;
    } else {
      boardSize = 19;
    }

    // 修剪/擴展到棋盤大小
    final hFinal =
        _trimToSize(hLines, boardSize, hSpacing, size.toDouble());
    final vFinal =
        _trimToSize(vLines, boardSize, vSpacing, size.toDouble());

    final intersections = <List<cv.Point2f>>[];
    for (int r = 0; r < boardSize; r++) {
      final row = <cv.Point2f>[];
      for (int c = 0; c < boardSize; c++) {
        row.add(cv.Point2f(vFinal[c], hFinal[r]));
      }
      intersections.add(row);
    }

    debugPrint(
        '[BoardRecognition] 間距: H=${hSpacing.toStringAsFixed(1)} (inl=$hInl), V=${vSpacing.toStringAsFixed(1)} (inl=$vInl)');
    debugPrint(
        '[BoardRecognition] 格線: ${hLines.length}x${vLines.length} → ${boardSize}x$boardSize');
    return (boardSize, intersections);
  }

  /// 暴力搜尋最佳間距：對每個候選間距，找最佳相位，算 inlier 數
  (double, double, int) _findBestSpacing(List<double> positions, double totalSize) {
    if (positions.length < 3) {
      return (totalSize / 14, totalSize * 0.05, 0);
    }

    final minSp = (totalSize * 0.04).toInt(); // ~19 路最小間距
    final maxSp = (totalSize * 0.12).toInt(); // ~9 路最大間距

    var bestSpacing = totalSize / 14;
    var bestPhase = 0.0;
    var bestScore = 0;

    for (int sp = minSp; sp <= maxSp; sp++) {
      final tolerance = sp * 0.12;
      for (final ref in positions) {
        final phase = ref % sp;
        var inliers = 0;
        for (final p in positions) {
          var remainder = (p - phase) % sp;
          if (remainder > sp / 2) remainder = sp - remainder;
          if (remainder < tolerance) inliers++;
        }
        if (inliers > bestScore) {
          bestScore = inliers;
          bestSpacing = sp.toDouble();
          bestPhase = phase;
        }
      }
    }

    // Harmonic 修正：如果 2× 間距也有類似的 inlier 數，
    // 說明當前結果是半間距（harmonic），改用 2×
    final doubleSp = (bestSpacing * 2).round();
    if (doubleSp <= maxSp) {
      final dTol = doubleSp * 0.12;
      var dBestInliers = 0;
      var dBestPhase = 0.0;
      for (final ref in positions) {
        final phase = ref % doubleSp;
        var inliers = 0;
        for (final p in positions) {
          var remainder = (p - phase) % doubleSp;
          if (remainder > doubleSp / 2) remainder = doubleSp - remainder;
          if (remainder < dTol) inliers++;
        }
        if (inliers > dBestInliers) {
          dBestInliers = inliers;
          dBestPhase = phase;
        }
      }
      // 半間距的 inlier 數幾乎相同 → 是 harmonic，改用全間距
      if (dBestInliers >= bestScore * 0.8) {
        bestSpacing = doubleSp.toDouble();
        bestPhase = dBestPhase;
        bestScore = dBestInliers;
      }
    }

    // 用 inlier 位置精修間距
    final tolerance = bestSpacing * 0.12;
    final inlierPos = <double>[];
    for (final p in positions) {
      var remainder = (p - bestPhase) % bestSpacing;
      if (remainder > bestSpacing / 2) remainder = bestSpacing - remainder;
      if (remainder < tolerance) inlierPos.add(p);
    }
    inlierPos.sort();

    if (inlierPos.length >= 2) {
      final refinedDiffs = <double>[];
      for (int i = 0; i < inlierPos.length - 1; i++) {
        final d = inlierPos[i + 1] - inlierPos[i];
        final n = (d / bestSpacing).round();
        if (n > 0) refinedDiffs.add(d / n);
      }
      if (refinedDiffs.isNotEmpty) {
        bestSpacing = refinedDiffs.reduce((a, b) => a + b) / refinedDiffs.length;
      }
    }

    // 用精修間距重算最佳相位
    final tol2 = bestSpacing * 0.12;
    var bestPhase2 = 0.0;
    var bestInliers2 = 0;
    for (final ref in positions) {
      final phase = ref % bestSpacing;
      var inliers = 0;
      for (final p in positions) {
        var remainder = (p - phase) % bestSpacing;
        if (remainder > bestSpacing / 2) remainder = bestSpacing - remainder;
        if (remainder < tol2) inliers++;
      }
      if (inliers > bestInliers2) {
        bestInliers2 = inliers;
        bestPhase2 = phase;
      }
    }

    return (bestSpacing, bestPhase2, bestInliers2);
  }

  /// 從間距和相位生成格線位置
  List<double> _generateGridFromSpacing(double phase, double spacing, double totalSize) {
    final lines = <double>[];
    var k = 0;
    while (phase + k * spacing >= 0) {
      k--;
    }
    k++;
    while (phase + k * spacing < totalSize) {
      lines.add(phase + k * spacing);
      k++;
    }
    return lines;
  }

  /// 修剪或擴展格線到目標數量（置中）
  List<double> _trimToSize(List<double> lines, int target, double spacing, double totalSize) {
    final result = List<double>.from(lines);
    if (result.length > target) {
      final start = (result.length - target) ~/ 2;
      return result.sublist(start, start + target);
    }
    // 始終擴展到 target：優先在邊界內，必要時允許超出
    while (result.length < target) {
      final nextP = result.last + spacing;
      final prevP = result.first - spacing;
      if (nextP < totalSize) {
        result.add(nextP);
      } else if (prevP >= 0) {
        result.insert(0, prevP);
      } else if (nextP - totalSize < result.first.abs()) {
        result.add(nextP);
      } else {
        result.insert(0, prevP);
      }
    }
    return result.sublist(0, target);
  }

  /// 計算單通道影像的行或列平均值（投影）
  List<double> _computeProjection(cv.Mat singleChannel, bool horizontal) {
    final rows = singleChannel.rows;
    final cols = singleChannel.cols;
    final data = singleChannel.data;

    if (horizontal) {
      // Row means → 偵測水平格線
      final proj = List<double>.filled(rows, 0);
      for (int y = 0; y < rows; y++) {
        double sum = 0;
        final offset = y * cols;
        for (int x = 0; x < cols; x++) {
          sum += data[offset + x];
        }
        proj[y] = sum / cols;
      }
      return proj;
    } else {
      // Column means → 偵測垂直格線
      final proj = List<double>.filled(cols, 0);
      for (int y = 0; y < rows; y++) {
        final offset = y * cols;
        for (int x = 0; x < cols; x++) {
          proj[x] += data[offset + x];
        }
      }
      for (int x = 0; x < cols; x++) {
        proj[x] /= rows;
      }
      return proj;
    }
  }

  /// 移動平均平滑
  List<double> _smoothProfile(List<double> profile, int kernelSize) {
    final result = List<double>.filled(profile.length, 0);
    final halfK = kernelSize ~/ 2;
    for (int i = 0; i < profile.length; i++) {
      double sum = 0;
      int count = 0;
      final start = max(0, i - halfK);
      final end = min(profile.length - 1, i + halfK);
      for (int j = start; j <= end; j++) {
        sum += profile[j];
        count++;
      }
      result[i] = sum / count;
    }
    return result;
  }

  /// 找投影中的低谷（local minima，代表格線位置）
  List<int> _findDips(List<double> profile, int minDist) {
    // 計算中位數
    final sorted = List<double>.from(profile)..sort();
    final median = sorted[sorted.length ~/ 2];

    final dips = <int>[];
    for (int i = minDist; i < profile.length - minDist; i++) {
      // 在 ±minDist 窗口內找最小值
      double minVal = double.infinity;
      for (int j = max(0, i - minDist);
          j <= min(profile.length - 1, i + minDist);
          j++) {
        if (profile[j] < minVal) minVal = profile[j];
      }
      // 此位置是局部最小，且低於中位數
      if (profile[i] == minVal && profile[i] < median) {
        dips.add(i);
      }
    }
    return dips;
  }

  /// 合併投影 dips 和 Hough clusters，重新聚類
  List<double> _combinePositions(
      List<int> dips, List<double> hough, double tolerance) {
    final allPos = <double>[];
    for (final d in dips) {
      allPos.add(d.toDouble());
    }
    allPos.addAll(hough);
    allPos.sort();

    if (allPos.isEmpty) return [];
    final clusters = <double>[];
    var cSum = allPos[0];
    var cCount = 1;
    for (int i = 1; i < allPos.length; i++) {
      if (allPos[i] - allPos[i - 1] < tolerance) {
        cSum += allPos[i];
        cCount++;
      } else {
        clusters.add(cSum / cCount);
        cSum = allPos[i];
        cCount = 1;
      }
    }
    clusters.add(cSum / cCount);
    return clusters;
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

  // ============================================================
  // 步驟 3：棋子偵測
  // ============================================================

  /// 1D k-means 聚類
  List<double> _kMeans1D(List<double> values, int k, {int maxIter = 50}) {
    if (values.isEmpty) return [];
    final sorted = List<double>.from(values)..sort();
    List<double> centers;
    if (k == 3) {
      centers = [
        sorted[(sorted.length * 0.1).floor()],
        sorted[sorted.length ~/ 2],
        sorted[(sorted.length * 0.9).floor().clamp(0, sorted.length - 1)],
      ];
    } else {
      centers = List.generate(
        k, (i) => sorted[(sorted.length * (i + 1) / (k + 1)).floor()],
      );
    }

    for (int iter = 0; iter < maxIter; iter++) {
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
      bool converged = true;
      for (int i = 0; i < k; i++) {
        if (clusters[i].isEmpty) continue;
        final nc = clusters[i].reduce((a, b) => a + b) / clusters[i].length;
        if ((nc - centers[i]).abs() > 0.5) converged = false;
        centers[i] = nc;
      }
      if (converged) break;
    }
    centers.sort();
    return centers;
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
    final samples = <_IntersectionSample>[];

    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        final pt = intersections[r][c];
        final x = pt.x.round();
        final y = pt.y.round();

        // 超出影像邊界的交叉點標記為棋盤色（不會被判為棋子）
        if (x < sampleRadius || x >= warped.cols - sampleRadius ||
            y < sampleRadius || y >= warped.rows - sampleRadius) {
          samples.add(_IntersectionSample(
            row: r, col: c, avgV: 160.0, avgS: 120.0, stdV: 0.0,
          ));
          continue;
        }

        var totalV = 0.0;
        var totalS = 0.0;
        var totalV2 = 0.0;
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
          row: r, col: c,
          avgV: avgV, avgS: avgS, stdV: stdV,
        ));
      }
    }
    hsv.dispose();

    // === V/S 值統計 ===
    final vValues = samples.map((s) => s.avgV).toList();
    final sValues = samples.map((s) => s.avgS).toList();
    final sortedV = List<double>.from(vValues)..sort();
    final sortedS = List<double>.from(sValues)..sort();
    debug.vMin = sortedV.first;
    debug.vMax = sortedV.last;
    final boardMedianS = sortedS[sortedS.length ~/ 2];

    // === k-means 找三群 + 安全上下限 ===
    final centers = _kMeans1D(vValues, 3);
    debug.clusterCenters = List.from(centers);

    // k-means 中點作為閾值，加上安全範圍
    var thresholdBB = (centers[0] + centers[1]) / 2;
    var thresholdBW = (centers[1] + centers[2]) / 2;
    thresholdBB = min(thresholdBB, 110.0);
    thresholdBW = max(thresholdBW, 170.0);
    debug.thresholdBlackBoard = thresholdBB;
    debug.thresholdBoardWhite = thresholdBW;

    // 棋子（黑白皆）為無彩色，飽和度遠低於棋盤（暖色木板）
    final satLimitBlack = min(boardMedianS * 0.7, 80.0);
    final satLimitWhite = min(boardMedianS * 0.5, 60.0);
    debug.satLimitBlack = satLimitBlack;
    debug.satLimitWhite = satLimitWhite;

    // === 分類 ===
    for (final sample in samples) {
      if (sample.avgV < thresholdBB && sample.avgS < satLimitBlack) {
        grid[sample.row][sample.col] = StoneColor.black;
        debug.blackCount++;
      } else if (sample.avgV > thresholdBW && sample.avgS < satLimitWhite) {
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

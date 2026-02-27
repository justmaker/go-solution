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
      // 除錯：儲存 warped 影像到外部儲存（方便離線分析）
      // 1. 偵測棋盤邊界並進行透視校正
      final warped = _findAndWarpBoard(img);

      // 儲存 warped 供除錯（release 也存，方便 adb pull 離線分析）
      try {
        cv.imwrite('/data/local/tmp/debug_warped.jpg', warped);
      } catch (_) {}

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
    if (kDebugMode) {
      debugPrint(
          '[BoardRecognition] 原始影像: ${original.cols}x${original.rows}');
    }

    // 方法 A：顏色偵測 — 嚴格範圍（實拍棋盤）
    var warped = _findBoardByColor(original, 12, 35, 50, 100);
    if (warped != null) {
      print('[BoardRecognition] 板面偵測: 顏色 (Strict), warped=${warped.cols}x${warped.rows}');
      return warped;
    }

    // 方法 A2：顏色偵測 — 寬鬆範圍（截圖/螢幕翻拍，色彩較淡）
    warped = _findBoardByColor(original, 8, 42, 15, 50);
    if (warped != null) {
      print('[BoardRecognition] 板面偵測: 顏色 (Loose), warped=${warped.cols}x${warped.rows}');
      return warped;
    }

    // 方法 B：增強型邊緣偵測（Canny + Dilate + Convex Hull）
    final gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (5, 5), 1.0);
    final edges = cv.canny(blurred, 30, 150);

    // Dilate (連接斷裂邊緣)
    final kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
    final dilated = cv.dilate(edges, kernel, iterations: 2);
    kernel.dispose();

    var edgeWarped = _findBoardByContours(dilated, original);
    if (edgeWarped != null) {
      print('[BoardRecognition] 板面偵測: 輪廓 (Convex Hull), warped=${edgeWarped.cols}x${edgeWarped.rows}');
      gray.dispose();
      blurred.dispose();
      edges.dispose();
      dilated.dispose();
      return edgeWarped;
    }

    // 方法 C：Hough Lines (長邊偵測)
    var houghWarped = _findBoardByHoughLines(dilated, original);

    gray.dispose();
    blurred.dispose();
    edges.dispose();
    dilated.dispose();

    if (houghWarped != null) {
      print('[BoardRecognition] 板面偵測: Hough Lines, warped=${houghWarped.cols}x${houghWarped.rows}');
      return houghWarped;
    }

    print('[BoardRecognition] 板面偵測: 全部失敗，使用原圖 ${original.cols}x${original.rows}');
    return original.clone();
  }

  /// 顏色偵測：用 HSV 過濾暖色木板，找最大連通區域
  cv.Mat? _findBoardByColor(cv.Mat original, int hLow, int hHigh, int sLow, int vLow) {
    final hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV);

    final lower = cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [hLow, sLow, vLow]);
    final upper = cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [hHigh, 255, 255]);
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

    if (contours.isEmpty) {
      if (kDebugMode) {
        debugPrint(
            '[BoardRecognition] 顏色偵測(H=$hLow-$hHigh S>=$sLow V>=$vLow): 無輪廓');
      }
      return null;
    }

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

    // 面積至少佔影像 10%
    final totalArea = original.rows * original.cols;
    final ratio = maxArea / totalArea;
    if (bestContour == null || ratio < 0.10) {
      if (kDebugMode) {
        debugPrint(
            '[BoardRecognition] 顏色偵測(H=$hLow-$hHigh S>=$sLow V>=$vLow): 面積不足 ${(ratio * 100).toStringAsFixed(1)}%');
      }
      return null;
    }

    // 取最小面積矩形
    final rect = cv.minAreaRect(bestContour);
    final boxPoints = cv.boxPoints(rect);
    final corners = _orderPoints2f(boxPoints);

    return _warpPerspective(original, corners);
  }

  /// 輪廓偵測：找最大凸包 (Convex Hull)
  cv.Mat? _findBoardByContours(cv.Mat processed, cv.Mat original) {
    final (contours, hierarchy) = cv.findContours(
      processed,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );

    cv.VecPoint? bestCnt;
    double maxArea = 0;

    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area < original.rows * original.cols * 0.1) continue;

      // 使用 Convex Hull 忽略邊緣凹陷 (如被手指或棋子遮擋)
      final hullMat = cv.convexHull(contour, returnPoints: true);
      final hull = cv.VecPoint.fromMat(hullMat);
      hullMat.dispose();

      final peri = cv.arcLength(hull, true);
      final approx = cv.approxPolyDP(hull, 0.02 * peri, true);
      hull.dispose();

      if (approx.length == 4 && area > maxArea) {
        bestCnt?.dispose();
        maxArea = area;
        bestCnt = approx;
      } else {
        approx.dispose();
      }
    }

    // Dispose contours and hierarchy
    contours.dispose();
    hierarchy.dispose();

    if (bestCnt == null) return null;

    final corners = _orderPoints(bestCnt);
    bestCnt.dispose(); // Dispose after usage

    return _warpPerspective(original, corners.map((p) => cv.Point2f(p.x.toDouble(), p.y.toDouble())).toList());
  }

  /// Hough Lines 偵測
  cv.Mat? _findBoardByHoughLines(cv.Mat edges, cv.Mat original) {
    final linesMat = cv.HoughLinesP(edges, 1, pi / 180, 50,
        minLineLength: original.cols * 0.2, maxLineGap: 20);

    if (linesMat.rows < 4) {
      linesMat.dispose();
      return null;
    }

    final horizontals = <cv.Vec4i>[];
    final verticals = <cv.Vec4i>[];

    for (int i = 0; i < linesMat.rows; i++) {
      final line = linesMat.at<cv.Vec4i>(i, 0);
      final p1 = cv.Point(line.val1, line.val2);
      final p2 = cv.Point(line.val3, line.val4);

      final dx = (p2.x - p1.x).abs();
      final dy = (p2.y - p1.y).abs();

      if (dx == 0) {
        verticals.add(line);
        continue;
      }

      final slope = dy / dx;
      if (slope < 0.5)
        horizontals.add(line);
      else if (slope > 2.0) verticals.add(line);
    }
    linesMat.dispose();

    if (horizontals.length < 2 || verticals.length < 2) return null;

    horizontals.sort((a, b) =>
        ((a.val2 + a.val4) / 2).compareTo((b.val2 + b.val4) / 2));
    final top = horizontals.first;
    final bottom = horizontals.last;

    verticals.sort((a, b) =>
        ((a.val1 + a.val3) / 2).compareTo((b.val1 + b.val3) / 2));
    final left = verticals.first;
    final right = verticals.last;

    cv.Point2f? intersection(cv.Vec4i l1, cv.Vec4i l2) {
      final x1 = l1.val1.toDouble();
      final y1 = l1.val2.toDouble();
      final x2 = l1.val3.toDouble();
      final y2 = l1.val4.toDouble();
      final x3 = l2.val1.toDouble();
      final y3 = l2.val2.toDouble();
      final x4 = l2.val3.toDouble();
      final y4 = l2.val4.toDouble();

      final d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
      if (d == 0) return null;

      final px =
          ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
              d;
      final py =
          ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
              d;
      return cv.Point2f(px, py);
    }

    final tl = intersection(top, left);
    final tr = intersection(top, right);
    final bl = intersection(bottom, left);
    final br = intersection(bottom, right);

    if (tl == null || tr == null || bl == null || br == null) return null;

    return _warpPerspective(original, [tl, tr, bl, br]);
  }

  cv.Mat? _warpPerspective(cv.Mat original, List<cv.Point2f> corners) {
    // Note: corners need to be converted to VecPoint2f for ordering if needed,
    // but _orderPoints2f takes VecPoint2f.
    // However, creating VecPoint2f from list allocates memory.
    final cornersVec = cv.VecPoint2f.fromList(corners);
    final sorted = _orderPoints2f(cornersVec);
    cornersVec.dispose(); // Dispose input vector

    final w = max(
      _distance2f(sorted[0], sorted[1]),
      _distance2f(sorted[2], sorted[3]),
    ).toInt();
    final h = max(
      _distance2f(sorted[0], sorted[3]),
      _distance2f(sorted[1], sorted[2]),
    ).toInt();
    final size = max(w, h);

    if (size < 100) return null;

    final srcPoints = cv.VecPoint2f.fromList(sorted);
    final dstPoints = cv.VecPoint2f.fromList([
      cv.Point2f(0, 0),
      cv.Point2f(size.toDouble() - 1, 0),
      cv.Point2f(size.toDouble() - 1, size.toDouble() - 1),
      cv.Point2f(0, size.toDouble() - 1),
    ]);

    final matrix = cv.getPerspectiveTransform2f(srcPoints, dstPoints);
    final warped = cv.warpPerspective(original, matrix, (size, size));

    matrix.dispose();
    srcPoints.dispose();
    dstPoints.dispose();

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

    // 1. 初步 Hough 偵測用於計算旋轉校正
    var gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
    var blurred = cv.gaussianBlur(gray, (3, 3), 0);
    var edges = cv.canny(blurred, 50, 150);

    // 寬鬆參數以捕捉更多線段
    var linesMat = cv.HoughLinesP(
      edges, 1, pi / 180,
      50,
      minLineLength: warped.cols * 0.1, // 降低最小長度
      maxLineGap: warped.cols / 10,     // 增加最大間隙
    );

    // 計算旋轉角度
    final angles = <double>[];
    if (linesMat.rows > 0) {
      for (int i = 0; i < linesMat.rows; i++) {
        final line = linesMat.at<cv.Vec4i>(i, 0);
        final x1 = line.val1.toDouble();
        final y1 = line.val2.toDouble();
        final x2 = line.val3.toDouble();
        final y2 = line.val4.toDouble();
        final dx = x2 - x1;
        final dy = y2 - y1;
        if (dx == 0) continue;
        final angle = atan(dy / dx);

        // 收集接近水平或垂直的角度偏差
        if (angle.abs() < pi / 6) {
          angles.add(angle);
        } else if (angle.abs() > pi / 3) {
          if (angle > 0) {
            angles.add(angle - pi / 2);
          } else {
            angles.add(angle + pi / 2);
          }
        }
      }
    }
    linesMat.dispose();

    // 應用旋轉校正
    if (angles.isNotEmpty) {
      angles.sort();
      final medianAngle = angles[angles.length ~/ 2];
      if (medianAngle.abs() > 0.005) {
        // > 0.3度才校正
        if (kDebugMode) {
          debugPrint(
              '[BoardRecognition] 應用旋轉校正: ${(medianAngle * 180 / pi).toStringAsFixed(2)}度');
        }
        final center = cv.Point2f(size / 2, size / 2);
        final rotMat = cv.getRotationMatrix2D(center, medianAngle * 180 / pi, 1.0);
        final rotated = cv.warpAffine(warped, rotMat, (size, size), borderMode: cv.BORDER_REPLICATE);

        // 更新 warped (注意：這裡不能直接替換 warped 引用，因為是參數，但我們可以操作內容或更新局部變數)
        // 這裡我們更新 gray/blurred/edges 用於後續步驟
        warped.copyTo(warped); // 保持 warped 不變? 不，我们需要后续步骤使用校正后的图像
        // Dart参数传递是引用传递，但这就意味着外部的 warped 不会被修改，这很好。
        // 但我们需要修改 warped 对应的处理结果

        // 為了簡單起見，我們更新 gray/blurred/edges，並在後續使用 rotWarped 如果需要
        // 實際上，後續步驟 `_detectStones` 需要校正後的 `warped`。
        // 所以我們必須更新 `warped` 變數。
        // 但 Dart 中參數是 final 的... 不，這裡 `cv.Mat warped` 不是 final。
        // 但我們無法修改調用者的 `warped`。
        // 不過 `_detectGridLines` 返回 intersections。
        // 如果我們在這裡旋轉了，返回的 intersection 座標是相對於旋轉後的影像。
        // 但外部的 `warped` 還是舊的。這會導致 `_detectStones` 取樣錯誤！

        // 解決方案：將旋轉邏輯移到 `_detectGridLines` 之外，或讓 `_detectGridLines` 返回校正後的 Mat。
        // 考慮到代碼結構，我們可以在這裡進行校正，並將 warped 的內容替換（copyTo）。
        rotated.copyTo(warped); //這會修改外部 warped 物件的內容！

        rotMat.dispose();
        rotated.dispose();

        // 重新計算特徵
        gray.dispose();
        blurred.dispose();
        edges.dispose();
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
        blurred = cv.gaussianBlur(gray, (3, 3), 0);
        edges = cv.canny(blurred, 50, 150);
      }
    }

    // === 正式 Hough 線偵測 ===
    linesMat = cv.HoughLinesP(
      edges, 1, pi / 180,
      50,
      minLineLength: warped.cols * 0.1,
      maxLineGap: warped.cols / 10,
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

    // === 過濾影像邊緣位置（座標標籤通常在外側 3-5%）===
    final edgeMargin = size * 0.05;
    final hFiltered = hCombined.where((p) => p >= edgeMargin && p <= size - edgeMargin).toList();
    final vFiltered = vCombined.where((p) => p >= edgeMargin && p <= size - edgeMargin).toList();

    if (kDebugMode) {
      debugPrint('[BoardRecognition] Warped size: ${size}x$size');
      debugPrint(
          '[BoardRecognition] Hough: H=${hClusters.length}, V=${vClusters.length}');
      debugPrint('[BoardRecognition] Dips: H=${hDips.length}, V=${vDips.length}');
      debugPrint(
          '[BoardRecognition] Combined: H=${hCombined.length}, V=${vCombined.length}');
      debugPrint(
          '[BoardRecognition] Edge filtered (margin=${edgeMargin.toStringAsFixed(1)}): H=${hFiltered.length}, V=${vFiltered.length}');
      // 列出過濾後位置的連續差值，幫助診斷
      if (hFiltered.length >= 2) {
        final hDiffs = <String>[];
        for (int i = 0; i < hFiltered.length - 1; i++) {
          hDiffs.add((hFiltered[i + 1] - hFiltered[i]).toStringAsFixed(0));
        }
        debugPrint('[BoardRecognition] H diffs: $hDiffs');
      }
      if (vFiltered.length >= 2) {
        final vDiffs = <String>[];
        for (int i = 0; i < vFiltered.length - 1; i++) {
          vDiffs.add((vFiltered[i + 1] - vFiltered[i]).toStringAsFixed(0));
        }
        debugPrint('[BoardRecognition] V diffs: $vDiffs');
      }
    }

    // === 暴力搜尋 H/V 各自的最佳間距和相位 ===
    var (hSpacing, hPhase, hInl) =
        _findBestSpacing(hFiltered, size.toDouble());
    var (vSpacing, vPhase, vInl) =
        _findBestSpacing(vFiltered, size.toDouble());

    // Cross-validate: warped 是正方形，H/V 間距應相近
    // 若差異超過 15%，用 inlier 較多的方向修正較弱的方向
    final spacingDiff = (hSpacing - vSpacing).abs() / max(hSpacing, vSpacing);
    if (spacingDiff > 0.15) {
      if (hInl >= vInl) {
        final (newPhase, newInl) = _findBestPhase(vFiltered, hSpacing);
        vSpacing = hSpacing;
        vPhase = newPhase;
        vInl = newInl;
        print('[BoardRecognition] V 間距修正: 使用 H spacing $hSpacing (V inl: $newInl)');
      } else {
        final (newPhase, newInl) = _findBestPhase(hFiltered, vSpacing);
        hSpacing = vSpacing;
        hPhase = newPhase;
        hInl = newInl;
        print('[BoardRecognition] H 間距修正: 使用 V spacing $vSpacing (H inl: $newInl)');
      }
    }

    // 從最佳間距生成格線
    final hLines =
        _generateGridFromSpacing(hPhase, hSpacing, size.toDouble());
    final vLines =
        _generateGridFromSpacing(vPhase, vSpacing, size.toDouble());

    // 決定棋盤大小：用 spacing ratio（imageSize / spacing）最穩定
    // 9x9: ratio≈10, 13x13: ratio≈14, 19x19: ratio≈20
    final bestSpacing = hInl >= vInl ? hSpacing : vSpacing;
    final sizeRatio = size / bestSpacing;
    final int boardSize;
    if (sizeRatio < 11.5) {
      boardSize = 9;
    } else if (sizeRatio < 17) {
      boardSize = 13;
    } else {
      boardSize = 19;
    }

    // 修剪/擴展到棋盤大小
    final hTrimmed =
        _trimToSize(hLines, boardSize, hSpacing, size.toDouble());
    final vTrimmed =
        _trimToSize(vLines, boardSize, vSpacing, size.toDouble());

    // 用投影 profile 逐線精修位置，消除間距累積誤差
    final hFinal = _refineToProjection(hTrimmed, hSmooth, hSpacing);
    final vFinal = _refineToProjection(vTrimmed, vSmooth, vSpacing);

    final intersections = <List<cv.Point2f>>[];
    for (int r = 0; r < boardSize; r++) {
      final row = <cv.Point2f>[];
      for (int c = 0; c < boardSize; c++) {
        row.add(cv.Point2f(vFinal[c], hFinal[r]));
      }
      intersections.add(row);
    }

    // Release 可見的關鍵日誌
    print('[BoardRecognition] 格線偵測: combined H=${hCombined.length}/V=${vCombined.length}'
        ' → filtered H=${hFiltered.length}/V=${vFiltered.length} (margin=${edgeMargin.toStringAsFixed(0)})');
    print('[BoardRecognition] 間距: H=${hSpacing.toStringAsFixed(1)} (inl=$hInl), V=${vSpacing.toStringAsFixed(1)} (inl=$vInl)');
    print('[BoardRecognition] 格線: ${hLines.length}x${vLines.length}, ratio=${sizeRatio.toStringAsFixed(1)} → ${boardSize}x$boardSize');
    print('[BoardRecognition] H trimmed→refined: ${hTrimmed.map((p) => p.toStringAsFixed(0)).join(",")} → ${hFinal.map((p) => p.toStringAsFixed(0)).join(",")}');
    print('[BoardRecognition] V trimmed→refined: ${vTrimmed.map((p) => p.toStringAsFixed(0)).join(",")} → ${vFinal.map((p) => p.toStringAsFixed(0)).join(",")}');

    if (kDebugMode) {
      debugPrint(
          '[BoardRecognition] H diffs after spacing: ${hLines.length >= 2 ? List.generate(hLines.length - 1, (i) => (hLines[i + 1] - hLines[i]).toStringAsFixed(0)) : "N/A"}');
    }
    return (boardSize, intersections);
  }

  /// 對給定間距找最佳相位（用於 cross-validation 修正）
  (double, int) _findBestPhase(List<double> positions, double spacing) {
    final tolerance = spacing * 0.12;
    var bestPhase = 0.0;
    var bestInliers = 0;
    for (final ref in positions) {
      final phase = ref % spacing;
      var inliers = 0;
      for (final p in positions) {
        var remainder = (p - phase) % spacing;
        if (remainder > spacing / 2) remainder = spacing - remainder;
        if (remainder < tolerance) inliers++;
      }
      if (inliers > bestInliers) {
        bestInliers = inliers;
        bestPhase = phase;
      }
    }
    return (bestPhase, bestInliers);
  }

  /// 暴力搜尋最佳間距：對每個候選間距，找最佳相位，用 inliers * sqrt(sp) 評分
  /// sqrt(sp) 權重適度偏好較大間距，自然避免 harmonic 半間距問題
  (double, double, int) _findBestSpacing(List<double> positions, double totalSize) {
    if (positions.length < 3) {
      return (totalSize / 14, totalSize * 0.05, 0);
    }

    final minSp = (totalSize * 0.04).toInt(); // ~19 路最小間距
    final maxSp = (totalSize * 0.12).toInt(); // ~9 路最大間距

    var bestSpacing = totalSize / 14;
    var bestPhase = 0.0;
    var bestScore = 0.0;

    for (int sp = minSp; sp <= maxSp; sp++) {
      final tolerance = sp * 0.12;
      final spSqrt = sqrt(sp.toDouble());

      // 對標準棋盤尺寸給予加分
      final ratio = totalSize / sp;
      var scoreMultiplier = 1.0;
      if (ratio >= 18 && ratio <= 20) { // 19x19
        scoreMultiplier = 1.2;
      } else if (ratio >= 12 && ratio <= 14) { // 13x13
        scoreMultiplier = 1.1;
      } else if (ratio >= 8 && ratio <= 10) { // 9x9
        scoreMultiplier = 1.1;
      }

      for (final ref in positions) {
        final phase = ref % sp;
        var inliers = 0;
        for (final p in positions) {
          var remainder = (p - phase) % sp;
          if (remainder > sp / 2) remainder = sp - remainder;
          if (remainder < tolerance) inliers++;
        }
        var score = inliers * spSqrt * scoreMultiplier;

        if (score > bestScore) {
          bestScore = score;
          bestSpacing = sp.toDouble();
          bestPhase = phase;
        }
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

  /// 用投影 profile 做全局相位精修：搜尋使所有格線的投影值加總最小的偏移量
  /// 保持均勻間距，避免 per-line 精修受棋子 dip 干擾
  List<double> _refineToProjection(
      List<double> positions, List<double> profile, double spacing) {
    final searchRadius = (spacing * 0.25).toInt();
    var bestShift = 0;
    var bestScore = double.infinity;
    for (int shift = -searchRadius; shift <= searchRadius; shift++) {
      var score = 0.0;
      for (final pos in positions) {
        final idx = (pos + shift).round().clamp(0, profile.length - 1);
        score += profile[idx];
      }
      if (score < bestScore) {
        bestScore = score;
        bestShift = shift;
      }
    }
    return positions.map((p) => p + bestShift).toList();
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

    // 取樣半徑用 0.35 倍格距：比 0.2 更容易覆蓋稍有偏移的棋子，
    // 同時星位小點在大面積中被稀釋，不易誤判
    final sampleRadius = (warped.cols / boardSize * 0.35).round();
    final samples = <_IntersectionSample>[];

    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        final pt = intersections[r][c];
        final x = pt.x.round();
        final y = pt.y.round();

        // 超出影像邊界的交叉點標記為無效（-1.0），統計時排除，分類時視為 Empty
        if (x < sampleRadius || x >= warped.cols - sampleRadius ||
            y < sampleRadius || y >= warped.rows - sampleRadius) {
          samples.add(_IntersectionSample(
            row: r, col: c, avgV: -1.0, avgS: 0.0, stdV: 0.0,
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

    // === V/S 值統計 (排除無效樣本) ===
    final validSamples = samples.where((s) => s.avgV >= 0).toList();
    if (validSamples.isEmpty) return grid; // Should not happen

    final vValues = validSamples.map((s) => s.avgV).toList();
    final sValues = validSamples.map((s) => s.avgS).toList();

    // 簡單統計
    var minV = 255.0;
    var maxV = 0.0;
    for(final v in vValues) {
      if(v < minV) minV = v;
      if(v > maxV) maxV = v;
    }
    debug.vMin = minV;
    debug.vMax = maxV;

    final sortedS = List<double>.from(sValues)..sort();
    final boardMedianS = sortedS.isNotEmpty ? sortedS[sortedS.length ~/ 2] : 0.0;

    // === Mode-based 自適應閾值 ===
    // 計算 V 直方圖找主峰 (Mode)，假設主峰為棋盤顏色
    const binSize = 8;
    final bins = List.filled(256 ~/ binSize + 1, 0);
    for (final v in vValues) {
       bins[v.toInt() ~/ binSize]++;
    }
    var maxBin = 0;
    var maxCount = 0;
    for (int i = 0; i < bins.length; i++) {
        if (bins[i] > maxCount) {
            maxCount = bins[i];
            maxBin = i;
        }
    }
    final modeV = (maxBin * binSize + binSize / 2).toDouble();
    debug.clusterCenters = [modeV]; // 借用欄位顯示 Mode

    // 基於 Mode 設定黑白閾值
    // 黑子顯著暗於棋盤，白子顯著亮於棋盤
    var thresholdBB = modeV - 60.0;
    var thresholdBW = modeV + 40.0;

    // 安全鉗位：避免閾值過於極端
    thresholdBB = max(thresholdBB, 50.0);
    thresholdBW = min(thresholdBW, 220.0);

    debug.thresholdBlackBoard = thresholdBB;
    debug.thresholdBoardWhite = thresholdBW;

    // 黑子飽和度限制：數位棋盤的黑子有暖色渲染，S 可能等於甚至高於棋盤
    // 黑子已由低 V 值強力識別，飽和度只需排除極端彩色區域
    final satLimitBlack = max(boardMedianS * 1.5, 50.0);
    // 白子飽和度限制：白子通常低飽和度，需排除高飽和度的亮色區域
    final satLimitWhite = max(min(boardMedianS * 0.8, 60.0), 25.0);
    debug.satLimitBlack = satLimitBlack;
    debug.satLimitWhite = satLimitWhite;

    // === 分類 ===
    var darkCount = 0; // V < thresholdBB 的樣本數（含黑子候選）
    var darkRejectedBySat = 0; // 暗但被飽和度拒絕
    double darkMinS = 999, darkMaxS = 0;
    for (final sample in samples) {
      if (sample.avgV < 0) {
        // 無效樣本（出界）視為空
        debug.emptyCount++;
        continue;
      }

      if (sample.avgV < thresholdBB) {
        darkCount++;
        if (sample.avgS < darkMinS) darkMinS = sample.avgS;
        if (sample.avgS > darkMaxS) darkMaxS = sample.avgS;
        if (sample.avgS < satLimitBlack) {
          grid[sample.row][sample.col] = StoneColor.black;
          debug.blackCount++;
        } else {
          darkRejectedBySat++;
          debug.emptyCount++;
        }
      } else if (sample.avgV > thresholdBW && sample.avgS < satLimitWhite) {
        grid[sample.row][sample.col] = StoneColor.white;
        debug.whiteCount++;
      } else {
        debug.emptyCount++;
      }
    }

    // Release 可見的棋子偵測摘要
    print('[BoardRecognition] 棋子: 黑=${debug.blackCount}, 白=${debug.whiteCount}, 空=${debug.emptyCount} '
        '(modeV=${modeV.toStringAsFixed(0)}, medS=${boardMedianS.toStringAsFixed(0)}, '
        'BB<${thresholdBB.toStringAsFixed(0)}, BW>${thresholdBW.toStringAsFixed(0)}, '
        'satB<${satLimitBlack.toStringAsFixed(0)}, satW<${satLimitWhite.toStringAsFixed(0)})');
    print('[BoardRecognition] 暗位置: $darkCount個 (V<${thresholdBB.toStringAsFixed(0)}), '
        'S範圍=${darkMinS.toStringAsFixed(1)}~${darkMaxS.toStringAsFixed(1)}, '
        '被S拒絕=$darkRejectedBySat');

    if (kDebugMode) {
      lastDebugInfo = debug;
      debugPrint(debug.toString());
    }
    return grid;
  }
}

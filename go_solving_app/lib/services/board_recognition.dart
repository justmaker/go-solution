import 'dart:math';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../models/board_state.dart';

/// OpenCV 棋盤辨識服務
class BoardRecognition {
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

      // 4. 偵測格線並推斷棋盤大小
      final (boardSize, intersections) = _detectGridLines(warped);

      // 5. 在每個交叉點偵測棋子
      final grid = _detectStones(warped, boardSize, intersections);

      return BoardState(boardSize: boardSize, grid: grid);
    } finally {
      img.dispose();
    }
  }

  /// 影像前處理：灰階、模糊、CLAHE 對比增強
  cv.Mat _preprocess(cv.Mat img) {
    final gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (5, 5), sigmaX: 1.0);
    final clahe = cv.CLAHE(clipLimit: 2.0, tileGridSize: (8, 8));
    final enhanced = clahe.apply(blurred);
    gray.dispose();
    blurred.dispose();
    return enhanced;
  }

  /// 偵測棋盤邊界並進行四點透視校正
  cv.Mat _findAndWarpBoard(cv.Mat processed, cv.Mat original) {
    // 使用 Canny 邊緣偵測
    final edges = cv.canny(processed, 50, 150);

    // 尋找輪廓
    final (contours, _) = cv.findContours(
      edges,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    edges.dispose();

    if (contours.isEmpty) {
      // 若找不到輪廓，直接使用原圖
      return original.clone();
    }

    // 找到面積最大的四邊形輪廓
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

    // 排序四個角點（左上、右上、右下、左下）
    final corners = _orderPoints(bestContour);

    // 計算目標尺寸
    final width = max(
      _distance(corners[0], corners[1]),
      _distance(corners[2], corners[3]),
    ).toInt();
    final height = max(
      _distance(corners[0], corners[3]),
      _distance(corners[1], corners[2]),
    ).toInt();

    final size = max(width, height);

    // 建立目標點
    final dst = cv.VecPoint.fromList([
      cv.Point(0, 0),
      cv.Point(size - 1, 0),
      cv.Point(size - 1, size - 1),
      cv.Point(0, size - 1),
    ]);

    // 透視變換
    final srcPoints = corners.map((p) => cv.Point2f(p.x.toDouble(), p.y.toDouble())).toList();
    final dstPoints = dst.toList().map((p) => cv.Point2f(p.x.toDouble(), p.y.toDouble())).toList();

    final matrix = cv.getPerspectiveTransform(
      cv.VecPoint2f.fromList(srcPoints),
      cv.VecPoint2f.fromList(dstPoints),
    );

    final warped = cv.warpPerspective(
      original,
      matrix,
      (size, size),
    );

    matrix.dispose();
    return warped;
  }

  /// 排序四個角點為：左上、右上、右下、左下
  List<cv.Point> _orderPoints(cv.VecPoint points) {
    final pts = points.toList();

    // 按 x+y 排序，最小的是左上，最大的是右下
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];

    // 按 y-x 排序，最小的是右上，最大的是左下
    pts.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    final tr = pts[0];
    final bl = pts[3];

    return [tl, tr, br, bl];
  }

  double _distance(cv.Point a, cv.Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2).toDouble());
  }

  /// 使用霍夫線變換偵測格線，自動推斷棋盤大小
  (int, List<List<cv.Point2f>>) _detectGridLines(cv.Mat warped) {
    final gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (3, 3), sigmaX: 0);
    final edges = cv.canny(blurred, 50, 150);

    // 霍夫線變換
    final lines = cv.HoughLinesP(
      edges,
      1,
      pi / 180,
      80,
      minLineLength: warped.cols * 0.3,
      maxLineGap: warped.cols ~/ 20,
    );

    gray.dispose();
    blurred.dispose();
    edges.dispose();

    // 分離水平線和垂直線
    final horizontalYs = <double>[];
    final verticalXs = <double>[];

    for (final line in lines.toList()) {
      final x1 = line.val1.toDouble();
      final y1 = line.val2.toDouble();
      final x2 = line.val3.toDouble();
      final y2 = line.val4.toDouble();

      final angle = atan2((y2 - y1).abs(), (x2 - x1).abs());
      if (angle < pi / 6) {
        // 接近水平
        horizontalYs.add((y1 + y2) / 2);
      } else if (angle > pi / 3) {
        // 接近垂直
        verticalXs.add((x1 + x2) / 2);
      }
    }

    // 聚類分析找出 N 條線
    final hClusters = _clusterValues(horizontalYs, warped.rows * 0.02);
    final vClusters = _clusterValues(verticalXs, warped.cols * 0.02);

    // 自動偵測棋盤大小
    final detectedSize = ((hClusters.length + vClusters.length) / 2).round();
    final int boardSize;
    if (detectedSize <= 11) {
      boardSize = 9;
    } else if (detectedSize <= 16) {
      boardSize = 13;
    } else {
      boardSize = 19;
    }

    // 計算交叉點
    hClusters.sort();
    vClusters.sort();

    // 若偵測到的線數不足，均勻分佈
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

  /// 聚類分析：將接近的值合併為一個群
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

  /// 均勻分佈位置
  List<double> _generateUniformPositions(int count, double totalSize) {
    final margin = totalSize * 0.05;
    final step = (totalSize - 2 * margin) / (count - 1);
    return List.generate(count, (i) => margin + i * step);
  }

  /// 在每個交叉點取樣 HSV 色彩分析偵測棋子
  List<List<StoneColor>> _detectStones(
    cv.Mat warped,
    int boardSize,
    List<List<cv.Point2f>> intersections,
  ) {
    final hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV);
    final grid = List.generate(
      boardSize,
      (_) => List.filled(boardSize, StoneColor.empty),
    );

    final sampleRadius = (warped.cols / boardSize * 0.2).round();

    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        final pt = intersections[r][c];
        final x = pt.x.round().clamp(0, warped.cols - 1);
        final y = pt.y.round().clamp(0, warped.rows - 1);

        // 取樣周圍像素的平均值
        var totalV = 0.0;
        var totalS = 0.0;
        var sampleCount = 0;

        for (int dy = -sampleRadius; dy <= sampleRadius; dy++) {
          for (int dx = -sampleRadius; dx <= sampleRadius; dx++) {
            final sx = (x + dx).clamp(0, warped.cols - 1);
            final sy = (y + dy).clamp(0, warped.rows - 1);
            final pixel = hsv.at<cv.Vec3b>(sy, sx);
            totalS += pixel.val2; // Saturation
            totalV += pixel.val3; // Value
            sampleCount++;
          }
        }

        final avgV = totalV / sampleCount;
        final avgS = totalS / sampleCount;

        // 根據亮度和飽和度判斷棋子顏色
        if (avgV < 80) {
          grid[r][c] = StoneColor.black;
        } else if (avgV > 180 && avgS < 40) {
          grid[r][c] = StoneColor.white;
        }
        // 否則為空位
      }
    }

    hsv.dispose();
    return grid;
  }
}

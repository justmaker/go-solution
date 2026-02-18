import 'package:flutter/material.dart';
import '../models/board_state.dart';
import '../models/analysis_result.dart';

/// 棋盤繪製 Widget
class BoardPainterWidget extends StatelessWidget {
  final BoardState boardState;
  final AnalysisResult? analysisResult;

  const BoardPainterWidget({
    super.key,
    required this.boardState,
    this.analysisResult,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.maxWidth < constraints.maxHeight
            ? constraints.maxWidth
            : constraints.maxHeight;
        return Center(
          child: SizedBox(
            width: size,
            height: size,
            child: CustomPaint(
              painter: _BoardCustomPainter(
                boardState: boardState,
                analysisResult: analysisResult,
              ),
            ),
          ),
        );
      },
    );
  }
}

class _BoardCustomPainter extends CustomPainter {
  final BoardState boardState;
  final AnalysisResult? analysisResult;

  _BoardCustomPainter({
    required this.boardState,
    this.analysisResult,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final n = boardState.boardSize;
    final margin = size.width * 0.04;
    final boardWidth = size.width - 2 * margin;
    final cellSize = boardWidth / (n - 1);

    // 繪製背景
    final bgPaint = Paint()..color = const Color(0xFFDEB887);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), bgPaint);

    // 繪製格線
    final linePaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 1.0;

    for (int i = 0; i < n; i++) {
      final offset = margin + i * cellSize;
      // 水平線
      canvas.drawLine(
        Offset(margin, offset),
        Offset(size.width - margin, offset),
        linePaint,
      );
      // 垂直線
      canvas.drawLine(
        Offset(offset, margin),
        Offset(offset, size.height - margin),
        linePaint,
      );
    }

    // 繪製星位
    _drawStarPoints(canvas, n, margin, cellSize);

    // 繪製地域歸屬熱力圖
    if (analysisResult?.ownership != null) {
      _drawOwnership(canvas, n, margin, cellSize);
    }

    // 繪製棋子
    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        final stone = boardState.grid[r][c];
        if (stone != StoneColor.empty) {
          _drawStone(canvas, margin, cellSize, r, c, stone);
        }
      }
    }

    // 繪製最佳手建議
    if (analysisResult != null) {
      _drawMoveSuggestions(canvas, margin, cellSize);
    }
  }

  void _drawStarPoints(Canvas canvas, int n, double margin, double cellSize) {
    final starPaint = Paint()..color = Colors.black;
    final List<int> starPositions;
    if (n == 19) {
      starPositions = [3, 9, 15];
    } else if (n == 13) {
      starPositions = [3, 6, 9];
    } else if (n == 9) {
      starPositions = [2, 4, 6];
    } else {
      return;
    }

    for (final r in starPositions) {
      for (final c in starPositions) {
        canvas.drawCircle(
          Offset(margin + c * cellSize, margin + r * cellSize),
          cellSize * 0.1,
          starPaint,
        );
      }
    }
  }

  void _drawStone(
    Canvas canvas,
    double margin,
    double cellSize,
    int row,
    int col,
    StoneColor color,
  ) {
    final center = Offset(margin + col * cellSize, margin + row * cellSize);
    final radius = cellSize * 0.45;

    if (color == StoneColor.black) {
      final paint = Paint()..color = Colors.black;
      canvas.drawCircle(center, radius, paint);
    } else {
      final paint = Paint()..color = Colors.white;
      canvas.drawCircle(center, radius, paint);
      final borderPaint = Paint()
        ..color = Colors.black
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;
      canvas.drawCircle(center, radius, borderPaint);
    }
  }

  void _drawOwnership(
    Canvas canvas,
    int n,
    double margin,
    double cellSize,
  ) {
    final ownership = analysisResult!.ownership!;
    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        if (boardState.grid[r][c] != StoneColor.empty) continue;

        final value = ownership[r][c];
        if (value.abs() < 0.1) continue;

        final center = Offset(margin + c * cellSize, margin + r * cellSize);
        final rect = Rect.fromCenter(
          center: center,
          width: cellSize * 0.5,
          height: cellSize * 0.5,
        );
        final color = value > 0
            ? Colors.black.withValues(alpha: value.abs() * 0.3)
            : Colors.white.withValues(alpha: value.abs() * 0.3);
        canvas.drawRect(rect, Paint()..color = color);
      }
    }
  }

  void _drawMoveSuggestions(
    Canvas canvas,
    double margin,
    double cellSize,
  ) {
    final moves = analysisResult!.topMoves;
    final colors = [
      Colors.red,
      Colors.orange,
      Colors.yellow,
    ];

    for (int i = 0; i < moves.length && i < 3; i++) {
      final move = moves[i];
      final center = Offset(
        margin + move.position.col * cellSize,
        margin + move.position.row * cellSize,
      );
      final radius = cellSize * 0.35;

      // 繪製半透明圓
      final fillPaint = Paint()
        ..color = colors[i].withValues(alpha: 0.6);
      canvas.drawCircle(center, radius, fillPaint);

      // 繪製邊框
      final borderPaint = Paint()
        ..color = colors[i]
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;
      canvas.drawCircle(center, radius, borderPaint);

      // 繪製排名數字
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${i + 1}',
          style: TextStyle(
            color: Colors.white,
            fontSize: cellSize * 0.35,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(
          center.dx - textPainter.width / 2,
          center.dy - textPainter.height / 2,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BoardCustomPainter oldDelegate) {
    return oldDelegate.boardState != boardState ||
        oldDelegate.analysisResult != analysisResult;
  }
}

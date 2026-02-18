import 'package:flutter/material.dart';
import '../models/analysis_result.dart';

/// 最佳手建議顯示 Widget
class MoveSuggestionWidget extends StatelessWidget {
  final AnalysisResult result;

  const MoveSuggestionWidget({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    if (result.topMoves.isEmpty) {
      return const Padding(
        padding: EdgeInsets.all(8.0),
        child: Text('無建議著手'),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '最佳著手建議',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          ...result.topMoves.take(3).map((move) => _buildMoveRow(context, move)),
        ],
      ),
    );
  }

  Widget _buildMoveRow(BuildContext context, MoveSuggestion move) {
    final colors = [Colors.red, Colors.orange, Colors.yellow];
    final color = move.rank <= 3 ? colors[move.rank - 1] : Colors.grey;

    // 將座標轉換為圍棋標記法（A1 格式）
    final colLabel = String.fromCharCode(
      65 + move.position.col + (move.position.col >= 8 ? 1 : 0),
    );
    final rowLabel = result.boardSize - move.position.row;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color,
            ),
            child: Center(
              child: Text(
                '${move.rank}',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            '$colLabel$rowLabel',
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: LinearProgressIndicator(
              value: move.probability,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            move.probabilityText,
            style: const TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
}

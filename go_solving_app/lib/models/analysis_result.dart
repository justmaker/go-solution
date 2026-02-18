import 'board_state.dart';

/// 單一著手建議
class MoveSuggestion {
  final BoardPosition position;
  final double probability;
  final int rank;

  const MoveSuggestion({
    required this.position,
    required this.probability,
    required this.rank,
  });

  /// 格式化機率為百分比字串
  String get probabilityText => '${(probability * 100).toStringAsFixed(1)}%';

  @override
  String toString() =>
      'Move #$rank: $position ($probabilityText)';
}

/// KataGo 分析結果模型
class AnalysisResult {
  /// Top-N 最佳著手建議
  final List<MoveSuggestion> topMoves;

  /// 勝率 [黑勝, 白勝, 和棋]
  final List<double> winrate;

  /// 地域歸屬圖（每個交叉點的歸屬值，-1.0 = 白，+1.0 = 黑）
  final List<List<double>>? ownership;

  /// 分析時的棋盤大小
  final int boardSize;

  /// 分析時的下一手玩家
  final StoneColor nextPlayer;

  const AnalysisResult({
    required this.topMoves,
    required this.winrate,
    this.ownership,
    required this.boardSize,
    required this.nextPlayer,
  });

  /// 黑棋勝率
  double get blackWinrate => winrate.isNotEmpty ? winrate[0] : 0.0;

  /// 白棋勝率
  double get whiteWinrate => winrate.length > 1 ? winrate[1] : 0.0;

  /// 最佳著手
  MoveSuggestion? get bestMove => topMoves.isNotEmpty ? topMoves.first : null;

  /// 格式化勝率文字
  String get winrateText {
    final bw = (blackWinrate * 100).toStringAsFixed(1);
    final ww = (whiteWinrate * 100).toStringAsFixed(1);
    return '黑 $bw% / 白 $ww%';
  }

  @override
  String toString() {
    final sb = StringBuffer();
    sb.writeln('AnalysisResult (${boardSize}x$boardSize, next: $nextPlayer)');
    sb.writeln('Winrate: $winrateText');
    for (final move in topMoves) {
      sb.writeln('  $move');
    }
    return sb.toString();
  }
}

import 'dart:async';
import 'package:katago_onnx_mobile/katago_onnx_mobile.dart' as kg;
import '../models/board_state.dart';
import '../models/analysis_result.dart';

/// KataGo 引擎 wrapper，使用 katago_onnx_mobile 套件
class KataGoEngine {
  // 使用 Native Engine (C++ + ONNX Runtime)
  final kg.KataGoEngine _engine = kg.KataGoEngine();

  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// 初始化引擎
  Future<void> initialize() async {
    if (_isInitialized) return;

    // 初始化，使用 19x19 棋盤 (KataGo Native 會自動調整)
    // 注意：katago_onnx_mobile 會自動解壓縮內建模型
    final success = await _engine.start(boardSize: 19);
    if (!success) {
      throw Exception('Failed to initialize KataGo engine');
    }
    _isInitialized = true;
  }

  /// 分析棋盤狀態
  Future<AnalysisResult> analyze(BoardState board) async {
    if (!_isInitialized) {
      await initialize();
    }

    // 將 BoardState 轉換為 KataGo 需要的 moves 列表
    // 由於我們需要設置任意棋盤狀態，我們使用 "下子 + Pass" 的方式來放置每一顆棋子
    final moves = _convertBoardToMoves(board);

    // 執行分析
    // maxVisits 設定為 100，在手機上是一個合理的平衡點
    final result = await _engine.analyze(
      boardSize: board.boardSize,
      moves: moves,
      komi: board.komi,
      maxVisits: 100,
    );

    return _convertResult(result, board);
  }

  /// 將棋盤狀態轉換為 GTP moves 列表
  List<String> _convertBoardToMoves(BoardState board) {
    final moves = <String>[];

    // 目前模擬的下一手玩家，初始為黑棋 (空盤開始)
    StoneColor currentNext = StoneColor.black;

    // 遍歷棋盤，放置所有棋子
    for (int r = 0; r < board.boardSize; r++) {
      for (int c = 0; c < board.boardSize; c++) {
        final stone = board.grid[r][c];
        if (stone == StoneColor.empty) continue;

        final coord = _toGtpCoord(r, c, board.boardSize);

        if (stone == StoneColor.black) {
          // 如果當前輪到白棋，白棋先 Pass
          if (currentNext == StoneColor.white) {
            moves.add('W pass');
            currentNext = StoneColor.black;
          }
          // 黑棋下子
          moves.add('B $coord');
          currentNext = StoneColor.white;
        } else if (stone == StoneColor.white) {
          // 如果當前輪到黑棋，黑棋先 Pass
          if (currentNext == StoneColor.black) {
            moves.add('B pass');
            currentNext = StoneColor.white;
          }
          // 白棋下子
          moves.add('W $coord');
          currentNext = StoneColor.black;
        }
      }
    }

    // 調整最後的下一手玩家以符合 BoardState
    if (currentNext != board.nextPlayer) {
      if (currentNext == StoneColor.black) {
        moves.add('B pass');
      } else {
        moves.add('W pass');
      }
    }

    return moves;
  }

  /// 座標轉換：(row, col) -> GTP (e.g., "Q16")
  String _toGtpCoord(int r, int c, int size) {
    // Row 0 is top, GTP row 1 is bottom.
    final row = size - r;

    // Col 0 is A, skip I.
    // A=0, B=1, ..., H=7, J=8 (I skipped)
    final colIndex = c >= 8 ? c + 1 : c;
    final colChar = String.fromCharCode('A'.codeUnitAt(0) + colIndex);

    return '$colChar$row';
  }

  /// 座標轉換：GTP (e.g., "Q16") -> BoardPosition
  BoardPosition _fromGtpCoord(String gtp, int size) {
    final s = gtp.trim().toUpperCase();
    if (s == 'PASS') return BoardPosition(-1, -1);

    if (s.length < 2) return BoardPosition(-1, -1);

    final colChar = s.codeUnitAt(0);
    var col = colChar - 'A'.codeUnitAt(0);
    if (colChar >= 'I'.codeUnitAt(0)) col--; // Adjust for skipped I

    final rowStr = s.substring(1);
    final rowVal = int.tryParse(rowStr) ?? 0;
    final row = size - rowVal; // GTP 1 is bottom (size-1)

    return BoardPosition(row, col);
  }

  /// 將分析結果轉換為 App 使用的格式
  AnalysisResult _convertResult(kg.EngineAnalysisResult engineResult, BoardState board) {
    final totalVisits = engineResult.visits;

    // 轉換建議著手
    final moves = <MoveSuggestion>[];
    for (int i = 0; i < engineResult.topMoves.length; i++) {
      final m = engineResult.topMoves[i];
      if (m.move.toLowerCase() == 'pass') continue; // 忽略 Pass 建議? 或者保留? 通常保留

      final pos = _fromGtpCoord(m.move, board.boardSize);
      if (pos.row < 0 || pos.col < 0) continue;

      // 使用 visits 比例作為機率 (Policy 近似)
      final prob = totalVisits > 0 ? m.visits / totalVisits : 0.0;

      moves.add(MoveSuggestion(
        position: pos,
        probability: prob,
        rank: i + 1,
      ));
    }

    // 計算勝率
    // KataGo Native 報告的是黑棋勝率 (根據 config reportAnalysisWinratesAs = BLACK)
    // 取最佳著手的勝率作為當前局面勝率
    double blackWinrate = 0.5;
    if (engineResult.topMoves.isNotEmpty) {
      blackWinrate = engineResult.topMoves.first.winrate;
    }

    return AnalysisResult(
      topMoves: moves,
      winrate: [blackWinrate, 1.0 - blackWinrate, 0.0],
      ownership: null, // Native engine 不支援 ownership
      boardSize: board.boardSize,
      nextPlayer: board.nextPlayer,
    );
  }

  /// 釋放資源
  Future<void> dispose() async {
    await _engine.stop();
    _isInitialized = false;
  }
}

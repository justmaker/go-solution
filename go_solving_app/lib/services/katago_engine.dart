import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:path/path.dart' as p;
import '../models/board_state.dart';
import '../models/analysis_result.dart';

/// KataGo ONNX 推論引擎
class KataGoEngine {
  static const String _modelAssetPath = 'assets/models/katago_b6c64.onnx';
  static const int _binaryChannels = 22;
  static const int _globalFeatures = 19;

  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  OrtSession? _session;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// 初始化 ONNX Runtime 並載入模型
  Future<void> initialize() async {
    if (_isInitialized) return;

    // 從 assets 複製模型到臨時目錄
    final modelPath = await _copyModelToTemp();

    // 建立 ONNX Runtime Session
    _session = await _onnxRuntime.createSession(modelPath);
    _isInitialized = true;
  }

  /// 複製模型檔案到臨時目錄（每次都重新複製以確保使用最新模型）
  Future<String> _copyModelToTemp() async {
    final tempDir = await getTemporaryDirectory();
    final modelFile = File(p.join(tempDir.path, 'katago_b6c64.onnx'));

    // 每次都重新複製以確保使用最新版本
    final data = await rootBundle.load(_modelAssetPath);
    await modelFile.writeAsBytes(data.buffer.asUint8List());

    return modelFile.path;
  }

  static const int _modelBoardSize = 19;

  /// 分析棋盤狀態，回傳分析結果
  Future<AnalysisResult> analyze(BoardState board) async {
    if (!_isInitialized || _session == null) {
      throw StateError('KataGoEngine 尚未初始化，請先呼叫 initialize()');
    }

    final n = board.boardSize;

    // 編碼輸入特徵（始終填充到 19x19）
    final binaryInput = _encodeBinaryFeatures(board);
    final globalInput = _encodeGlobalFeatures(board);

    // 建立輸入張量（始終使用 19x19）
    final binaryTensor = await OrtValue.fromList(
      Float32List.fromList(binaryInput),
      [1, _binaryChannels, _modelBoardSize, _modelBoardSize],
    );
    final globalTensor = await OrtValue.fromList(
      Float32List.fromList(globalInput),
      [1, _globalFeatures],
    );

    try {
      // 執行推論
      print('[KataGo] Running inference for ${n}x$n board (padded to ${_modelBoardSize}x$_modelBoardSize)...');
      final outputs = await _session!.run({
        'input_binary': binaryTensor,
        'input_global': globalTensor,
      });

      // 解析輸出
      final result = await _parseOutputs(outputs, board);
      print('[KataGo] Top moves: ${result.topMoves.map((m) => '${m.position}=${m.probability}').toList()}');
      print('[KataGo] Winrate: ${result.winrate}');
      return result;
    } finally {
      await binaryTensor.dispose();
      await globalTensor.dispose();
    }
  }

  /// 編碼 22 個二進位空間特徵平面 (KataGo V7 format)
  /// Shape: [1, 22, boardSize, boardSize]
  ///
  /// Ch 0:  棋盤 mask（所有合法位置 = 1）
  /// Ch 1:  己方棋子（當前玩家）
  /// Ch 2:  對方棋子
  /// Ch 3:  氣數 = 1 的棋子（叫吃）
  /// Ch 4:  氣數 = 2 的棋子
  /// Ch 5:  氣數 = 3 的棋子
  /// Ch 6-8: 劫/還棋（簡化版設為 0）
  /// Ch 9-13: 過去 1-5 手位置
  /// Ch 14-21: 梯子/地域/還棋（簡化版設為 0）
  List<double> _encodeBinaryFeatures(BoardState board) {
    final n = board.boardSize;
    final m = _modelBoardSize; // 始終使用 19x19
    final features = List<double>.filled(_binaryChannels * m * m, 0.0);

    final currentPlayer = board.nextPlayer;
    final opponent = currentPlayer.opponent;

    // 計算每個棋群的氣數
    final libertyMap = _computeLiberties(board);

    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        final stone = board.grid[r][c];
        final idx = r * m + c; // 使用 m (19) 作為行寬

        // Ch 0: 棋盤 mask（只有棋盤範圍內的位置設為 1）
        features[0 * m * m + idx] = 1.0;

        // Ch 1: 己方棋子
        if (stone == currentPlayer) {
          features[1 * m * m + idx] = 1.0;
        }
        // Ch 2: 對方棋子
        if (stone == opponent) {
          features[2 * m * m + idx] = 1.0;
        }

        // Ch 3-5: 氣數特徵（1, 2, 3 氣）
        if (stone != StoneColor.empty) {
          final libs = libertyMap[r][c];
          if (libs == 1) features[3 * m * m + idx] = 1.0;
          if (libs == 2) features[4 * m * m + idx] = 1.0;
          if (libs == 3) features[5 * m * m + idx] = 1.0;
        }
      }
    }

    // Ch 9-13: 過去 1-5 手位置
    for (int h = 0; h < 5 && h < board.moveHistory.length; h++) {
      final move = board.moveHistory[board.moveHistory.length - 1 - h];
      final idx = move.row * m + move.col;
      features[(9 + h) * m * m + idx] = 1.0;
    }

    // Ch 6-8, 14-21: 其餘特徵設為 0（劫、梯子、地域、還棋等）

    return features;
  }

  /// 計算棋盤上每個棋子所屬棋群的氣數
  List<List<int>> _computeLiberties(BoardState board) {
    final n = board.boardSize;
    final liberties = List.generate(n, (_) => List.filled(n, 0));
    final visited = List.generate(n, (_) => List.filled(n, false));

    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        if (board.grid[r][c] != StoneColor.empty && !visited[r][c]) {
          // BFS 找出整個棋群
          final color = board.grid[r][c];
          final group = <(int, int)>[];
          final libSet = <(int, int)>{};
          final queue = <(int, int)>[(r, c)];
          visited[r][c] = true;

          while (queue.isNotEmpty) {
            final (gr, gc) = queue.removeLast();
            group.add((gr, gc));
            for (final (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]) {
              final nr = gr + dr, nc = gc + dc;
              if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
              if (board.grid[nr][nc] == StoneColor.empty) {
                libSet.add((nr, nc));
              } else if (board.grid[nr][nc] == color && !visited[nr][nc]) {
                visited[nr][nc] = true;
                queue.add((nr, nc));
              }
            }
          }

          // 設定群組中每個棋子的氣數
          final libCount = libSet.length;
          for (final (gr, gc) in group) {
            liberties[gr][gc] = libCount;
          }
        }
      }
    }
    return liberties;
  }

  /// 編碼 19 個全域特徵 (KataGo V7 format)
  /// Shape: [1, 19]
  ///
  /// G0-4:  過去 1-5 手是否為 pass
  /// G5:    selfKomi / 20（從當前玩家視角的貼目）
  /// G6-18: 規則設定（簡化版大多設為 0）
  List<double> _encodeGlobalFeatures(BoardState board) {
    final features = List<double>.filled(_globalFeatures, 0.0);

    // G0-4: 過去 1-5 手是否為 pass（目前不支援 pass，全部設為 0）

    // G5: selfKomi / 20（從當前玩家角度）
    // 貼目給白方，所以黑方視角為負，白方視角為正
    final selfKomi = board.nextPlayer == StoneColor.black
        ? -board.komi
        : board.komi;
    features[5] = selfKomi / 20.0;

    // G6-18: 規則設定（預設中國規則、簡單劫、不允許自殺）
    // 全部保持 0（對應簡單劫、面積計分、無稅、無還棋、無按鈕）

    return features;
  }

  /// 解析模型輸出
  /// 模型輸出 policy 為 19*19+1=362 維，需要映射回實際棋盤大小
  Future<AnalysisResult> _parseOutputs(
    Map<String, OrtValue> outputs,
    BoardState board,
  ) async {
    final n = board.boardSize;
    final m = _modelBoardSize;

    // 解析 policy（著手機率）
    final policyTensor = outputs['output_policy'];
    List<double> policyData = [];
    if (policyTensor != null) {
      final rawData = await policyTensor.asFlattenedList();
      policyData = rawData.cast<num>().map((e) => e.toDouble()).toList();
      await policyTensor.dispose();
    }

    // 從 19x19 policy 中提取實際棋盤範圍的 logits
    // policy shape: [m*m + 1]，其中最後一個是 pass
    final boardLogits = <MapEntry<int, double>>[];
    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        final modelIdx = r * m + c; // 在 19x19 格式中的位置
        final boardIdx = r * n + c; // 在 NxN 格式中的位置
        if (modelIdx < policyData.length) {
          boardLogits.add(MapEntry(boardIdx, policyData[modelIdx]));
        }
      }
    }

    // 對棋盤範圍內的 logits 做 softmax
    final logitValues = boardLogits.map((e) => e.value).toList();
    final probs = _softmax(logitValues);

    // 取 top-N 最佳著手（過濾已有棋子的位置）
    final moves = <MoveSuggestion>[];
    final indexed = List.generate(
      boardLogits.length,
      (i) => MapEntry(boardLogits[i].key, probs[i]),
    );
    indexed.sort((a, b) => b.value.compareTo(a.value));

    var rank = 1;
    for (final entry in indexed) {
      if (rank > 5) break;
      final idx = entry.key;
      final row = idx ~/ n;
      final col = idx % n;
      // 跳過已有棋子的位置
      if (row < board.boardSize && col < board.boardSize &&
          board.grid[row][col] != StoneColor.empty) {
        continue;
      }
      moves.add(MoveSuggestion(
        position: BoardPosition(row, col),
        probability: entry.value,
        rank: rank,
      ));
      rank++;
    }

    // 解析 value（勝率）
    final valueTensor = outputs['output_value'];
    List<double> winrate = [0.5, 0.5, 0.0];
    if (valueTensor != null) {
      final rawData = await valueTensor.asFlattenedList();
      final valueData = rawData.cast<num>().map((e) => e.toDouble()).toList();
      if (valueData.length >= 3) {
        winrate = _softmax(valueData.sublist(0, 3));
      }
      await valueTensor.dispose();
    }

    // 解析 ownership（地域歸屬）- 從 19x19 映射到 NxN
    List<List<double>>? ownership;
    final ownershipTensor = outputs['output_ownership'];
    if (ownershipTensor != null) {
      final rawData = await ownershipTensor.asFlattenedList();
      final ownershipData = rawData.cast<num>().map((e) => e.toDouble()).toList();
      if (ownershipData.length >= m * m) {
        ownership = List.generate(
          n,
          (r) => List.generate(
            n,
            (c) => ownershipData[r * m + c], // 使用 m 作為行寬
          ),
        );
      }
      await ownershipTensor.dispose();
    }

    // 清理未使用的輸出張量
    for (final entry in outputs.entries) {
      if (!['output_policy', 'output_value', 'output_ownership']
          .contains(entry.key)) {
        await entry.value.dispose();
      }
    }

    return AnalysisResult(
      topMoves: moves,
      winrate: winrate,
      ownership: ownership,
      boardSize: n,
      nextPlayer: board.nextPlayer,
    );
  }

  /// Softmax 函數
  List<double> _softmax(List<double> values) {
    if (values.isEmpty) return [];
    final maxVal = values.reduce(max);
    final exps = values.map((v) => exp(v - maxVal)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  /// 釋放資源
  Future<void> dispose() async {
    await _session?.close();
    _session = null;
    _isInitialized = false;
  }
}
